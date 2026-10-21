"""Backends behind one interface: `fake` (tests), `hf` (transformers, through the evaluator's own HFBackend: same loading, same chat
template, same greedy decoding) and `vllm` (vLLM with the LoRA adapter, greedy).

A backend serves ONE model (an adapter on its base) per process and decodes greedily with repetition_penalty 1.0 (the evaluator's
`greedy` mode). The two real backends hand concurrent requests to one worker thread through serving/batching.MicroBatcher (the
engine is called from that thread only). torch / transformers / vllm are imported only inside the real backends' load(), never at
module level, so the laptop tests need none of them."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from climate_slm import chat
from serving.batching import MicroBatcher


@dataclass
class GenResult:
    text: str
    n_prompt_tokens: int
    n_new_tokens: int
    hit_max_new_tokens: bool = False


class BackendError(RuntimeError):
    """Any failure of a backend while answering. The API turns it into HTTP 503 with a short message."""


class Backend:
    name = "base"

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def loaded(self) -> bool:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        """Tokens of the chat prompt (template applied, generation prompt added)."""
        raise NotImplementedError

    def generate(self, messages: list[dict], max_new_tokens: int) -> GenResult:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "loaded": self.loaded}

    def close(self) -> None:
        """Stop worker threads (real backends)."""


def qwen_style_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """A stand-in chat template with the Qwen layout (used by the fake backend only; the real ones use the tokenizer's)."""
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class FakeBackend(Backend):
    """Deterministic backend for tests. `responder(messages) -> text`; tokens = whitespace words of the templated prompt."""
    name = "fake"

    def __init__(self, model_name: str = "fake-model", responder: Callable[[list[dict]], str] | None = None,
                 fail_with: Exception | None = None, loaded: bool = True):
        super().__init__(model_name)
        self.responder = responder or (lambda messages: "fake answer")
        self.fail_with = fail_with
        self._loaded = loaded
        self.calls: list[tuple[list[dict], int]] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return len(chat.prompt_text(qwen_style_template, messages).split())

    def generate(self, messages: list[dict], max_new_tokens: int) -> GenResult:
        self.calls.append((messages, max_new_tokens))
        if self.fail_with is not None:
            raise self.fail_with
        words = self.responder(messages).split()
        hit = len(words) > max_new_tokens
        words = words[:max_new_tokens]
        return GenResult(" ".join(words), self.count_prompt_tokens(messages), len(words), hit)


class _BatchedBackend(Backend):
    """Shared part of the two real backends: a MicroBatcher in front of the engine and a separate tokenizer (with a lock) for the
    token counting done in the request threads (a fast tokenizer must not be used from several threads at once)."""

    def __init__(self, model_name: str, adapter: str, base_model_id: str, window_ms: float, max_batch: int):
        super().__init__(model_name)
        self.adapter, self.base_model_id = adapter, base_model_id
        self.window_ms, self.max_batch = window_ms, max_batch
        self._batcher: MicroBatcher | None = None
        self._count_tok = None
        self._count_lock = threading.Lock()

    def _start(self) -> None:
        """Called at the end of load(), after the engine exists."""
        from transformers import AutoTokenizer
        self._count_tok = AutoTokenizer.from_pretrained(self.base_model_id)
        self._batcher = MicroBatcher(self._run_batch, self.max_batch, self.window_ms)

    def _run_batch(self, items: list[tuple[list[dict], int]]) -> list[GenResult]:
        raise NotImplementedError

    def _template(self, tok):
        return lambda msgs, gen: tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        with self._count_lock:
            text = chat.prompt_text(self._template(self._count_tok), messages)
            return len(self._count_tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages: list[dict], max_new_tokens: int) -> GenResult:
        if self._batcher is None:
            raise BackendError("model not loaded")
        return self._batcher.submit((messages, max_new_tokens)).result()

    def batch_stats(self) -> dict:
        return self._batcher.stats() if self._batcher else {}

    def close(self) -> None:
        if self._batcher is not None:
            self._batcher.close()

    def info(self) -> dict:
        return {**super().info(), "adapter": self.adapter, "base_model_id": self.base_model_id, "max_batch": self.max_batch,
                "batch_window_ms": self.window_ms, **self.batch_stats()}


class HFServingBackend(_BatchedBackend):
    """transformers backend = the evaluator's HFBackend (greedy, float16, same template, left padding). Requests that arrive together
    are one batch (up to the evaluator's own batch size); requests with a different max_new_tokens form separate calls.
    NOT run on the laptop (needs torch and a GPU)."""
    name = "hf"

    def __init__(self, model_name: str, adapter: str, base_model_id: str, eval_cfg: dict, window_ms: float = 5, max_batch: int = 8):
        super().__init__(model_name, adapter, base_model_id, window_ms, max_batch)
        self.eval_cfg = eval_cfg
        self._backend = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None and self._batcher is not None

    def load(self) -> None:
        from climate_slm.evaluation.evaluate import HFBackend
        self._backend = HFBackend(self.eval_cfg, self.adapter, "greedy", None, self.base_model_id)
        self._start()

    def _run_batch(self, items: list[tuple[list[dict], int]]) -> list[GenResult]:
        b, out = self._backend, [None] * len(items)
        for max_new in sorted({n for _, n in items}):
            idx = [i for i, (_, n) in enumerate(items) if n == max_new]
            old = b.params
            b.params = {**old, "max_new_tokens": max_new}
            try:
                gens = b.generate([items[i][0] for i in idx])
            finally:
                b.params = old
            for i, g in zip(idx, gens):
                out[i] = GenResult(g["text"], self._hf_prompt_tokens(items[i][0]), g["n_new_tokens"], bool(g["hit_max_new_tokens"]))
        return out

    def _hf_prompt_tokens(self, messages: list[dict]) -> int:
        text = self._backend._prompt_text(messages)                      # called from the worker thread only
        return len(self._backend.tok(text, add_special_tokens=False)["input_ids"])


class VLLMServingBackend(_BatchedBackend):
    """vLLM backend: the base model in half precision with the LoRA adapter as a vLLM LoRA request, greedy (temperature 0,
    repetition_penalty 1.0). Requests that arrive together are one `LLM.generate` call (offline engine, micro-batched: see
    serving/batching.py). n_new_tokens is what vLLM reports (it may include the end-of-turn token; the evaluator does not count it).
    NOT run on the laptop (needs vllm and a GPU); P5-0 showed on Colab that vLLM 0.32.0 serves these adapters."""
    name = "vllm"

    def __init__(self, model_name: str, adapter: str, base_model_id: str, vllm_cfg: dict, max_model_len: int,
                 window_ms: float = 5, max_batch: int = 16):
        super().__init__(model_name, adapter, base_model_id, window_ms, max_batch)
        self.vllm_cfg, self.max_model_len = vllm_cfg, max_model_len
        self._llm = self._lora = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None and self._batcher is not None

    def load(self) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        v = self.vllm_cfg
        self._llm = LLM(model=self.base_model_id, dtype=v["dtype"], max_model_len=self.max_model_len,
                        gpu_memory_utilization=v["gpu_memory_utilization"], enable_lora=v["enable_lora"],
                        max_lora_rank=v["max_lora_rank"])
        self._lora = LoRARequest(self.model_name, 1, self.adapter)
        self._start()

    def _run_batch(self, items: list[tuple[list[dict], int]]) -> list[GenResult]:
        from vllm import SamplingParams
        tok = self._llm.get_tokenizer()                                    # worker thread only
        prompts = [chat.prompt_text(self._template(tok), m) for m, _ in items]
        params = [SamplingParams(temperature=0.0, max_tokens=n, repetition_penalty=1.0) for _, n in items]
        outs = self._llm.generate(prompts, params, lora_request=self._lora, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenResult(c.text.strip(), len(o.prompt_token_ids), len(c.token_ids), c.finish_reason == "length"))
        return res

    def info(self) -> dict:
        return {**super().info(), "vllm": self.vllm_cfg}


BACKEND_NAMES = ("vllm", "hf", "fake")


def make_backend(name: str, model_key: str, cfg5: dict, eval_cfg: dict | None = None) -> Backend:
    """The backend `name` for the model entry `model_key` of configs/phase5.yaml."""
    if name not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {name!r}; choose from {list(BACKEND_NAMES)}")
    if model_key not in cfg5["models"]:
        raise ValueError(f"unknown model {model_key!r}; choose from {sorted(cfg5['models'])}")
    entry = cfg5["models"][model_key]
    if name == "fake":
        return FakeBackend(model_key)
    from climate_slm.config import REPO_ROOT
    adapter = str(REPO_ROOT / entry["adapter"])
    b = cfg5["batching"]
    if name == "hf":
        from climate_slm.config import load_eval_config
        return HFServingBackend(model_key, adapter, entry["base_model_id"], eval_cfg or load_eval_config(), b["window_ms"], b["max_batch"]["hf"])
    return VLLMServingBackend(model_key, adapter, entry["base_model_id"], cfg5["vllm"], cfg5["service"]["max_model_len"],
                              b["window_ms"], b["max_batch"]["vllm"])
