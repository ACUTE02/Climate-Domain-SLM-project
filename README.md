# 🌦️ Climate-Domain-SLM

**A small language model, tuned on Indian climate and disaster-management documents, that answers a question from a given passage or says it cannot.**

![Python](https://img.shields.io/badge/Python-3.11_%7C_3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.19-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-FFD21E?logo=huggingface&logoColor=black)
![vLLM](https://img.shields.io/badge/vLLM-0.32-30A2FF)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/tests-368_passed-brightgreen)

> It answers from the passage, or it says it cannot. The dataset is small and the error margins are wide; this README states them openly.

## 📖 Overview

A small language model (Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct) is tuned on questions about Indian climate and disaster-management government documents (drought, heat waves, agro-advisories). Given a passage and a question it should answer from the passage only, and when the passage does not contain the answer it should reply exactly: `I cannot answer this from the provided context.` Tuning is supervised fine-tuning (SFT) with LoRA, followed by DPO experiments; training runs are tracked with MLflow, and the tuned models are served with vLLM behind a FastAPI service.

The dataset is small (252 questions) and the results have wide error margins. Read the **Limits** section before the numbers.

## ✨ Features

- **Grounded answers with refusals.** Some training questions are deliberately paired with a passage that does *not* contain the answer, so the model learns to refuse with one exact sentence.
- **SFT with LoRA**, a small grid on the 0.5B model (learning rate, LoRA against 4-bit QLoRA, three seeds for the winner) and one run on the 1.5B model.
- **DPO experiments** (v1 and v2) on top of the SFT model, with the SFT model as the control in every table.
- **One evaluator for every model**: greedy decoding, answerable token F1, strict and loose correct-refusal counts, false refusals, number faithfulness, 95% bootstrap intervals.
- **Selection on validation only**, by a rule written before the runs; the test split is used once per frozen winner.
- **FastAPI service** that serves one model per process:
  - `POST /answer` takes `{"context": "...", "question": "...", "max_new_tokens": 256}` (`max_new_tokens` is optional, 1 to 256) and returns `answer`, `refused` (true if the answer is the exact refusal text), `model`, token counts, `latency_ms` and a `request_id`.
  - `GET /health` returns whether the model is loaded, the backend, the model, the library versions, the start-up time and batch statistics.
  - Error codes: **422** for an empty or missing context or question, an unknown field, a `max_new_tokens` outside 1 to 256, or a model name this process does not serve; **413** if the prompt is longer than 1792 tokens (the context is never cut silently); **503** if the backend fails or the model is not loaded (a short message, no stack trace).
  - The service logs the request id, model, latency, token counts and whether it refused. **It never logs the context, the question or the answer.** There is no CORS middleware and it binds to 127.0.0.1 by default.
  - Backends, chosen with `PHASE5_BACKEND`: `vllm` (vLLM with the LoRA adapter, Linux and a GPU), `hf` (transformers + peft with the same code as the evaluator), `fake` (a constant answer, no model, no GPU, used by the tests).
  - The prompt is built by one shared function (`serving/prompting.py`) that produces exactly the training and evaluation prompt, and decoding is greedy, as in the evaluator.
  - Requests that arrive together are collected into one batch (5 ms window) in front of the engine, the same way for `vllm` and `hf`.
- **Parity check** that proves the service answers like the evaluator, and a **benchmark** of vLLM against the `hf` backend.

## 🏗️ Architecture

```
  six government documents
            |
            v
 +----------------------+     181 chunks -> question/answer items, refusal items,
 | 1. DATA              |     LLM verification + human spot-check
 |    156 / 53 / 43     |     (train / validation / test)
 +----------------------+
            |
            v
 +----------------------+     LoRA on Qwen2.5-0.5B (grid, 3 seeds) and Qwen2.5-1.5B;
 | 2. SFT               |     checkpoint chosen on validation only
 +----------------------+
            |
            v
 +----------------------+     v1 failed on validation; v2 (new pairs + NLL term)
 | 3. DPO (0.5B)        |     gave one frozen winner
 +----------------------+
            |
            v
 +----------------------+     one evaluator, greedy decoding, bootstrap intervals,
 | 4. EVALUATION        |     test split used once per frozen winner
 +----------------------+
            |
            v
 +----------------------+     FastAPI: POST /answer, GET /health
 | 5. SERVING           |     backends: vllm | hf | fake; parity check vs evaluator
 +----------------------+
            |
            v
 +----------------------+     vLLM vs hf on a free Colab T4:
 | 6. BENCHMARK         |     latency p50 / p95, tokens/s, start-up, GPU memory
 +----------------------+
```

## 📁 Project Structure

```
Climate-Domain-SLM/
├── climate_slm/          # data building, the evaluator, SFT and DPO training code
│   ├── data/             #   chunk cleaning, splits, question building, verification checks
│   ├── evaluation/       #   evaluator, metrics, scoring, result collection
│   └── training/         #   SFT, DPO v1 / v2, grids, selection, final test, MLflow logging
├── serving/              # FastAPI service, backends (vllm, hf, fake), micro-batching, client helpers
├── scripts/              # parity check, benchmark, report and Colab-bundle scripts, small one-off tools
├── configs/              # one yaml per phase (phase1 ... phase5)
├── data/
│   └── instruction/      # the final train / validation / test files and the DPO pair files
├── notebooks/            # Colab notebook that runs every GPU step
├── tests/                # pytest tests (no GPU needed; the real models are replaced by fakes)
├── requirements.txt          # laptop: data, tests, tables
├── requirements-serve.txt    # laptop: the service and its tests
├── requirements-gpu.txt      # Colab: training and evaluation
└── requirements-local.txt    # local NVIDIA GPU: the hf backend
```

## 🧰 Tech Stack

| area | tools |
|---|---|
| models | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct |
| training | PyTorch, Hugging Face Transformers, PEFT (LoRA, 4-bit QLoRA via bitsandbytes), custom DPO loss |
| tracking | MLflow (local file store) |
| evaluation | own evaluator, rouge-score, bootstrap intervals |
| serving | FastAPI, uvicorn, pydantic, vLLM, httpx |
| compute | free Colab T4 for all GPU work; laptop for data, tests and tables |
| testing | pytest |

## 📊 Measured Results

All numbers are on the **test split** unless a row says otherwise: 43 items, 32 answerable and 11 refusal items. "Token F1" is measured on the answerable items. "False refusals" = answerable questions the model refused. "Strict refusals" = refusal items where the model gave exactly the right refusal. Intervals are 95% bootstrap intervals over the items.

### Model quality

| model | token F1 (95% interval) | false refusals | strict correct refusals |
|---|---|---|---|
| 0.5B, untuned | 0.542 [0.421, 0.663] | 0/32 | 0/11 |
| 0.5B, untuned, "strict sentence" prompt | 0.670 [0.564, 0.771] | 0/32 | 0/11 |
| 0.5B, untuned, few-shot prompt | 0.648 [0.545, 0.740] | 0/32 | 0/11 |
| 0.5B SFT | 0.617 [0.477, 0.743] | 4/32 | 9/11 |
| 0.5B DPO v2 | 0.666 [0.536, 0.784] | 2/32 | 8/11 |
| 1.5B, untuned | 0.656 [0.579, 0.729] | 0/32 | 6/11 |
| **1.5B SFT** | **0.791 [0.688, 0.882]** | **0/32** | **11/11** |
| 0.5B DPO v1 | not run on test | | |

- **DPO v1** failed on validation: none of its 12 candidates passed the filter (strict correct refusals 0 to 3 of 11, the rule needed at least 7). Their false refusals were 0/42, but only because the models had almost stopped refusing. The test split was never used for v1.
- **0.5B SFT vs 0.5B DPO v2**: DPO v2 has fewer false refusals (2/32 against 4/32), one refusal item less (8/11 against 9/11) and a higher F1 (0.666 against 0.617). **All intervals overlap, so none of these differences is established.** The project does not claim that DPO is better than SFT.
- **1.5B SFT** has the best or equal-best point value in every column. Its F1 interval overlaps those of the tuned 0.5B models; it does not overlap the interval of the untuned 0.5B model with the default prompt.
- The untuned 0.5B model with a stricter prompt scores a higher F1 than the SFT model (intervals overlap); what tuning clearly adds is the refusal behaviour (0/11 before, 8/11 to 11/11 after).
- Every tuned model still has one or two outputs flagged as repeating (0.5B SFT 2, 0.5B DPO v2 1, 1.5B SFT 1 on test).

### Serving: does the service answer like the evaluator? (parity, 53 validation prompts)

Each model was run through the live vLLM service, one prompt at a time, and scored with the evaluator's own scoring code. The rule was fixed beforehand: stop if refusal counts move by more than one item or F1 by more than 0.03. It was not triggered.

| model | identical outputs | token F1 (evaluator → service) | strict refusals | false refusals |
|---|---|---|---|---|
| 0.5B SFT | 52/53 | 0.6154 → 0.6167 | 8/11 → 8/11 | 3/42 → 3/42 |
| 0.5B DPO v2 | 52/53 | 0.6360 → 0.6420 | 8/11 → 8/11 | 0/42 → 0/42 |
| 1.5B SFT | 53/53 | 0.7865 → 0.7865 | 9/11 → 9/11 | 0/42 → 0/42 |

The one differing output in each 0.5B model was not investigated (a small numerical difference between the two engines in half precision is the likely cause, not proven).

### Serving: vLLM against the `hf` backend (free Colab T4)

Client-side latency of one request (queueing included) and output tokens per second. 53 validation prompts per repeat, 3 repeats, median of the repeats shown; concurrency = requests in flight at once. No request failed. The test split was not used.

| model | concurrency | vLLM p50 / p95 latency (ms) | vLLM tokens/s | hf p50 / p95 latency (ms) | hf tokens/s |
|---|---|---|---|---|---|
| 0.5B SFT | 1 | 577 / 1638 | 49.2 | 1891 / 5326 | 14.9 |
| 0.5B SFT | 16 | 3076 / 3330 | 207.2 | 8925 / 15097 | 50.8 |
| 0.5B DPO v2 | 1 | 669 / 1703 | 49.6 | 2299 / 5753 | 14.3 |
| 0.5B DPO v2 | 16 | 3139 / 4623 | 198.7 | 11481 / 19546 | 47.4 |
| 1.5B SFT | 1 | 1201 / 2837 | 26.1 | 2626 / 5962 | 11.9 |
| 1.5B SFT | 16 | 6012 / 6716 | 96.5 | 14583 / 20287 | 35.0 |

The min to max ranges over the repeats of vLLM and hf do not overlap in any setting (concurrency 4 was measured too and behaves the same way). On this T4, with this service wrapper, **vLLM is about 3 to 4 times faster than `hf` for the 0.5B models (output tokens per second 3.2 to 4.2 times higher) and about 2 to 3 times faster for the 1.5B model (2.2 to 2.8 times)**.

**Start-up time** (process start until the model is loaded; reported separately from the request numbers):

| backend | 0.5B SFT | 0.5B DPO v2 | 1.5B SFT |
|---|---|---|---|
| vLLM | 82 s | 80 s | 92 s |
| hf | 12 s | 14 s | 22 s |

**GPU memory** (whole GPU as seen by `nvidia-smi`, peak while serving; the T4 has 15,360 MiB): vLLM 12,543 to 12,743 MiB for every model, hf 1,295 to 2,033 MiB for the 0.5B models and 3,499 to 4,595 MiB for the 1.5B model, growing with concurrency. The vLLM number is a **reserved pool** (vLLM claims a fixed share of the GPU when it starts, 85% here), not the memory the model needs. Do not read it as "vLLM needs 12 GB".

## 🚀 Run it locally

> On the author's laptop (RTX 3050 Laptop GPU, 4 GB, Windows, hf backend, one model per process), all three models were started and given two test requests each: one answerable question and one question the passage cannot answer. All three models answered the first from the passage and replied with the exact refusal sentence to the second (refused = true). The 1.5B model loaded and ran on the 4 GB card without a CUDA out-of-memory error. This is a two-request smoke test, not a benchmark and not an accuracy measurement. The vLLM backend was not tried on the laptop, because vLLM does not run on native Windows.

**What works where.** vLLM does not run on native Windows, so the local path on Windows is the `hf` backend. (The vLLM backend was run on Linux, on Colab.) The tuned models are LoRA adapters on top of a base model.

**1. Without a GPU: fake backend.** Needs only the lightweight environment:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-serve.txt
pytest tests
```

(`requirements-serve.txt` also lists `mlflow`; only the results-logging script needs it, the service does not.)

Start the service with the fake backend (PowerShell; in bash use `PHASE5_BACKEND=fake PHASE5_PORT=8000 python -m serving.app`):

```
$env:PHASE5_BACKEND = "fake"
python -m serving.app
```

**2. With an NVIDIA GPU: `hf` backend.** Use a separate environment, because the project `.venv` has no torch on purpose:

```
python -m venv .venv-serve
.venv-serve\Scripts\activate
# install the CUDA build of torch first, see the comments in requirements-local.txt
pip install -r requirements-local.txt
```

`requirements-local.txt` pins the versions the Colab run used (torch 2.13.0, transformers 5.19.0, peft 0.21.2). These are Colab versions; the laptop versions were not tested. The adapters are looked up by `configs/phase5.yaml` in these folders (relative to the repo root):

| `PHASE5_MODEL` | adapter folder | base model |
|---|---|---|
| `sft_0.5b` | `outputs/adapters/lora_fp16_lr2e-4/e2` | Qwen/Qwen2.5-0.5B-Instruct |
| `dpo2_0.5b` | `outputs/adapters_dpo2/dpo2_mix_a1_lr1e-5/e1` | Qwen/Qwen2.5-0.5B-Instruct |
| `sft_1.5b` | `outputs/adapters/q15_lora_fp16/e3` | Qwen/Qwen2.5-1.5B-Instruct |

```
$env:PHASE5_BACKEND = "hf"
$env:PHASE5_MODEL = "sft_0.5b"
python -m serving.app
```

(bash: `PHASE5_BACKEND=hf PHASE5_MODEL=sft_0.5b python -m serving.app`. `PHASE5_PORT` changes the port, default 8000.)

- **The base model is downloaded from the Hugging Face Hub on the first start**, so the first start needs internet and takes a while.
- **The adapters are not in this repository** (`outputs/` is not published). The published material is the code, the configs, the data files and the notebook. A reader can train the adapters again with `notebooks/colab_runner.ipynb` (a free Colab T4 is enough: one 0.5B LoRA run took about 2 minutes, the 1.5B run about 6 minutes); that gives adapters trained the same way, not necessarily identical files.
- GPU memory: the T4 numbers above are whole-GPU readings. The 1.5B model with the `hf` backend used 3,299 MiB right after loading and up to 4,595 MiB at concurrency 16 on the T4, and the laptop GPU has 4,096 MiB (`nvidia-smi`), so start with a 0.5B model. This was not tried.

**3. Ask a question.** (works with every backend; with `fake` the answer is a constant text)

```python
import httpx

r = httpx.post("http://127.0.0.1:8000/answer", timeout=120, json={
    "context": "A heat wave is declared when the maximum temperature is at least 40 C in the plains.",
    "question": "At what temperature is a heat wave declared in the plains?",
})
print(r.status_code, r.json())          # 200, with "answer", "refused", "latency_ms", ...
print(httpx.get("http://127.0.0.1:8000/health").json())
```

## 🧪 Testing

```
pytest tests
```

368 tests pass and 36 are skipped, with no GPU and no model. The real models are replaced by fakes. A test is skipped when it needs a file that is not in this repository: the saved result files and adapters in `outputs/`, or the working files of the dataset build (draft candidates, spot-check sheets, the raw chunk snapshot); the skip message names the missing file.

## ⚠️ Limits

- **A question type was dropped.** A third kind of preference pair for DPO (a reference answer against a wrong answer of the untuned model) was built, all 54 examples were read by hand, many turned out to be short but correct answers, and the type was dropped. Its code is still in the repo, switched off.
- **The dataset is small.** 156 train, 53 validation and 43 test items, and the test split has only **11 refusal items**. The confidence intervals are wide. The differences between the 0.5B SFT and the 0.5B DPO v2 overlap and are **not established**; a single refusal item moves a refusal score by 9 points.
- **The data was written and checked by the same model family.** Drafts, test drafts and the verification pass all come from LLMs of one family; only a spot-check was done by a human, and refusal items are the weakest part of that check. In the training-pool sample the human and the AI verifier disagreed on 6 of 30 rows (0.20), on 2 of 12 test rows (0.17); the stop rule had a limit of 0.20, and it was passed on purpose by the project reviewer. This is not a fully human-reviewed dataset. One source (the NDMA drought guidelines) supplies 109 of the 252 items.
- **DPO v1 failed**: the models copied the context instead of refusing. **DPO v2 changed several things at once** (balanced pairs, an NLL term, new rejected texts), so this project cannot say which change fixed it. 78 of the 80 rejected texts came from the untuned model, not from the SFT model (off-policy). The weight of the NLL term (0.5 against 1) made no visible difference. **36 validation candidates were tried in total for the DPO choice, so the validation set was not a clean check for it.**
- **The best model is 1.5B SFT.** DPO v2 was only done for the 0.5B model.
- **The benchmark is narrow.** It ran on a free Colab T4, with micro-batching in front of vLLM's offline engine (not vLLM's own asynchronous engine), on 53 validation prompts, 3 repeats. The `hf` backend used a maximum batch of 8, vLLM 16. vLLM's GPU memory is a reserved pool, and vLLM counts one end-of-answer token more per answer than the `hf` backend. The strongest sentence this supports: *vLLM is about 3 to 4 times faster than `hf` on this T4 with this service wrapper (0.5B models; about 2 to 3 times for the 1.5B model).* It says nothing about production load, and it is not the best case for vLLM.
- **SFT against DPO latency in vLLM** differs because the answers differ in length, not because one model is faster: the tokens per second are almost the same.
- **Training logs are local.** The MLflow runs (`phase3_sft`, `phase4_dpo`, `phase4b_dpo2`, `phase5_serving`) and all result files are in a local, unpublished `outputs/` folder; this README is the public record of the numbers.
- **Serving environment of the Colab runs:** Python 3.13.15, vLLM 0.32.0, torch 2.13.0, transformers 5.19.0, peft 0.21.2, FastAPI 0.136.3, pydantic 2.14.1, uvicorn 0.54.0, on a Tesla T4 (15 GB). The laptop tests ran with FastAPI 0.153.0 and Python 3.11.

## 📚 Data

The training text comes from six Indian government publications:

- NDMA National Disaster Management Guidelines: Management of Drought
- NDMA Heat Wave hazard page
- IMD FAQ on Heat Wave
- IMD Standard Operating Procedure for Agromet Advisory Services
- ICAR-ATARI Jodhpur Agro-Advisory Services
- NIH Roorkee, Standardized Precipitation Index methodology

The files in data/instruction are the published versions: the generator label was renamed and contact details that appeared inside the source PDFs were replaced by placeholders, so their hashes differ from the training copies. The .sha256 files refer to the published files.
