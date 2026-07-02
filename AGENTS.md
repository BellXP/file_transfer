# file_transfer

## Cursor Cloud specific instructions

This repository is a **file-dump / scratch repository**, not a runnable software product.
It contains loose, unrelated assets rather than an application:

- `bench.py` — a standalone benchmark script for speculative-decoding LLM inference.
- `aime25.jsonl`, `aime25_mtp_8192_tmp0.jsonl` — AIME 2025 benchmark prompts and result data.
- `efficientvit_cls_bs_r224.hef`, `levit_c_128s.hef` — compiled Hailo edge-AI model binaries.
- `hf_note` — a one-line Hugging Face dataset download command.
- `declaration_form.pdf`, `sc.pptx` — unrelated documents.

There is **no application/service to run**, **no automated tests**, **no build system**, and **no
dependency manifest** (no `requirements.txt`, `pyproject.toml`, `package.json`, `Dockerfile`, etc.).
So there is nothing to install on startup; the update script is intentionally a no-op verification.

### Working on `bench.py`
`bench.py` is **not runnable as-is** in this environment. To run it you would need all of:
- A CUDA GPU — the device is hardcoded to `torch.device("cuda:0")` (this VM has no GPU).
- A local `model` module (`from model import DFlashDraftModel, sample, extract_context_feature`)
  that **does not exist** in this repo.
- Local model weights under `models_from_hf/` (`Qwen3-8B`, a `DFlash` draft model) that are absent.
- The `torch` and `transformers` packages (not installed by default; install them only if you
  actually need to edit/type-check `bench.py`).

### What you can verify without any dependencies
- Syntax/lint check: `python3 -m py_compile bench.py`
- Data integrity: the `.jsonl` files are line-delimited JSON (30 records each) and parse cleanly.
