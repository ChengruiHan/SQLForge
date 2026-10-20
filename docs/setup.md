# Setup

The recorded GPU runs used Python 3.12, torch 2.8.0+cu128, vLLM 0.11.0, Transformers 4.57.1 and PEFT 0.18.1. A single exposed 32GB GPU completed the formal workflow; this does not guarantee fit on every GPU or driver. The evaluation stack and training stack may be separate virtual environments.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gpu.txt
python -m pip install -e '.[dev]'
mkdir -p vendor
git clone https://github.com/verl-project/verl.git vendor/verl-v0.6.1
git -C vendor/verl-v0.6.1 checkout d62da4950573d7a4b7ef2362337952e7ab59e78d
python -m pip install -e vendor/verl-v0.6.1
python -m pip check
python scripts/fetch_bird_evaluator.py
```

The evaluator fetcher pins BIRD-RL commit `b26c4285ef69dfb6c096b076a7499018c6b370ca` and saves licenses/file hashes. No upstream source tree is vendored into this repository.

Download the original `Qwen/Qwen3-4B-Instruct-2507` model into `models/Qwen3-4B-Instruct-2507` using your preferred Hugging Face client. Model weights are not in Git. Preparation hashes the local model files; every arm must use the same unquantized base and tokenizer.

The training entry point applies one checked upstream patch: honor per-action `max_tokens=512` in the asynchronous vLLM server. It does not alter verl's objectives or advantage estimators. `sqlforge.verl_padding` trims inactive edge padding for single-trajectory microbatches and delegates the active computation to the native actor. Run the numerical checks on the intended GPU stack before training.

The GPU environment has not been freshly installed or trained during publication. Dependency changes and different GPU kernels can change numerical results. `python -m pip check`, objective checks and one-step smokes are required setup validation.
