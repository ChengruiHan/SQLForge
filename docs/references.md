# References and attribution

| Project | What SQLForge uses or follows |
|---|---|
| [verl](https://github.com/verl-project/verl) | Native Trainer, AgentLoop API, GRPO/GSPO/RLOO estimators and objectives; pinned to v0.6.1 commit d62da4950573d7a4b7ef2362337952e7ab59e78d |
| [BIRD-RL](https://github.com/bird-bench/BIRD-RL) | Pinned original SQL evaluator and executable verification; upstream code fetched separately with its license |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | Organizational reference for configuration, agent runtime, docs and regression tests; no SWE-agent implementation is copied |
| [SIX-GYM-SQLite](https://huggingface.co/datasets/birdsql/six-gym-sqlite) | Source Query repair tasks and SQLite databases; obtained separately |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Original model and tokenizer; weights obtained separately |

Repository organization follows the separation of configuration, runnable examples, runtime and tests visible in these projects. SQLForge does not claim to reproduce the full BIRD-RL training recipe or SWE-agent benchmark results. Its H0 action protocol and bounded search are project implementations.

The MIT license covers SQLForge-owned source. Fetched upstream code, models and datasets retain their original licenses. No upstream source tree, dataset database or model checkpoint is republished here. The `fetch_bird_evaluator.py` helper records the upstream license and file identities alongside downloaded code.
