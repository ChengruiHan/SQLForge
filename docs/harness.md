# Bounded harness search

Three declarative modules surround fixed model weights:

| Module | Allowed changes |
|---|---|
| Prompt guidance | Append at most 2048 characters to the mandatory H0 instruction |
| Feedback presentation | Raw/status-first rendering;0–5 preview rows |
| History/state | Full/compact older observations; retain 1–3 recent observations; public state |

Budgets, permissions, scoring, actions and model weights are fixed. Raw observations remain in traces. Compaction retains model actions, marks omissions and does not invent semantic correctness. No module forces an extra tool call or final submission.

`partition_development` splits dev 128 into 64 search and 64 selection tasks, balances database/category strata, and keeps near-duplicate question components together. The search model is frozen GRPO, chosen from previously observed development scores, not test performance.

Search includes H0, three static candidates, and at most four DeepSeek Flash proposals. A failed, duplicate or invalid proposal consumes its slot. The external proposer gets only search traces/scores: never reference answers, hidden test code, selection feedback or test trajectories. Its serialized UTF-8 input plus framing allowance is capped at 8192 bytes; this is a conservative input bound, not an exact provider tokenizer count. Output is capped at 4096 tokens including reasoning. API credentials are read from `SQLFORGE_API_KEY`; running search can incur four external API calls.

```bash
export SQLFORGE_API_KEY='your-key'  # configure locally; never commit the value
python scripts/search_harness.py --run runs/harness --tasks data/tasks.jsonl \
  --model models/Qwen3-4B-Instruct-2507 --arms runs/rl/arms.json --audit data/audit.json
```

The top two non-H0 candidates and H0 are evaluated on the same 64 selection tasks with sampling seeds 42 and 43. A candidate must have strictly higher mean success and no per-seed regression; ties use generated tokens then name. Otherwise retain H0. Raw receives a transfer comparison on the selection tasks; when H0 wins, only the Raw/H0 baseline is needed.

The public runner stops after saving `selected-harness.json` and `FINAL_LOCK.json`; it does not automatically touch test. Explicitly freeze/evaluate that harness with the evaluation CLI:

```bash
python scripts/evaluate.py --freeze --tasks data/tasks.jsonl --model models/Qwen3-4B-Instruct-2507 \
  --arms runs/rl/arms.json --harness runs/harness/selected-harness.json --output runs/HARNESS_TEST_LOCK.json
```

Use the same `--harness` and `--lock` with `--split test` if a new test comparison is intended. In the recorded experiment H0 won, so historical H0 test generations were reused and uniformly rescored instead of generating another test sample. Do not present the reused H0/selected columns as independent conditions or evidence of a new harness's zero effect.

The existing dev/test had already been used. This is a retrospective project experiment with an external LLM proposing configuration changes, not autonomous recursive model-weight improvement.
