# Training and evaluation

Run from the repository root with the environment from `setup.md` activated. The three formal RL arms start from the original Qwen base, with identical zero-output LoRA initialization. There is no SFT prerequisite.

```bash
python scripts/prepare.py --run runs/rl --tasks data/tasks.jsonl \
  --model models/Qwen3-4B-Instruct-2507 --task-order results/task-order.json
python scripts/check_verl_h0.py --run runs/rl
python scripts/check_verl_padding.py --run runs/rl
for method in grpo gspo rloo; do
  python scripts/train.py --run runs/rl --method "$method" --compose-only
  python scripts/train.py --run runs/rl --method "$method" --smoke
  python scripts/train.py --run runs/rl --method "$method"
done
python scripts/export_adapters.py --run runs/rl
```

Preparation validates 256 train and 128 dev tasks, writes parquet prompts, saves `MODEL.json`, `DATA.json`, `audit.json`, and one common `initial-adapter`. Initial prompts are the same public H0 protocol used by the AgentLoop. Refuse to proceed when audits fail. The fixed historical IDs do not silently fall back to easier replacements.

Each training arm samples 4 sibling trajectories for each of 256 tasks, with 4 tasks per batch and 64 rollout iterations. Max 4 actions, 512 action tokens, 384 observation tokens and 8192 context tokens; SQL execution deadline 3 seconds. LR 1e-5, KL coefficient 0.02, LoRA rank 16/alpha 32. GRPO/GSPO use 2 optimization epochs per batch (128 updates); RLOO uses 1 (64 updates). Native algorithm details are recorded in `configs/rl-protocol.json` and fully composed overrides in each run.

Only generated action tokens receive the episode advantage. Public observations and inserted chat boundaries have mask 0. External reward is 1 iff all terminal tests pass; no learned step reward or shaping. GRPO/GSPO use differentiable k3 KL; RLOO places sampled sequence k1 KL in its internal return. Their external success metric remains identical.

The training dev 32 curve runs every 16 batches and does not choose the final model. The checkpoint is always the fixed final 256-task budget. `export_adapters.py` copies the final adapters without deleting training/checkpoint data.

```bash
python scripts/evaluate.py --tasks data/tasks.jsonl --model models/Qwen3-4B-Instruct-2507 \
  --arms runs/rl/arms.json --audit data/audit.json --output runs/dev --split dev
python scripts/evaluate.py --freeze --tasks data/tasks.jsonl \
  --model models/Qwen3-4B-Instruct-2507 --arms runs/rl/arms.json --output runs/TEST_LOCK.json
python scripts/evaluate.py --tasks data/tasks.jsonl --model models/Qwen3-4B-Instruct-2507 \
  --arms runs/rl/arms.json --audit data/audit.json --output runs/test --split test --lock runs/TEST_LOCK.json
```

`--mode single` and `--mode independent4` are development baseline options. The latter uses public execution validity/modal output, not hidden-test oracle selection. Formal model evaluation uses temperature 0.7, not greedy. Lock files include base/adapter/config/task/harness identities. Cached generations are retained before scoring; infrastructure failure must not cause favorable resampling.

**Scoring exception:** the historical TRAIN_4648 test timeout required a separately documented index on a scoring-only copy. The portable evaluator does not silently add this index. A fresh original-scorer run can stop at that task. Preserve its cached answer, document any scorer change and apply it to all comparison arms; see `results.md`. Reproducing the published amended score requires this explicit treatment, not treating timeout as reward 0.

**Publication validation:** CPU tests and reported-score reconstruction are executable without data or GPU. The entry-point cleanup was not followed by a new GPU experiment; use the smoke checks above before a full run. Original source hashes and score provenance remain in `results/source-provenance.json`.
