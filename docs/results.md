# Results, evidence and limits

The published scores are reconstructed by `python scripts/verify_results.py` from 512 per-task test outcomes in `results/test-outcomes.jsonl`. The export contains IDs, successes and costs, without dataset questions, reference answers or hidden tests. Aggregate data are in `rl-summary.json` and `harness-summary.json`; original-report/archive SHA-256 values are in `source-provenance.json`.

## RL

Raw/GRPO/GSPO/RLOO pass 41/46/45/42 development tasks and 39/39/38/42 test tasks, each out of 128. RLOO gains 3 test successes and loses 0 relative to Raw (+2.34 percentage points); GRPO gains 3 and loses 3; GSPO gains 2 and loses 3. One training seed and one final sampling seed do not establish stable algorithm ranking or significance. Same-database tasks are correlated. Development gains did not consistently transfer to test.

The 1,024 on-policy training trajectories per method include 334/348/346 successful trajectories for GRPO/GSPO/RLOO. These are sampled across changing policies, not final-checkpoint accuracy. Only 34/40/38 task groups have mixed external success rewards; this is a diagnostic observation, not proof of the bottleneck.

## Scoring amendment

TRAIN_4648's original GSPO submission executed within the 3-second candidate budget, but a hidden test removing DISTINCT caused a pathological SQLite query plan. Original hidden grading timed out at 30 seconds and 300 seconds. This is a scoring infrastructure exception, not an automatic model failure.

A disposable scoring database copy received:

```sql
CREATE INDEX sqlforge_recovery_person ON games_competitor(person_id);
```

All table contents, original candidate/reference outputs and original database identity were checked unchanged; the original test code was retained. The resulting grade was failure, completing GSPO at 38/128. Without resolving that exception, the original-rule bound is 38/128–39/128.

The final historical comparison uniformly applied the disclosed copy-index treatment to all four models and revalidated reference-pass/issue-fail. Original-source candidate execution times were checked separately: GSPO~0.449 seconds and RLOO~0.567 seconds; Raw/GRPO had no submission on this task. The final counts stayed 39/39/38/42. This is not an entirely unamended official benchmark run. The public runtime does not silently apply benchmark-specific index changes.

## Harness

| Candidate | Search /64 | Selection seed 42 /64 | Selection seed 43 /64 |
|---|---:|---:|---:|
| H0 |25|21|19|
| adaptive7 |28|18|17|
| adaptive6 |27|19|16|

The selection means are 31.25% for H0 and 27.34% for both finalists. The two seeds reuse the same 64 tasks; they are not 128 independent questions. Candidates failed the predeclared adoption rule and H0 was retained. Task-level paired uncertainty intervals cover zero; do not claim statistically established degradation either.

Eight candidates, four API proposals and 1024 new dev trajectories were completed: 512 search, 384 selection, 128 Raw/H0 baseline. No new test generations and no demonstrated new-harness transfer benefit. Reusing 512 historical H0 trajectories is not an additional 512 independent test samples.

## Release scope

This public snapshot centers on the three resume claims: native multi-turn RL, modular harness optimization and the held-out-database test comparison. Earlier SFT/data-generation experiments and private operational files remain outside the release. The README reports completed experiments; cleaned command wrappers have CPU validation, but have not been rerun on GPU as a separate reproduction. The dates of the stored run artifacts are preserved; no results are backdated to match resume dates.
