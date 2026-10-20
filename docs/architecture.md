# Runtime boundaries

`tasks.py` stores task metadata and identities; `environment.py` creates the public task view. `database.py` operates on per-trajectory SQLite copies with an execution deadline and authorizer. `judges.py` runs hidden tests in separate scoring copies. Reference SQL and test code are excluded from policy observations.

`verl_agent.py` preserves generated token IDs while inserting Qwen chat/tool context with zero loss mask. Terminal grading runs after the sample is persisted. Infrastructure errors raise rather than becoming reward 0. Native verl owns optimization; `scripts/train.py` composes the frozen objective settings and records resolved configuration/source identities.

`harness.py` renders optional public prompt/feedback/history changes. `harness_search.py` partitions tasks and enforces the selection rule. `scripts/search_harness.py` orchestrates search on frozen models. `sampling.py` checkpoints each generated batch and rejects task/harness mismatches. `eval_cli.py` locks model/config/task identities before test evaluation and reuses saved generations during scoring recovery.

Resource assumptions: a single isolated GPU job, trusted benchmark test code, Linux for GPU execution, and no simultaneous edits to model/data files after locking. This is an experimental pipeline; it is not a production database authorization service or an OS sandbox.
