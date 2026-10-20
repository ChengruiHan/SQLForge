# Data contract

Data source: [SIX-GYM-SQLite](https://huggingface.co/datasets/birdsql/six-gym-sqlite), revision `d603cc8af5294bb75415b8137d362080dab01bfd`. Obtain the Query task export and SQLite databases from upstream under its terms. This repository distributes task IDs, scores and code, not benchmark data or hidden tests.

`results/task-splits.json` fixes 256 formal training IDs, 128 development IDs and 128 test IDs. `results/task-order.json` fixes training order and the 32 development curve tasks. Training databases: books, book_publishing_company, car_retails, lego, movie_3, netflix. Development: chinook and hockey. Test: address, olympics and public_review_platform. The historical source selection favored smaller databases for compute reasons; this is a selection bias.

```bash
sqlforge import --input /path/to/six-gym-export.jsonl \
  --db-root /path/to/database --split-map results/task-splits.json \
  --output data/tasks.jsonl
sqlforge audit --tasks data/tasks.jsonl --output data/audit.json \
  --evaluator vendor/bird-evaluator
```

The importer normalizes downloaded records and selects IDs; it does not generate new questions, teacher traces or SFT samples. It requires source fields `instance_id`, `db_id`, `query`, `issue_sql`, `sol_sql`, `test_cases`, and Query `category`; optional `preprocess_sql` and `conditions` are preserved. Missing IDs fail instead of silently substituting tasks. Databases can be named `<db_id>.sqlite` or `<db_id>/<db_id>_template.sqlite` beneath the supplied root.

The normalized JSONL uses `sqlforge.tasks.Task`. It contains private reference SQL and tests for the scorer. The environment only receives the public projection. Audit verifies reference success, original issue failure, database identities and partition separation. Audit failure must be investigated, never converted into model failure or a replacement label.

A finite hidden test suite is not universal semantic certification. Some cross-database tasks share requirement text; database isolation does not imply completely unseen language. Historical development/test data have already been examined.
