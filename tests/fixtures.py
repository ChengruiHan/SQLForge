from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from sqlforge.tasks import Task, write_tasks


def build_demo(out_dir: str) -> Path:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        db_path = root / f"shop_{seed}.sqlite"
        if db_path.exists():
            raise FileExistsError(f"Refusing to overwrite demo DB: {db_path}")
        rng = random.Random(seed)
        with sqlite3.connect(db_path) as db:
            db.executescript("""
                CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT);
                CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER,
                                    total REAL, refunded INTEGER);
                CREATE TABLE excluded(customer_id INTEGER);
                INSERT INTO customers VALUES (1,'Ada','east'), (2,'Bo','west'),
                    (3,'Chen','east'), (4,'Di','north');
                INSERT INTO excluded VALUES (2), (NULL);
            """)
            db.executemany(
                "INSERT INTO orders VALUES (?,?,?,?)",
                [
                    (1, 1, 10 + rng.randrange(10), 0),
                    (2, 1, 20, 0),
                    (3, 2, 30, 1),
                    (4, 3, 5, 0),
                ],
            )
    specs = [
        (
            "distinct",
            "Return each customer ID that has an order, once per customer.",
            "SELECT customer_id FROM orders",
            "SELECT DISTINCT customer_id FROM orders",
        ),
        (
            "anti_null",
            "Return IDs of customers not present in excluded; NULL entries exclude nobody.",
            "SELECT id FROM customers WHERE id NOT IN (SELECT customer_id FROM excluded)",
            "SELECT c.id FROM customers c WHERE NOT EXISTS "
            "(SELECT 1 FROM excluded e WHERE e.customer_id=c.id)",
        ),
        (
            "left_count",
            "Return every customer ID and their order count, including zero orders.",
            "SELECT c.id, COUNT(*) FROM customers c LEFT JOIN orders o "
            "ON o.customer_id=c.id GROUP BY c.id",
            "SELECT c.id, COUNT(o.id) FROM customers c LEFT JOIN orders o "
            "ON o.customer_id=c.id GROUP BY c.id",
        ),
        (
            "refund",
            "Return the sum of totals of orders that were not refunded.",
            "SELECT SUM(total) FROM orders WHERE refunded=1",
            "SELECT SUM(total) FROM orders WHERE refunded=0",
        ),
        (
            "syntax",
            "Return each customer's name.",
            "SELECT full_name FROM customers",
            "SELECT name FROM customers",
        ),
        (
            "having",
            "Return customer IDs with at least two orders.",
            "SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 2",
            "SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) >= 2",
        ),
    ]
    tasks = [
        Task(
            task_id=f"demo_{name}",
            db_id="shop",
            db_path="shop_0.sqlite",
            question=question,
            issue_sql=[issue],
            reference_sql=[reference],
            hidden_db_paths=["shop_1.sqlite", "shop_2.sqlite"],
            judge="execution",
            source="synthetic_integration_fixture",
            split="train" if i < 4 else "dev",
        )
        for i, (name, question, issue, reference) in enumerate(specs)
    ]
    path = root / "tasks.jsonl"
    write_tasks(path, tasks)
    return path
