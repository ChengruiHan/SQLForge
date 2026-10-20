import json
import tempfile
import unittest
from pathlib import Path

from fixtures import build_demo

from sqlforge.environment import (
    DIRECT_PROMPT,
    ONE_ACTION_PROMPT,
    SYSTEM_PROMPT,
    EnvConfig,
    SQLRepairEnv,
)
from sqlforge.tasks import load_tasks


class ProtocolVersionTests(unittest.TestCase):
    def test_only_system_prompt_changes_and_multiple_actions_are_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = load_tasks(build_demo(str(Path(tmp) / "data")))[0]
            old = SQLRepairEnv(task, EnvConfig(max_turns=4))
            new = SQLRepairEnv(task, EnvConfig(max_turns=4, prompt_version="one_action_v1"))
            direct = SQLRepairEnv(task, EnvConfig(submit_only=True, prompt_version="one_action_v1"))
            try:
                self.assertEqual(old.messages[0]["content"], SYSTEM_PROMPT)
                self.assertEqual(new.messages[0]["content"], ONE_ACTION_PROMPT)
                self.assertEqual(direct.messages[0]["content"], DIRECT_PROMPT)
                self.assertEqual(old.messages[1:], new.messages[1:])
                invalid = '{"name":"schema","arguments":{}}\n{"name":"schema","arguments":{}}'
                self.assertEqual(old.step(invalid), new.step(invalid))
                self.assertEqual(new.execute_calls, 0)
                self.assertEqual(new.invalid_actions, 1)
                action = json.dumps({"name": "execute_sql", "arguments": {"sql": "SELECT 1"}})
                old.step(action)
                new.step(action)
                self.assertEqual(old.messages[1:], new.messages[1:])
                self.assertEqual(old.turns, new.turns)
            finally:
                old.close()
                new.close()
                direct.close()


if __name__ == "__main__":
    unittest.main()
