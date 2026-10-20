"""Copy and hash fixed-final-budget LoRA adapters; retain all source checkpoints."""

import argparse
import shutil
from pathlib import Path

from sqlforge.tasks import file_sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    root = Path(args.run).resolve()
    arms = {"raw": {"adapter": None}}
    for method in ["grpo", "gspo", "rloo"]:
        source = root / method / "checkpoints/global_step_64/actor/lora_adapter"
        if not (source / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(source)
        target = root / "final-adapters" / method
        if not target.exists():
            shutil.copytree(source, target)
        for filename in ["adapter_model.safetensors", "adapter_config.json"]:
            if file_sha256(target / filename) != file_sha256(source / filename):
                raise ValueError("Existing export differs from fixed final checkpoint")
        arms[method] = {"adapter": str(target)}
    write_json(root / "arms.json", arms)


if __name__ == "__main__":
    main()
