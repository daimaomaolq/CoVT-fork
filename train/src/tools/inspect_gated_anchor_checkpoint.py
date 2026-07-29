from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    path = Path(args.checkpoint) / "non_lora_state_dict.bin"
    state = torch.load(path, map_location="cpu")
    interesting = {}
    for name, value in state.items():
        if "_gate_" in name or "_prompt_embeddings" in name:
            tensor = value.detach().float()
            interesting[name] = {
                "shape": list(tensor.shape),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()) if tensor.numel() > 1 else 0.0,
                "abs_max": float(tensor.abs().max()),
                "nonzero": int(torch.count_nonzero(tensor)),
            }
    print(json.dumps({"tensor_count": len(state), "gate_tensors": interesting}, indent=2))


if __name__ == "__main__":
    main()
