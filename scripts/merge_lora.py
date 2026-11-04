#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into its base policy -> a full, servable pretrained_model dir.

The lerobot policy server has no PEFT path: it calls `from_pretrained(path)` directly (bypassing
factory.make_policy's adapter branch), which expects a full `model.safetensors` and will not load or
apply a raw LoRA adapter dir. So a LoRA run's `checkpoints/<step>/pretrained_model` must be merged
into the base weights first. This script does that, reusing lerobot 0.6.1's own load primitives (the
same ones policies/factory.py make_policy uses for PEFT):

  1. read the adapter's PeftConfig  -> base model id (e.g. lerobot/pi05_base)
  2. build the base policy with the CHECKPOINT's config (its UR5 input/output features + dims)
     but the BASE weights
  3. wrap with PeftModel.from_pretrained(policy, adapter_dir) and merge_and_unload()
  4. save_pretrained() the merged policy  (writes config.json + model.safetensors ONLY)
  5. copy the pre/post-processor files across -- save_pretrained does NOT save them, and they hold
     the normalization statistics; without them inference de/normalizes wrong and the arm gets
     garbage actions.

Usage (run with the lerobot venv's python so lerobot+peft+torch resolve):
  ~/abhay_ws/lerobot/.venv/bin/python scripts/merge_lora.py \
      <adapter_dir>/pretrained_model  <out_dir>/pretrained_model

  # then point configs/policy/lerobot_ur5_finetuned.yaml:pretrained_name_or_path at <out_dir>/pretrained_model

Notes:
  * CPU is the default and safest (a ~4B pi05 in bf16 needs ~8-9 GB RAM); pass --device cuda to use
    the GPU if it has the headroom.
  * The base weights (lerobot/pi05_base, public) download from the HF Hub on first run. If the policy
    constructor trips a gated-repo error for the PaliGemma processor, `export HF_TOKEN=hf_...` first
    (the same read token used for training).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Files the merge WRITES or that are adapter-only -- never copied from the source checkpoint.
_SKIP = {"adapter_config.json", "adapter_model.safetensors", "config.json", "model.safetensors"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir", type=Path, help="the LoRA checkpoint's pretrained_model dir (has adapter_config.json)")
    ap.add_argument("out_dir", type=Path, help="destination pretrained_model dir for the merged model")
    ap.add_argument("--base", default=None, help="override base model id (default: read from the adapter config)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device for the merge (default cpu)")
    args = ap.parse_args()

    adapter_dir: Path = args.adapter_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        ap.error(f"no adapter_config.json in {adapter_dir} -- is this a LoRA checkpoint's pretrained_model dir?")
    if out_dir == adapter_dir:
        ap.error("out_dir must differ from adapter_dir")

    # Imported here so --help works without the heavy deps loaded.
    import torch  # noqa: F401  (ensures a clear error if the wrong python is used)
    from peft import PeftConfig, PeftModel

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_id = args.base or peft_config.base_model_name_or_path
    if not base_id:
        ap.error("adapter config has no base_model_name_or_path; pass --base explicitly")

    # The checkpoint's own policy config: carries the trained input/output features + normalization
    # mode, so the architecture matches what the adapter was trained against.
    cfg = PreTrainedConfig.from_pretrained(str(adapter_dir))
    cfg.device = args.device
    policy_cls = get_policy_class(cfg.type)

    print(f"[merge] policy type : {cfg.type}")
    print(f"[merge] base weights: {base_id}")
    print(f"[merge] adapter     : {adapter_dir}")
    print(f"[merge] device      : {args.device}")

    # Base policy = checkpoint CONFIG + BASE weights (exactly what factory.make_policy does for PEFT).
    #
    # MEMORY: lerobot's from_pretrained does `instance = cls(config)` -- it MATERIALIZES the whole
    # ~4.14B-param model in RAM (real tensors) before loading weights. torch's default dtype is
    # float32, so that transient skeleton is ~16.6 GB; on a loaded desktop this drove the session
    # into swap-thrash and systemd-oomd reaped Chrome + the terminal. pi05 was trained in bf16, so we
    # build in bf16 (~8.3 GB, lossless vs the trained precision and matching a normal pi05 checkpoint)
    # by setting the default dtype for the duration of construction + merge, then restoring it.
    print("[merge] loading base policy in bf16 (downloads base weights on first run)...")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        policy = policy_cls.from_pretrained(base_id, config=cfg)
        print("[merge] applying adapter + merge_and_unload()...")
        peft_model = PeftModel.from_pretrained(policy, str(adapter_dir), config=peft_config)
        merged = peft_model.merge_and_unload()
    finally:
        torch.set_default_dtype(prev_dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] writing merged weights -> {out_dir}")
    merged.save_pretrained(out_dir)

    # save_pretrained writes only config.json + model.safetensors. Copy everything else from the
    # source checkpoint (crucially the policy_preprocessor*/policy_postprocessor* normalization
    # stats; also train_config.json for provenance). Without the processors, inference is wrong.
    copied = []
    for f in sorted(adapter_dir.iterdir()):
        if f.is_file() and f.name not in _SKIP:
            shutil.copy2(f, out_dir / f.name)
            copied.append(f.name)
    print(f"[merge] copied {len(copied)} aux file(s): {', '.join(copied)}")

    # Sanity: the normalization processors must have made it across.
    procs = [p.name for p in out_dir.glob("policy_pre*") ] + [p.name for p in out_dir.glob("policy_post*")]
    if not procs:
        raise SystemExit("ERROR: no policy_pre/postprocessor files in the output -- normalization stats are "
                         "MISSING; the merged model would produce wrong actions. Check the source checkpoint.")

    print(f"[merge] done. normalization processors present: {len(procs)}")
    print(f"[merge] point configs/policy/lerobot_ur5_finetuned.yaml:pretrained_name_or_path at:\n          {out_dir}")


if __name__ == "__main__":
    main()
