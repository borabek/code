# Thin convenience wrapper: run the full training on the real ABB corpus with
# the known-good flags baked in, so nobody has to retype the long command.
# All it does is assemble argv and hand off to train_cp.main -- no logic lives
# here, so there is nothing to duplicate.
#
# Usage:
#   python run_full.py                      # uses the defaults below
#   python run_full.py --epochs 50          # any train_cp.py flag overrides/extends
#   python run_full.py C:\other\corpus ...  # positional corpus overrides the default
#
# Defaults target the native-Windows layout (knngraph backbone, no native geometry
# deps). To run the diffusionnet backbone use WSL and override corpus/backbone/device:
#   python run_full.py /mnt/c/Users/DE00024082/Desktop/JSON --backbone diffusionnet --device cuda

import sys
import train_cp

# (flag, value) defaults; a flag already present in argv is NOT overridden.
# value=None means a bare boolean flag. Checkpoints/cache/results are
# repo-relative so the same command works on every machine the repo is cloned
# to; --resume makes re-running this script CONTINUE the run from the last
# committed checkpoint instead of starting over.
#
# The training-quality flags below are the known-good config (same as train_1h.py:
# augmentation + reflection, global context, gradient accumulation, LR decay,
# weight decay, dropout, grad clip, centernet heat loss, leakage-safe geometry
# split, frozen test split). The decode flags are the HELD-OUT-best operating point found
# by sweep_decode.py (heatmap_thresh=0.40, min_votes=1, nms=5.0). So one command
# trains AND exports a model carrying the honest operating point.
DEFAULT_CORPUS = r"C:\Users\DE00024082\Desktop\JSON"
DEFAULTS = [
    ("--backbone", "knngraph"),
    ("--device", "cuda"),
    ("--max-gpu-verts", "7000"),     # safe on a 4 GB GPU; OOM falls back to CPU
    ("--epochs", "200"),
    # --- known-good training config ---
    ("--augment", "4"),
    ("--aug-reflect", None),
    ("--knn-global", None),
    ("--accum-steps", "8"),
    ("--lr-decay-every", "12"),
    ("--weight-decay", "5e-4"),
    ("--grad-clip", "5"),
    ("--knn-dropout", "0.1"),
    ("--heat-loss", "centernet"),
    ("--split-group", "geometry"),   # group by actual mesh (no leak); prefix mode
                                      # is naming-scheme-dependent and known broken
                                      # for wscad+PXC (dumps them onto one side)
    ("--test-frac", "0.15"),
    ("--eval-every", "3"),
    ("--patience", "12"),
    # --- held-out-best decode operating point (sweep_decode.py) ---
    ("--heatmap-thresh", "0.40"),
    ("--min-votes", "1"),
    ("--nms-clearance-mm", "5.0"),
    # --- bookkeeping ---
    ("--ckpt-every", "1"),
    ("--resume", None),
    ("--export-model", "model_full.pt"),
    ("--out", "results_full.json"),
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # positional corpus: if the user gave none (argv empty or starts with a flag),
    # prepend the default so train_cp's required `source` arg is satisfied.
    if not argv or argv[0].startswith("-"):
        argv = [DEFAULT_CORPUS] + argv
    for flag, value in DEFAULTS:
        if flag not in argv:
            argv += [flag] if value is None else [flag, value]
    train_cp.main(argv)


if __name__ == "__main__":
    main()
