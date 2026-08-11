# -*- coding: utf-8 -*-
"""HUMAN CableEntry marks + MODEL prediction for the other 4 classes = full 5-class labels.

WHY: the annotator marked CableEntry only, so `--partial-dir` had to train those parts with a
masked BCE that supervises 1 of 5 classes. Measured: it HURTS (val Connection IoU 0.622 -> 0.586
tight, 0.584 dilated; arbiter CP F1 0.520 -> 0.308). Dilating the marks to the corpus convention
changed nothing, so the extent was not the problem -- the OBJECTIVE MISMATCH is the suspect.

THIS turns the same human work into labels that train the IDENTICAL 5-class loss as the corpus:
take the product model's own prediction for Housing/Contact/SnapPoint/LabelSurface, then overwrite
the CableEntry channel with the human marks (human wins every conflict). Thesis-endorsed: the
conclusion (line 2157) asks for exactly this feedback loop -- predictions back into the labelling
tool. Unlike pure pseudo-labels these carry real human information on the class that matters.

Adopt only if it beats the product on the arbiter CP F1 (0.520). Otherwise: measured negative.
"""
import os, sys, glob, shutil, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D
from region_label_helper import load_obj
from infer_step_cp import load_any

CE = 3
OP = "results/step_infer/ops"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_label_targets_dilated",
                    help="human CableEntry marks (dilated by default: same extent as the corpus)")
    ap.add_argument("--dst", default="_label_targets_hybrid")
    ap.add_argument("--ckpt", default="results/seg_extra/selftrain_120.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, meta, _ = load_any(a.ckpt, dev=a.device)
    os.makedirs(a.dst, exist_ok=True)
    kept = 0
    for d in sorted(glob.glob(os.path.join(a.src, "*"))):
        if not os.path.isdir(d): continue
        pid = os.path.basename(os.path.normpath(d))
        of, lf = os.path.join(d, f"{pid}.obj"), os.path.join(d, f"{pid}.labels.txt")
        if not (os.path.exists(of) and os.path.exists(lf)): continue
        H = np.array([int(x) for x in open(lf).read().split()], np.int64)
        V, F = load_obj(of)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        pred = np.asarray(D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP))
        lab = pred.copy()
        lab[pred == CE] = 0        # drop the model's CableEntry guess entirely...
        lab[H == CE] = CE          # ...the human decides where CableEntry is
        od = os.path.join(a.dst, pid); os.makedirs(od, exist_ok=True)
        shutil.copy(of, os.path.join(od, f"{pid}.obj"))
        open(os.path.join(od, f"{pid}.labels.txt"), "w").write("\n".join(str(int(x)) for x in lab))
        agree = float(((pred == CE) & (H == CE)).sum()) / max((H == CE).sum(), 1)
        print(f"  {pid:18s} human CE {100*(H==CE).mean():4.1f}%  model agreed on {100*agree:4.1f}% of it", flush=True)
        kept += 1
    print(f"\n{kept} hybrid parts -> {a.dst}  (train with --extra-dir/--pseudo-dir, NOT --partial-dir)")


if __name__ == "__main__":
    main()
