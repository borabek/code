# -*- coding: utf-8 -*-
"""Score the model against a HUMAN-SEALED CP benchmark. This REFUSES to produce a number unless
every part is fully reviewed and signed:
  - every part in the split has a <pid>.json,
  - review_status == "SEALED",
  - every cp has a decision in {accept, reject, move, add, flip_direction} (none None/pending),
  - seal.reviewer, seal.reviewed_utc, seal.signature are all non-empty,
  - source_sha matches the current corpus files (no silent data drift).
If any check fails it prints the blockers and EXITS WITHOUT A SCORE. There is no auto-confirm.

GT = cps with decision in {accept, move, add} (moved -> use its (possibly edited) point).
PRED = model with the frozen cp-v2 prediction post-processing.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_seal_score.py --split test_locked
"""
import argparse, os, json, hashlib, sys
import numpy as np
import scheffler_dataset as dataset
import diffusionnet, metrics, cp_openings

CKPT = "results/scheffler_semantic/refit91.pt"; OPCACHE = "results/scheffler_semantic/operators"
ACCEPT = {"accept", "move", "add"}
VALID = ACCEPT | {"reject", "flip_direction"}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None


def check(rec, pid, split):
    b = []
    if rec.get("review_status") != "SEALED":
        b.append(f"{pid}: review_status={rec.get('review_status')} (need SEALED)")
    seal = rec.get("seal") or {}
    if not (seal.get("reviewer") and seal.get("reviewed_utc") and seal.get("signature")):
        b.append(f"{pid}: unsigned seal (reviewer/reviewed_utc/signature required)")
    for c in rec.get("cps", []):
        if c.get("decision") not in VALID:
            b.append(f"{pid}: cp {c.get('id')} decision={c.get('decision')} (undecided)")
    pdir = os.path.join("wscad_corpus_scheffler_exact", split, pid)
    src = rec.get("source_sha") or {}
    if src.get("step") and src["step"] != sha16(os.path.join(pdir, f"{pid}.stp")):
        b.append(f"{pid}: STEP sha drift (labels/geometry changed since review)")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_locked"); ap.add_argument("--conf-dir", default="_cp_confirm")
    ap.add_argument("--min-v", type=int, default=20); ap.add_argument("--vertex-conf", type=float, default=0.9)
    ap.add_argument("--match", type=float, default=5.0)
    a = ap.parse_args()
    samples = dataset.load_split("wscad_corpus_scheffler_exact", a.split,
                                 allow_locked=(a.split == "test_locked"), verify_hashes=False)
    blockers = []; recs = {}
    for s in samples:
        pid = s["part_id"]; jp = os.path.join(a.conf_dir, f"{pid}.json")
        if not os.path.exists(jp):
            blockers.append(f"{pid}: no confirm JSON"); continue
        rec = json.load(open(jp)); recs[pid] = rec
        blockers += check(rec, pid, a.split)
    if blockers:
        print("SCORE REFUSED -- the benchmark is not human-sealed. Blockers:")
        for x in blockers[:40]:
            print("  -", x)
        print(f"\n{len(blockers)} blocker(s). Auto-confirmed CPs counted as human = 0 (by design).")
        sys.exit(2)

    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    TP = FP = FN = 0; per = []
    for s in samples:
        pid = s["part_id"]; V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        gt = np.array([c["point"] for c in recs[pid]["cps"] if c.get("decision") in ACCEPT] or []).reshape(-1, 3)
        plab, probs = diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE, return_probs=True)
        pr = np.array([c["point"] for c in cp_openings.connection_points(
            V, F, np.asarray(plab), min_v=a.min_v, probs=probs, vertex_conf=a.vertex_conf)] or []).reshape(-1, 3)
        if len(gt) and len(pr):
            m, up, ug = metrics.match_predictions(pr, gt, a.match); tp, fp, fn = len(m), len(up), len(ug)
        else:
            tp, fp, fn = 0, len(pr), len(gt)
        TP += tp; FP += fp; FN += fn; per.append((pid, len(gt), len(pr), tp, fp, fn))
    f1 = 2*TP/max(2*TP+FP+FN, 1); prec = TP/max(TP+FP, 1); rec_ = TP/max(TP+FN, 1)
    for pid, ng, npd, tp, fp, fn in per:
        print(f"  {pid:<11}GT{ng:>3} pred{npd:>3} TP{tp:>3} FP{fp:>3} FN{fn:>3}{'  <--' if (fp or fn) else ''}")
    print(f"\n>>> HUMAN-SEALED CP-F1 ({a.split}): F1={f1:.3f}  precision={prec:.3f}  recall={rec_:.3f}  TP={TP} FP={FP} FN={FN}")
    print("DONE")


if __name__ == "__main__":
    main()
