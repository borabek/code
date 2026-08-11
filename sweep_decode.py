# Decode-parameter sweep for a trained connection-point regressor.
#
# Why this exists: the keypoint metrics (precision / F1) are *decode-bound*, not
# model-bound -- the network's per-vertex (heatmap, offset, direction) output is
# fixed once trained, but how many discrete connection points you read out of it
# depends entirely on these knobs in cp_targets.decode_predictions:
#
#   heatmap_thresh   min sigmoid heat for a vertex to vote   (higher -> fewer FP)
#   min_votes        min voting vertices per accepted cluster (higher -> fewer FP)
#   nms_clearance_mm vote-merge radius = max(2*sigma, this)   (larger -> fewer FP)
#
# plus the SCORING knob dist_thresh_mm (how close a prediction must be to a GT
# point to count as a hit), which is not a model parameter but materially moves
# TP/FP/FN, so it is swept too.
#
# This tool runs inference ONCE per part (the expensive step), caches the (N,7)
# arrays (optionally to disk via --cache-arrs), then sweeps a grid of those knobs
# and reports the configuration that maximises F1 -- no retraining. Point it at
# the same corpus / --val-frac / --seed / --max-parts you trained with so the val
# split matches exactly.
#
# HONESTY: by default the winning config is BOTH selected and reported on the val
# split, which is optimistic (you take the max over the whole grid on the same
# data). Pass --holdout-frac to split the chosen split into a selection subset and
# a held-out report subset: the config is picked on selection and the headline F1
# is measured on the held-out parts the selection never saw. Without it, the
# printout is clearly labelled "selected-on-val (optimistic)".
#
# Example:
#   python sweep_decode.py checkpoints/cp_knngraph_best.ckpt parts/ \
#       --val-frac 0.2 --seed 0 --holdout-frac 0.4 --min-precision 0.5
#
# Take the winning row's --heatmap-thresh / --min-votes / --nms-clearance-mm and
# pass them to train_cp.py (or your inference) to lock the operating point in.

import os
import json
import argparse
import hashlib
import logging
import itertools

import numpy as np

import json_dataset as jd
import cp_targets as ct
import cp_regressor as cpr
import metrics as mcp
from train_cp import split_part_ids, aggregate, _nms_radius

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# per-backbone inference: sample -> (N,7) numpy (heatmap sigmoid'd, dir unit)
# ---------------------------------------------------------------------------

def _build_infer(backbone, model, meta, device, op_cache_dir, max_gpu_verts):
    """Mirror the inference path train_cp.train_and_eval builds for each backbone.

    The knngraph subsample is deterministic (part_subsample_seed, keyed on the
    checkpoint's training seed + this part's PartNr -- the SAME subsample training
    used for this exact part, not a fixed RandomState(0)), so the (N,7) array for a
    part is identical across runs -- which is what makes the --cache-arrs disk
    cache and grid re-sweeps reproducible.
    """
    if backbone == "knngraph":
        def _infer_knn(s, pm):
            return cpr.infer_knngraph(model, meta, s["verts_norm"], device=device,
                                      max_gpu_verts=max_gpu_verts,
                                      offset_scale=s["scale"],
                                      part_nr=s.get("part_nr"))
        return _infer_knn
    if backbone == "diffusionnet":
        def _infer_dn(s, pm):
            return cpr.infer_diffusionnet(model, meta, pm["vertices"], s["faces"],
                                          s["verts_norm"], op_cache_dir=op_cache_dir,
                                          device=device, max_gpu_verts=max_gpu_verts,
                                          offset_scale=s["scale"])
        return _infer_dn
    if backbone == "mlp":
        import torch

        def _infer_mlp(s, pm):
            model.eval()
            with torch.no_grad():
                x = torch.tensor(s["verts_norm"], dtype=torch.float32, device=device)
                return cpr.pred_to_array(model(x), offset_scale=s["scale"])
        return _infer_mlp
    raise ValueError(f"unknown backbone {backbone!r}")


# ---------------------------------------------------------------------------
# load + split the corpus the same way train_cp does, keep one split
# ---------------------------------------------------------------------------

def _load_split(source, split, val_frac, seed, max_parts, split_group="geometry",
                extra_sources=None, keep_prefixes=None, max_bbox_mm=None):
    """Return (samples, metas) for the requested split ('val' | 'train' | 'all'),
    skipping parts with no connection points (exactly as training does).

    split_group must MATCH the training run ('none'/'prefix'/'geometry') so the val
    split is identical -- otherwise the sweep tunes on a different set of parts.
    extra_sources/keep_prefixes/max_bbox_mm MUST also match the training run's
    --extra-source/--keep-prefixes/--max-bbox-mm, or this reconstructs the WRONG
    population entirely (e.g. sweeping thresholds against the full 479-part real
    corpus when the checkpoint was trained scoped to 306 connector-sized parts).
    """
    parts = []
    for part in jd.iter_parts(source):
        parts.append(part)
        if max_parts and len(parts) >= max_parts:
            break
    for extra in (extra_sources or []):
        for part in jd.iter_parts(extra):
            parts.append(part)
            if max_parts and len(parts) >= max_parts:
                break
    if not parts:
        raise SystemExit(f"no parts found in {source}")
    if keep_prefixes:
        prefs = tuple(s.strip() for s in keep_prefixes.split(",") if s.strip())
        before = len(parts)
        parts = [p for p in parts if str(p.part_nr).startswith(prefs)]
        logger.info("scope filter --keep-prefixes %s: kept %d of %d part(s)",
                    keep_prefixes, len(parts), before)
    if max_bbox_mm:
        def _diag(p):
            V = np.asarray(p.vertices, dtype=np.float64)
            return float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 0.0
        before = len(parts)
        parts = [p for p in parts if _diag(p) <= max_bbox_mm]
        logger.info("scope filter --max-bbox-mm %.0f: kept %d of %d part(s)",
                    max_bbox_mm, len(parts), before)
    if not parts:
        raise SystemExit("no parts left after scope filters")

    ids = [p.part_nr for p in parts]
    group_keys = jd.build_group_keys(parts, mode=split_group)
    train_ids, val_ids = split_part_ids(ids, val_frac=val_frac, seed=seed,
                                        group_keys=group_keys)
    if not train_ids:                       # mirror train_cp's tiny-corpus guard
        train_ids = {ids[0]}
        val_ids.discard(ids[0])
        logger.warning("tiny corpus: train split fell back to a SINGLE part (%s); "
                       "raise --max-parts/--val-frac for a meaningful sweep", ids[0])

    samples, metas = [], []
    for p in parts:
        if p.n_cps == 0:
            continue
        in_val = p.part_nr in val_ids
        if split == "val" and not in_val:
            continue
        if split == "train" and in_val:
            continue
        samples.append(cpr.prepare_sample(p, dedup=True))
        metas.append({"part_nr": p.part_nr, "vertices": p.vertices})
    return samples, metas


# ---------------------------------------------------------------------------
# inference-array cache (the expensive step) -- disk-persist keyed by a signature
# ---------------------------------------------------------------------------

def _arrs_signature(ckpt, backbone, split, val_frac, seed, max_parts, part_nrs):
    """A stable key for the cached (N,7) arrays: anything that changes the model
    or the set/order of parts must invalidate the cache."""
    try:
        mtime = os.path.getmtime(ckpt)
    except OSError:
        mtime = 0.0
    payload = {
        "ckpt": os.path.abspath(ckpt), "ckpt_mtime": mtime, "backbone": backbone,
        "split": split, "val_frac": val_frac, "seed": seed,
        "max_parts": max_parts, "part_nrs": list(part_nrs),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _load_arrs_cache(path, signature):
    """Return cached arrs if the file exists and its signature matches, else None."""
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        if str(data["_signature"]) != signature:
            logger.info("cache %s is stale (signature mismatch) -- recomputing", path)
            return None
        n = int(data["_n"])
        arrs = [data[f"a{i}"] for i in range(n)]
        logger.info("loaded %d cached prediction arrays <- %s", n, path)
        return arrs
    except Exception as exc:                                # noqa: BLE001
        logger.warning("could not read array cache %s (%s) -- recomputing", path, exc)
        return None


def _save_arrs_cache(path, signature, arrs):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {f"a{i}": a for i, a in enumerate(arrs)}
    payload["_signature"] = np.array(signature)
    payload["_n"] = np.array(len(arrs))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)
    logger.info("cached %d prediction arrays -> %s", len(arrs), path)


# ---------------------------------------------------------------------------
# grid build / score / rank (shared with sweep_pos_weight via import)
# ---------------------------------------------------------------------------

def build_grid(thresholds, min_votes_list, nms_list, dist_list):
    """The full decode/score grid as a list of (thr, min_votes, nms, dist) tuples."""
    return list(itertools.product(thresholds, min_votes_list, nms_list, dist_list))


def _score(arrs, samples, metas, thr, min_votes, nms_clear, dist_thresh_mm):
    reports = []
    for arr, s, pm in zip(arrs, samples, metas):
        preds = ct.decode_predictions(pm["vertices"], arr, heatmap_thresh=thr,
                                       nms_radius_mm=_nms_radius(s, nms_clear),
                                       min_votes=min_votes)
        rep = mcp.keypoint_report(preds, s["gt_points"], s["gt_directions"],
                                  dist_thresh_mm=dist_thresh_mm)
        reports.append(rep)
    return aggregate(reports)


def sweep_grid(arrs, samples, metas, grid):
    """Score every config in `grid` over the cached arrays. Returns a list of row
    dicts (one per config). Configs whose decode yields no parts are skipped."""
    rows = []
    for thr, mv, nms, dist in grid:
        agg = _score(arrs, samples, metas, thr, mv, nms, dist)
        if not agg:
            continue
        rows.append({
            "heatmap_thresh": thr, "min_votes": mv, "nms_clearance_mm": nms,
            "dist_thresh_mm": dist,
            "micro_f1": agg["micro_f1"], "micro_precision": agg["micro_precision"],
            "micro_recall": agg["micro_recall"], "f1": agg["f1"],
            "precision": agg["precision"], "recall": agg["recall"],
            "tp": agg["total_tp"], "fp": agg["total_fp"], "fn": agg["total_fn"],
        })
    return rows


def rank_rows(rows, metric="micro_f1", min_precision=0.0, min_recall=0.0):
    """Sort rows best-first by `metric`, with a precision tie-break and optional
    precision/recall floors.

    The bare-F1 ranking can crown a lopsided high-recall/low-precision corner that
    ties on F1; tie-breaking on precision prefers the more usable operating point.
    The floors drop configs below a minimum precision/recall; if the floors would
    eliminate every config we fall back to the unfiltered set (and the caller is
    told via the returned `floored` flag) rather than returning nothing.
    """
    prec_key = "precision" if metric == "f1" else "micro_precision"
    eligible = [r for r in rows
                if r["micro_precision"] >= min_precision
                and r["micro_recall"] >= min_recall]
    floored = bool(eligible) and (min_precision > 0.0 or min_recall > 0.0)
    pool = eligible if eligible else list(rows)
    pool.sort(key=lambda r: (r[metric], r[prec_key]), reverse=True)
    return pool, floored


# ---------------------------------------------------------------------------
# held-out split: partition part indices into (selection, report) deterministically
# ---------------------------------------------------------------------------

def _holdout_indices(part_nrs, frac, seed=12345):
    """Split indices into (selection, report) by hashing part_nr -- stable across
    runs and independent of the train/val hash so it does not disturb that split."""
    report = []
    for i, pid in enumerate(part_nrs):
        h = int(hashlib.sha256(("ho:%d:%s" % (seed, pid)).encode()).hexdigest(), 16)
        if (h % 1000) / 1000.0 < frac:
            report.append(i)
    report_set = set(report)
    select = [i for i in range(len(part_nrs)) if i not in report_set]
    return select, report


def _subset(items, idx):
    return [items[i] for i in idx]


def _parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _print_table(rows, top, split, metric):
    hdr = ("  thr  votes   nms  dist |   F1(micro)  prec   recall |    TP    FP   FN")
    print("\n=== top %d decode configs on '%s' (ranked by %s) ==="
          % (min(top, len(rows)), split, metric))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows[:top]:
        print("  %.2f   %3d  %5.1f  %4.1f | %9.4f  %.4f  %.4f | %5d %5d %4d"
              % (r["heatmap_thresh"], r["min_votes"], r["nms_clearance_mm"],
                 r["dist_thresh_mm"], r["micro_f1"], r["micro_precision"],
                 r["micro_recall"], r["tp"], r["fp"], r["fn"]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Sweep decode params on a trained CP regressor (no retraining)")
    ap.add_argument("ckpt", help="trained checkpoint (e.g. checkpoints/cp_knngraph_best.ckpt)")
    ap.add_argument("source", help="same corpus you trained on (dir / big JSON / part file)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="which split to tune on (default val -- the honest one)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="must match the training run so the val split is identical")
    ap.add_argument("--seed", type=int, default=0, help="must match the training run")
    ap.add_argument("--max-parts", type=int, default=None,
                    help="must match the training run (caps parts materialised)")
    ap.add_argument("--split-group", choices=["none", "prefix", "geometry"],
                    default="geometry",
                    help="must match the training run so the val split is identical "
                         "(leakage-safe grouping; see train_cp.py --split-group) -- "
                         "default 'geometry' matches train_1h.py's production default; "
                         "passing the wrong value here silently reconstructs a DIFFERENT "
                         "val split than the checkpoint was actually validated against")
    ap.add_argument("--extra-source", action="append", default=[], dest="extra_sources",
                    help="must match the training run (repeatable)")
    ap.add_argument("--keep-prefixes", default=None,
                    help="must match the training run's --keep-prefixes scope filter")
    ap.add_argument("--max-bbox-mm", type=float, default=None, dest="max_bbox_mm",
                    help="must match the training run's --max-bbox-mm scope filter, "
                         "else large out-of-scope parts pollute the sweep")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--op-cache-dir", default=None,
                    help="DiffusionNet operator cache (diffusionnet backbone only)")
    ap.add_argument("--max-gpu-verts", type=int, default=60000)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8",
                    help="comma-separated heatmap thresholds to try")
    ap.add_argument("--min-votes-list", default="1,2,3,4,6",
                    help="comma-separated min-votes values to try (deploy default is 1)")
    ap.add_argument("--nms-list", default="5,10,15",
                    help="comma-separated nms clearance (mm) values to try")
    ap.add_argument("--dist-list", default="5",
                    help="comma-separated match radii (mm): a prediction counts as a "
                         "hit within this distance of a GT point. This is a SCORING "
                         "knob, not a model param -- swept so its effect on TP/FP/FN "
                         "is explicit. Default 5mm (the historical fixed value)")
    ap.add_argument("--metric", choices=["micro_f1", "f1"], default="micro_f1",
                    help="rank configs by pooled (micro) or per-part-mean (macro) F1")
    ap.add_argument("--min-precision", type=float, default=0.0,
                    help="floor: ignore configs below this micro-precision when "
                         "picking the winner (0 = no floor). Guards against a "
                         "lopsided high-recall corner winning on F1 alone")
    ap.add_argument("--min-recall", type=float, default=0.0,
                    help="floor: ignore configs below this micro-recall (0 = no floor)")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="if >0, split the chosen split into a selection subset and a "
                         "held-out report subset: pick the config on selection, report "
                         "the HEADLINE F1 on the held-out parts (honest, not "
                         "selected-on-the-same-data). 0 = select and report on the "
                         "whole split (optimistic; printout is labelled as such)")
    ap.add_argument("--cache-arrs", default=None,
                    help="path to an .npz cache of the per-part (N,7) prediction "
                         "arrays. If present and valid (same ckpt+split) it is loaded "
                         "instead of re-running inference, so repeated grid sweeps are "
                         "instant; otherwise it is computed and written here")
    ap.add_argument("--top", type=int, default=15, help="how many rows to print")
    ap.add_argument("--out", default=None, help="write the full sweep table to this JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model, meta, backbone = cpr.load_model(args.ckpt, device=args.device)
    logger.info("loaded %s checkpoint <- %s", backbone, args.ckpt)

    samples, metas = _load_split(args.source, args.split, args.val_frac,
                                 args.seed, args.max_parts, args.split_group,
                                 extra_sources=args.extra_sources,
                                 keep_prefixes=args.keep_prefixes,
                                 max_bbox_mm=args.max_bbox_mm)
    if not samples:
        raise SystemExit(f"no parts in the '{args.split}' split (check --val-frac/--seed)")
    part_nrs = [m["part_nr"] for m in metas]
    total_gt = sum(len(s["gt_points"]) for s in samples)
    logger.info("%s split: %d parts, %d ground-truth connection points",
                args.split, len(samples), total_gt)

    # the expensive step, done once (or loaded from disk): each part's (N,7) array
    signature = _arrs_signature(args.ckpt, backbone,
                                "%s/%s" % (args.split, args.split_group),
                                args.val_frac, args.seed, args.max_parts, part_nrs)
    arrs = _load_arrs_cache(args.cache_arrs, signature)
    if arrs is None:
        infer = _build_infer(backbone, model, meta, args.device,
                             args.op_cache_dir, args.max_gpu_verts)
        logger.info("running inference once per part ...")
        arrs = [infer(s, pm) for s, pm in zip(samples, metas)]
        _save_arrs_cache(args.cache_arrs, signature, arrs)

    grid = build_grid(_parse_floats(args.thresholds), _parse_ints(args.min_votes_list),
                      _parse_floats(args.nms_list), _parse_floats(args.dist_list))
    logger.info("sweeping %d decode configurations ...", len(grid))

    # --- selection vs held-out report ---------------------------------------
    use_holdout = bool(args.holdout_frac and args.holdout_frac > 0.0)
    sel_idx, rep_idx = [], []                       # list[int]: empty until split
    rep_s, rep_m, rep_a = [], [], []                # held-out report subset
    if use_holdout:
        sel_idx, rep_idx = _holdout_indices(part_nrs, args.holdout_frac, seed=args.seed)
        if not sel_idx or not rep_idx:
            logger.warning("--holdout-frac=%g produced an empty selection/report "
                           "subset (%d/%d parts) -- falling back to select+report on "
                           "the whole split", args.holdout_frac, len(sel_idx), len(rep_idx))
            use_holdout = False

    if use_holdout:
        sel_s, sel_m, sel_a = _subset(samples, sel_idx), _subset(metas, sel_idx), _subset(arrs, sel_idx)
        rep_s, rep_m, rep_a = _subset(samples, rep_idx), _subset(metas, rep_idx), _subset(arrs, rep_idx)
        logger.info("held-out: selecting on %d parts, reporting on %d held-out parts",
                    len(sel_idx), len(rep_idx))
        rows = sweep_grid(sel_a, sel_s, sel_m, grid)
    else:
        rows = sweep_grid(arrs, samples, metas, grid)
    if not rows:
        raise SystemExit("no decode configuration produced a score (empty grid?)")

    ranked, floored = rank_rows(rows, metric=args.metric,
                                min_precision=args.min_precision,
                                min_recall=args.min_recall)
    if (args.min_precision or args.min_recall) and not floored:
        logger.warning("no config met --min-precision=%.2f/--min-recall=%.2f; "
                       "ranking the UNFILTERED grid instead", args.min_precision,
                       args.min_recall)

    _print_table(ranked, args.top, args.split, args.metric)

    # baseline = the old hard-coded decode (thr=0.3, min_votes=2, nms=5, dist=5)
    base = next((r for r in ranked if r["heatmap_thresh"] == 0.3 and r["min_votes"] == 2
                 and r["nms_clearance_mm"] == 5.0 and r["dist_thresh_mm"] == 5.0), None)
    best = ranked[0]
    if base:
        print("\n  old default (thr=0.30 votes=2 nms=5.0 dist=5.0): "
              "F1=%.4f  prec=%.4f  recall=%.4f  (FP=%d)"
              % (base["micro_f1"], base["micro_precision"],
                 base["micro_recall"], base["fp"]))
    print("  BEST  (thr=%.2f votes=%d nms=%.1f dist=%.1f):          "
          "F1=%.4f  prec=%.4f  recall=%.4f  (FP=%d)"
          % (best["heatmap_thresh"], best["min_votes"], best["nms_clearance_mm"],
             best["dist_thresh_mm"], best["micro_f1"], best["micro_precision"],
             best["micro_recall"], best["fp"]))

    # --- honesty: where the headline number comes from ----------------------
    if use_holdout:
        held = _score(rep_a, rep_s, rep_m, best["heatmap_thresh"], best["min_votes"],
                      best["nms_clearance_mm"], best["dist_thresh_mm"])
        print("\n  selection F1 (where the config was chosen) : %.4f  [%d parts]"
              % (best["micro_f1"], len(sel_idx)))
        print("  >>> HELD-OUT test F1 (honest headline)      : %.4f  "
              "prec=%.4f recall=%.4f  [%d parts the selection never saw]"
              % (held["micro_f1"], held["micro_precision"], held["micro_recall"],
                 len(rep_idx)))
        headline = {"selection_f1": best["micro_f1"], "holdout_f1": held["micro_f1"],
                    "holdout_precision": held["micro_precision"],
                    "holdout_recall": held["micro_recall"],
                    "n_select": len(sel_idx), "n_holdout": len(rep_idx)}
    else:
        print("\n  NOTE: this F1 is SELECTED-ON-VAL (optimistic) -- the config was "
              "chosen by\n        max-over-grid on the same parts it is scored on. "
              "Use --holdout-frac for\n        an honest held-out number.")
        headline = {"selected_on_val_f1": best["micro_f1"], "holdout": False}

    print("\nlock it in:")
    print("  python train_cp.py %s --backbone %s --heatmap-thresh %.2f "
          "--min-votes %d --nms-clearance-mm %.1f --dist-thresh-mm %.1f "
          "--resume --epochs <target>"
          % (args.source, backbone, best["heatmap_thresh"], best["min_votes"],
             best["nms_clearance_mm"], best["dist_thresh_mm"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"backbone": backbone, "split": args.split,
                       "n_parts": len(samples), "total_gt": total_gt,
                       "metric": args.metric, "floored": floored,
                       "headline": headline, "best": best, "rows": ranked}, fh, indent=2)
        print("\nfull sweep table written to", args.out)


if __name__ == "__main__":
    main()
