"""Summarize CP training histories for plateau/collapse decisions."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def _metric(rec: dict[str, Any], name: str) -> float:
    for key in (f"val_{name}", name):
        val = rec.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return float("nan")


def _field(rec: dict[str, Any], name: str, default: float = float("nan")) -> float:
    val = rec.get(name)
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _run_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("_history"):
        return stem[: -len("_history")]
    return stem


def summarize_history(path: str, metric_name: str, baseline: float, max_gap: float) -> dict[str, Any] | None:
    with open(path, encoding="utf-8") as fh:
        history = json.load(fh)
    if not isinstance(history, list) or not history:
        return None
    scored = [(idx, rec, _metric(rec, metric_name)) for idx, rec in enumerate(history) if isinstance(rec, dict)]
    scored = [(idx, rec, score) for idx, rec, score in scored if score == score]
    if not scored:
        return None
    _idx, best, best_score = max(scored, key=lambda item: item[2])
    _last_idx, last, last_score = scored[-1]
    gap = best_score - last_score
    eps = 5e-5
    return {
        "run": _run_name(path),
        "path": path,
        "evals": len(scored),
        "best_epoch": int(_field(best, "epoch", -1)),
        "best_micro_f1": best_score,
        "best_micro_precision": _field(best, "val_micro_precision"),
        "best_micro_recall": _field(best, "val_micro_recall"),
        "best_tp": int(_field(best, "val_total_tp", -1)),
        "best_fp": int(_field(best, "val_total_fp", -1)),
        "best_fn": int(_field(best, "val_total_fn", -1)),
        "last_epoch": int(_field(last, "epoch", -1)),
        "last_micro_f1": last_score,
        "gap": gap,
        "beats_baseline": best_score + eps >= baseline,
        "stable_last": gap <= max_gap + eps,
    }


def collect(paths_or_patterns: list[str]) -> list[str]:
    out: list[str] = []
    for item in paths_or_patterns:
        matches = glob.glob(item)
        out.extend(matches or [item])
    return sorted(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", action="append", default=[], help="history JSON path or glob")
    ap.add_argument("--metric", default="micro_f1", help="metric suffix, default micro_f1")
    ap.add_argument("--baseline", type=float, default=0.6383, help="minimum acceptable best score")
    ap.add_argument("--max-gap", type=float, default=0.02, help="maximum acceptable best-last gap")
    ap.add_argument("--json-out", default=None, help="optional machine-readable summary path")
    args = ap.parse_args()

    patterns = args.pattern or ["checkpoints/*_history.json"]
    rows = []
    for path in collect(patterns):
        if os.path.exists(path):
            row = summarize_history(path, args.metric, args.baseline, args.max_gap)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: (row["best_micro_f1"], -row["gap"]), reverse=True)

    print(f"baseline={args.baseline:.4f} max_gap={args.max_gap:.4f} metric={args.metric}")
    print(
        "run".ljust(26),
        "evals",
        "best(ep/f1)",
        "last(ep/f1)",
        "gap",
        "P/R",
        "TP/FP/FN",
        "decision",
    )
    for row in rows:
        decision = []
        decision.append("best-ok" if row["beats_baseline"] else "best-low")
        decision.append("stable" if row["stable_last"] else "collapsed")
        pr = f"{row['best_micro_precision']:.3f}/{row['best_micro_recall']:.3f}"
        counts = f"{row['best_tp']}/{row['best_fp']}/{row['best_fn']}"
        print(
            row["run"].ljust(26),
            str(row["evals"]).rjust(5),
            f"{row['best_epoch']:>3}/{row['best_micro_f1']:.4f}",
            f"{row['last_epoch']:>3}/{row['last_micro_f1']:.4f}",
            f"{row['gap']:.4f}",
            pr,
            counts,
            ",".join(decision),
        )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
