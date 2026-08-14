"""Pre-training audit of a label corpus: everything that has silently cost a run before.

Each check below exists because it ONCE bit us (RESULTS section in brackets):
  1 corrupt/truncated JSON      -- 19 half-written files from the 2026-07-13 disk-full
  2 schema                      -- missing Graphic3d/Points or ConnectionPoints
  3 NaN/Inf vertices            -- poisons the loss silently, never raises
  4 degenerate meshes           -- <4 verts: kNN graph build divides by zero
  5 vertex-count tail           -- the EdgeConv memory cliff (~8000 verts) [T1200 note]
  6 zero-CP parts               -- legitimate (~3%) but must not be a family wipe-out
  7 CP outside the mesh bbox    -- a label the model can never regress to
  8 duplicate geometry          -- same block, two catalog numbers -> train/val leakage
  9 out-of-scope parts present  -- 12/21 PXC parts are not terminal blocks [PXC audit]
 10 exclude-file would MATCH    -- the guard matches FULL strings; wscaduniverse_<id>
                                   names silently never fire it [11r]
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--exclude-parts-file")
    ap.add_argument("--vert-cap", type=int, default=8000,
                    help="EdgeConv memory cliff; parts above this get subsampled")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, "*.json")))
    print(f"corpus: {a.corpus}  ({len(files)} parts)\n")

    corrupt, bad_schema, nan_parts, tiny, no_cp, cp_out = [], [], [], [], [], []
    verts, cps, sigs = [], [], collections.defaultdict(list)

    for f in files:
        pn = os.path.basename(f)[:-5]
        try:
            d = load(f)
        except Exception as e:                                   # noqa: BLE001
            corrupt.append((pn, type(e).__name__))
            continue
        try:
            P = d["Graphic3d"]["Points"]
            V = np.array([[p["X"], p["Y"], p["Z"]] for p in P], float)
            CP = d.get("ConnectionPoints") or []
        except Exception:                                        # noqa: BLE001
            bad_schema.append(pn)
            continue
        if not np.isfinite(V).all():
            nan_parts.append(pn)
            continue
        if len(V) < 4:
            tiny.append((pn, len(V)))
            continue
        verts.append(len(V))
        cps.append(len(CP))
        if not CP:
            no_cp.append(pn)
        else:
            lo, hi = V.min(0), V.max(0)
            pad = 0.05 * (hi - lo + 1e-9)
            for c in CP:
                p = c["Point"]
                q = np.array([p["X"], p["Y"], p["Z"]], float)
                if (q < lo - pad).any() or (q > hi + pad).any():
                    cp_out.append(pn)
                    break
        sigs[(tuple(np.round(V.max(0) - V.min(0), 0)), len(V), len(CP))].append(pn)

    ok = len(files) - len(corrupt) - len(bad_schema) - len(nan_parts) - len(tiny)
    verts, cps = np.array(verts), np.array(cps)

    def head(n, title, items, fmt=str):
        flag = "FAIL" if items else "ok  "
        print(f"[{flag}] {n:2d}. {title}: {len(items)}")
        for it in items[:5]:
            print(f"           - {fmt(it)}")
        if len(items) > 5:
            print(f"           ... +{len(items) - 5} more")

    head(1, "corrupt / unparseable JSON", corrupt, lambda t: f"{t[0]} ({t[1]})")
    head(2, "bad schema (no Graphic3d/Points)", bad_schema)
    head(3, "NaN/Inf vertices", nan_parts)
    head(4, "degenerate mesh (<4 verts)", tiny, lambda t: f"{t[0]} ({t[1]} verts)")

    over = int((verts > a.vert_cap).sum())
    print(f"[{'warn' if over else 'ok  '}] 5. verts > {a.vert_cap} (subsampled at train): {over}"
          f"  ({100.0 * over / max(len(verts), 1):.1f}%)")
    print(f"           verts: min {verts.min()}  median {int(np.median(verts))}  "
          f"p95 {int(np.percentile(verts, 95))}  max {verts.max()}")

    pct = 100.0 * len(no_cp) / max(ok, 1)
    print(f"[{'warn' if pct > 5 else 'ok  '}] 6. zero-CP parts: {len(no_cp)} ({pct:.1f}%)"
          f"   -- expected ~3% (UKH large-bore); >5% means a family got wiped")
    print(f"           CPs/part: min {cps.min()}  median {int(np.median(cps))}  "
          f"max {cps.max()}  total {int(cps.sum())}")

    head(7, "CP outside mesh bbox (unlearnable label)", cp_out)

    dupes = {s: p for s, p in sigs.items() if len(p) > 1}
    n_dupe = sum(len(p) - 1 for p in dupes.values())
    print(f"[{'FAIL' if n_dupe else 'ok  '}] 8. duplicate geometry (train/val leakage): "
          f"{n_dupe} extra copies in {len(dupes)} groups")
    for s, p in list(dupes.items())[:3]:
        print(f"           - {p[0]} == {', '.join(p[1:3])}")

    if a.exclude_parts_file and os.path.exists(a.exclude_parts_file):
        names = {os.path.basename(f)[:-5] for f in files}
        ex = [l.strip() for l in open(a.exclude_parts_file) if l.strip()]
        exact = [e for e in ex if e in names]
        substr = [e for e in ex if e not in names and any(e in n for n in names)]
        print(f"[{'FAIL' if substr and not exact else 'ok  '}] 9/10. exclude-file "
              f"'{a.exclude_parts_file}': {len(ex)} entries -> {len(exact)} match a part "
              f"name EXACTLY, {len(substr)} match only as a SUBSTRING")
        if substr:
            print("           the guard compares FULL strings -- these would NOT be excluded:")
            for e in substr[:5]:
                hit = next(n for n in names if e in n)
                print(f"           - '{e}'  is inside  '{hit}'")
    else:
        print("[warn] 9/10. no --exclude-parts-file given (PXC out-of-scope guard OFF)")

    print(f"\nUSABLE FOR TRAINING: {ok - len(no_cp)} parts with CPs "
          f"(+{len(no_cp)} zero-CP negatives), {len(files) - ok} unusable")


if __name__ == "__main__":
    main()
