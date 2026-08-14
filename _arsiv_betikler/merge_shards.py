"""Merge the sharded label-corpus runs into ONE corpus, reproducing exactly what a
single serial `--label-corpus` run would have written.

Why this exists: labelling 2012 STEPs serially takes ~3.5h; six parallel shards do
it in ~50min. Each part is labelled independently and deterministically, so the
per-part JSONs are identical either way -- the ONLY thing a shard cannot do is the
GLOBAL geometric dedup (write_labeled_corpus keeps a `seen` signature dict and
skips a part whose geometry matches one already written, so re-downloads of the
same block under different catalog numbers don't leak across the train/val split).

Serial semantics: iterate files in sorted() order, keep the FIRST part of each
geometry signature, skip the rest. The shards are contiguous slices of that same
sorted list, so replaying the merge in shard order 0..N with a first-wins rule
reproduces the serial result byte for byte.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def geom_sig(V, n_cps):
    """Same signature write_labeled_corpus uses (step_openings._geom_sig)."""
    if not len(V):
        return ("empty", n_cps)
    return (tuple(np.round(V.max(0) - V.min(0), 0).tolist()), len(V), n_cps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard dirs IN THE ORDER the serial run would have seen them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--move", action="store_true",
                    help="MOVE the kept files instead of copying them. A copy needs "
                         "a second full corpus worth of disk (the 2026-07-13 merge "
                         "filled the drive and aborted at 1351/1758 parts); a move "
                         "within the same volume is a rename and needs none.")
    ap.add_argument("--sig-cache", default="_merge_sigs.json",
                    help="signature cache. A resumed/incremental merge normally "
                         "re-lists --out as a shard so new parts dedup against the "
                         "parts already there -- but that re-PARSES the whole corpus "
                         "(0.5-14MB of mesh JSON each) on every invocation: 3010 "
                         "parts took the full 10min and only 24 new ones landed. "
                         "Cached by (name, size, mtime), so a re-run pays only for "
                         "files it has not seen.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cache, parsed, hits = {}, 0, 0
    if args.sig_cache and os.path.exists(args.sig_cache):
        try:
            with open(args.sig_cache, encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception:                                   # noqa: BLE001
            cache = {}                                      # corrupt cache: just rebuild

    def signature(f):
        """geom_sig(f), memoised on (name, size, mtime)."""
        nonlocal parsed, hits
        st = os.stat(f)
        key = os.path.basename(f)
        ent = cache.get(key)
        if ent and ent[0] == st.st_size and ent[1] == st.st_mtime:
            hits += 1
            return tuple(ent[2][0]), ent[2][1], ent[2][2]
        d = json.load(open(f, encoding="utf-8"))
        V = np.array([[p["X"], p["Y"], p["Z"]]
                      for p in d["Graphic3d"]["Points"]], dtype=float)
        n_cps = len(d.get("ConnectionPoints") or [])
        sig = geom_sig(V, n_cps)
        parsed += 1
        cache[key] = [st.st_size, st.st_mtime, [list(sig[0]), sig[1], sig[2]]]
        return sig

    seen, kept, dupes, zero = {}, 0, 0, 0
    for sd in args.shards:                      # shard order == sorted file order
        for f in sorted(glob.glob(os.path.join(sd, "*.json"))):
            sig = signature(f)
            if sig in seen:
                dupes += 1
                continue
            seen[sig] = os.path.basename(f)
            if sig[2] == 0:                     # n_cps -- counted here, not by a
                zero += 1                       # second full re-parse of --out
            dst = os.path.join(args.out, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dst):   # --out re-listed as a shard
                if args.move:
                    shutil.move(f, dst)
                else:
                    shutil.copyfile(f, dst)
            kept += 1

    if args.sig_cache:
        with open(args.sig_cache, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    print(f"merged -> {args.out}: {kept} parts kept, {dupes} geometric duplicates "
          f"dropped, {zero} zero-CP ({100.0 * zero / max(kept, 1):.1f}%) "
          f"[parsed {parsed}, cached {hits}]")


if __name__ == "__main__":
    main()
