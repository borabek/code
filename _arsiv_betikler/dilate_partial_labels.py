# -*- coding: utf-8 -*-
"""Grow the annotator's TIGHT CableEntry marks out to the CORPUS convention.

WHY: the 20 human-labelled parts mark only the core of each wire opening
(1.45% of vertices) while the 71 expert corpus parts mark the whole opening
region (4.99%). Training on both at once teaches two conflicting definitions
-- measured: val Connection IoU 0.676 -> 0.630, arbiter CP F1 0.520 -> 0.351.
The MARK LOCATIONS are correct (verified visually); only the extent differs,
so translate the extent instead of discarding the labels.

HOW: breadth-first ring expansion over mesh edges from the marked vertices,
one ring at a time, until the CableEntry fraction reaches --target. Expansion
is capped per connected component so a big region cannot swallow a small one,
and never crosses into vertices the annotator gave a non-Housing class.
"""
import os, sys, glob, shutil, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_label_helper import load_obj

CE = 3  # CableEntry class id (connector3d.CABLE_ENTRY)


def adjacency(nv, F):
    """Vertex neighbour lists from triangle edges."""
    nb = [set() for _ in range(nv)]
    for a, b, c in F:
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    return [np.fromiter(s, dtype=np.int64) for s in nb]


def components(seed_idx, nb):
    """Connected components of the marked vertex set (restricted to marked)."""
    marked = set(int(i) for i in seed_idx)
    seen, comps = set(), []
    for s in marked:
        if s in seen: continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            v = stack.pop(); comp.append(v)
            for u in nb[v]:
                u = int(u)
                if u in marked and u not in seen:
                    seen.add(u); stack.append(u)
        comps.append(np.array(comp, np.int64))
    return comps


def dilate(V, F, L, target_frac, max_growth):
    """Ring-expand each CableEntry component until total CE fraction >= target_frac.

    Per-component growth is capped at max_growth x its seed size so the largest
    region does not absorb the mesh while small ones stay unchanged.
    """
    nv = len(V)
    nb = adjacency(nv, F)
    seeds = np.where(L == CE)[0]
    if not len(seeds):
        return L.copy(), 0
    comps = components(seeds, nb)
    out = L.copy()
    # frontier per component; blocked = vertices the annotator assigned another class
    blocked = (L != CE) & (L != 0)
    frontier = [set(int(v) for v in c) for c in comps]
    owned = [set(int(v) for v in c) for c in comps]
    caps = [max_growth * len(c) for c in comps]
    rings = 0
    while (out == CE).mean() < target_frac and rings < 200:
        grew = False
        for k in range(len(comps)):
            if len(owned[k]) >= caps[k] or not frontier[k]:
                continue
            nxt = set()
            for v in frontier[k]:
                for u in nb[v]:
                    u = int(u)
                    if out[u] == CE or blocked[u]:
                        continue
                    nxt.add(u)
            if not nxt:
                frontier[k] = set(); continue
            # respect the cap: if adding the whole ring overshoots, take the closest ones
            room = caps[k] - len(owned[k])
            nxt = list(nxt)
            if len(nxt) > room:
                ctr = V[list(owned[k])].mean(0)
                d = np.linalg.norm(V[nxt] - ctr, axis=1)
                nxt = [nxt[i] for i in np.argsort(d)[:int(room)]]
            out[np.array(nxt, np.int64)] = CE
            owned[k].update(nxt); frontier[k] = set(nxt); grew = True
        rings += 1
        if not grew:
            break
    return out, rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_label_targets")
    ap.add_argument("--dst", default="_label_targets_dilated")
    ap.add_argument("--target", type=float, default=0.05,
                    help="target CableEntry fraction (corpus mean is 0.0499)")
    ap.add_argument("--max-growth", type=float, default=8.0,
                    help="per-component cap as a multiple of its seed size")
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.src, "*"))):
        if not os.path.isdir(d): continue
        pid = os.path.basename(os.path.normpath(d))
        lf = os.path.join(d, f"{pid}.labels.txt")
        of = os.path.join(d, f"{pid}.obj")
        if not (os.path.exists(lf) and os.path.exists(of)): continue
        L = np.array([int(x) for x in open(lf).read().split()], np.int64)
        V, F = load_obj(of)
        if len(L) != len(V):
            print(f"  SKIP {pid}: {len(L)} labels vs {len(V)} verts"); continue
        if not (L == CE).any():
            print(f"  SKIP {pid}: no CableEntry marked"); continue
        before = float((L == CE).mean())
        L2, rings = dilate(V, F, L, a.target, a.max_growth)
        after = float((L2 == CE).mean())
        od = os.path.join(a.dst, pid); os.makedirs(od, exist_ok=True)
        shutil.copy(of, os.path.join(od, f"{pid}.obj"))
        open(os.path.join(od, f"{pid}.labels.txt"), "w").write("\n".join(str(int(x)) for x in L2))
        rows.append((pid, before, after, rings))
        print(f"  {pid:18s} {100*before:5.2f}% -> {100*after:5.2f}%  ({rings} rings)", flush=True)

    if rows:
        print(f"\n{len(rows)} parts | mean {100*np.mean([r[1] for r in rows]):.2f}% -> "
              f"{100*np.mean([r[2] for r in rows]):.2f}%  (corpus convention 4.99%)")
        print(f"-> {a.dst}")


if __name__ == "__main__":
    main()
