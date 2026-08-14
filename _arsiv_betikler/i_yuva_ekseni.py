# -*- coding: utf-8 -*-
"""I: YUVA ekseni -- duzlem ciftlerinden analitik eksen, SADECE silindir kolunun sustugu yerde.

A (silindir) urune girdi ve eksen hatasini %24.9 -> %15.8 indirdi. Ama silindir bulamadigi
adaylarda susuyor. Bu kol tam o boslugu hedefler: yuva/kelepce girisleri silindir degil,
DUZLEM duvarli kanallar; ekseni yine tum duvar normallerine diktir.

TASARIM: once silindir denenir (urundeki hali). SADECE None donerse duzlem kolu calisir.
Boylece A'nin kazandigi yer BOZULMAZ -- yalnizca bos kalan yer doldurulur.

KILL: eksen >15d hatasi %15.8'in altina inmezse VE robot-hazir F1 0.5272'yi gecmezse duser.
"""
import os, sys, json, pickle
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}


def main():
    from big_arbiter import eligible
    import brep_axes as B
    cache = pickle.load(open("results/_tolerans_cache6.pkl", "rb"))
    LOCK = set(json.load(open("results/split_lock.json"))["locked_parts"])
    parts = []
    for m, p, jf, s in eligible():
        if p in LOCK: continue
        try: n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception: continue
        if n > 0: parts.append((m, p, jf, s, n))
    rng = np.random.RandomState(202)
    lo = [x for x in parts if x[4] < 8]; hi = [x for x in parts if x[4] >= 8]
    sel = ([lo[i] for i in rng.choice(len(lo), 70, replace=False)] +
           [hi[i] for i in rng.choice(len(hi), 30, replace=False)])
    assert len(sel) == len(cache) and all(s[4] == c["n"] for s, c in zip(sel, cache))
    print(f"{len(cache)} parca hizalandi\n", flush=True)

    def apply(dist, minf, flat, turn):
        out = []; used = tried = 0
        B.REJECT.clear()
        for (m, p, jf, stp, n), r in zip(sel, cache):
            try:
                cyl = B.cylinders(stp); pl = B.planes(stp)
            except Exception:
                out.append(r); continue
            Qd = np.array(r["Qd"], float).copy()
            for i in range(len(r["Q"])):
                # urundeki silindir kolu ZATEN uygulandi (cache6 onunla uretildi).
                # Burada yalnizca silindirin BULAMADIGI adaylara duzlem kolu denenir.
                a_cyl = B.axis_at(r["Q"][i], Qd[i], cyl, max_off_mm=5.0, max_turn_deg=60.0)
                if a_cyl is not None:
                    continue                       # A kazandi -> DOKUNMA
                tried += 1
                a_pl = B.axis_from_planes(r["Q"][i], Qd[i], pl, max_dist_mm=dist,
                                          min_faces=minf, flat_ratio=flat, max_turn_deg=turn)
                if a_pl is not None:
                    Qd[i] = a_pl; used += 1
            out.append(dict(r, Qd=Qd))
        return out, used, tried, dict(B.REJECT)

    def ang(c):
        A = []
        for r in c:
            Q, Qd, Gt, Gd = r["Q"], r["Qd"], r["G"], r["Gd"]
            if not len(Q) or not len(Gt): continue
            diff = Q[:, None, :] - Gt[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            t = max(3.0, 0.06 * r["diag"]); used = set(); hit = np.zeros(len(Gt), bool)
            for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(Q)) for b in range(len(Gt))):
                if d_ > t or abs(al[a_, b_]) > 40 or a_ in used or hit[b_]: continue
                hit[b_] = True; used.add(a_)
                A.append(np.degrees(np.arccos(min(1.0, abs(float(np.dot(Qd[a_], Gd[b_])))))))
        return np.array(A) if A else np.array([0.0])

    def f1(c, tol, am, pct=False):
        agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
        for r in c:
            Q, Qd, Gt, Gd = r["Q"], r["Qd"], r["G"], r["Gd"]
            hit = np.zeros(len(Gt), bool); used = set()
            if len(Q) and len(Gt):
                diff = Q[:, None, :] - Gt[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                an = np.degrees(np.arccos(np.clip(np.abs(Qd @ Gd.T), 0, 1)))
                t = max(3.0, 0.06 * r["diag"]) if pct else tol
                pe = np.where((np.abs(al) > 40) | (an > am), np.inf, pe)
                for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(Q)) for b in range(len(Gt))):
                    if d_ > t or a_ in used or hit[b_]: continue
                    hit[b_] = True; used.add(a_)
            tp = int(hit.sum()); k = "cok" if r["n"] >= 8 else "dusuk"
            agg[k][0] += tp; agg[k][1] += len(Q) - tp; agg[k][2] += len(Gt) - tp
        o = {}
        for k, (T, Fp, Fn) in agg.items():
            p = T / max(T + Fp, 1); rc = T / max(T + Fn, 1); o[k] = 2 * p * rc / max(p + rc, 1e-9)
        return sum(W[k] * o[k] for k in W), o["cok"], o["dusuk"]

    A0 = ang(cache); r0, c0, l0 = f1(cache, 2.0, 10); d0, _, _ = f1(cache, 0, 180, True)
    print(f"{'ayar':<34}{'>15d':>7}{'>45d':>7}{'ROBOT':>8}{'cok':>7}{'dus':>7}{'tespit':>8}{'kullanim':>10}")
    print(f"{'MEVCUT (yalniz silindir)':<34}{(A0>15).mean()*100:>6.1f}%{(A0>45).mean()*100:>6.1f}%"
          f"{r0:>8.4f}{c0:>7.4f}{l0:>7.4f}{d0:>8.4f}{'-':>10}")
    best = None
    for dist, minf, flat, turn in ((6.0,3,0.35,60.0),(8.0,3,0.35,60.0),(6.0,4,0.20,45.0),
                                   (10.0,3,0.50,60.0),(6.0,3,0.35,30.0)):
        c, u, t, rej = apply(dist, minf, flat, turn)
        A = ang(c); r_, ch, cl = f1(c, 2.0, 10); d_, _, _ = f1(c, 0, 180, True)
        lab = f"I dist{dist} minf{minf} flat{flat} t{turn}"
        print(f"{lab:<34}{(A>15).mean()*100:>6.1f}%{(A>45).mean()*100:>6.1f}%{r_:>8.4f}"
              f"{ch:>7.4f}{cl:>7.4f}{d_:>8.4f}{f'{u}/{t}':>10}", flush=True)
        if best is None or r_ > best[0]: best = (r_, lab, (A>15).mean(), rej)
    print(f"\nen iyi: {best[1]}  robot {best[0]:.4f}  >15d %{best[2]*100:.1f}")
    print(f"red sebepleri: {best[3]}")
    print(f"KILL: robot 0.5272'yi ve >15d %15.8'i gecmeli.")


if __name__ == "__main__":
    main()
