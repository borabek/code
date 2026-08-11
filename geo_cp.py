# -*- coding: utf-8 -*-
"""GEOMETRIK CP DEDEKTORU -- ML YOK, EGITIM YOK. Sadece JSON mesh geometrisinden oyuk/acikligi bul.
G0: konkavlik sinyali + validation (uretici CP'leri yuksek-konkav bolgede mi?) + ilk dedektor+eval.
Hizalama sorunu YOK: mesh ve CP ayni JSON frame'inde."""
import os, sys, json, glob, argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def load_json_mesh(jf):
    j = json.load(open(jf, encoding="utf-8-sig"))
    V = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
    idx = j["Graphic3d"].get("Indices", [])
    F = np.array(idx, int).reshape(-1, 3) if idx else None
    G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
    Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]]
                  for c in j.get("ConnectionPoints", []) if "InsertDirection" in c], float)
    return V, F, G, Gd


def concavity(mesh, R=3.0):
    """yerel-oyukluk skoru (ray-casting'siz): her vertex icin, R-komsularin OUTWARD-normal
    boyunca ortalama izdusumu. Konkav (oyuk icinde) = komsular disarida/yukarida -> yuksek pozitif."""
    V = mesh.vertices.view(np.ndarray); N = mesh.vertex_normals.view(np.ndarray)
    tree = cKDTree(V)
    sc = np.zeros(len(V))
    nbrs = tree.query_ball_point(V, R)
    for i, nb in enumerate(nbrs):
        if len(nb) < 4: continue
        d = V[nb] - V[i]                      # komsulara vektorler
        nrm = np.linalg.norm(d, axis=1) + 1e-9
        proj = (d @ N[i]) / nrm               # outward-normal boyunca (birim)
        sc[i] = float(np.mean(proj))          # >0 = komsular disarida = oyuk icindeyim (konkav)
    return sc


def validate(parts):
    """uretici CP'lerine yakin vertexlerin konkavligi vs genel -- sinyal CP'leri buluyor mu?"""
    print(f"{'pid':>12} {'mfg':>4} {'CP-near concav':>16} {'part-mean':>10} {'delta':>6}")
    ratios = []
    for jf in parts:
        V, F, G, Gd = load_json_mesh(jf)
        if F is None or not len(G): continue
        mesh = trimesh.Trimesh(V, F, process=False)
        sc = concavity(mesh)
        tree = cKDTree(V)
        near = set()
        for g in G:
            near.update(tree.query_ball_point(g, 4.0))
        near = list(near)
        if not near: continue
        cp_conc = float(np.mean(sc[near])); all_conc = float(np.mean(sc))
        ratio = cp_conc / (abs(all_conc) + 1e-6)
        ratios.append(cp_conc - all_conc)
        pid = os.path.basename(jf).split("_")[0].split(".")[-1]
        mfg = os.path.basename(jf).split(".")[0]
        print(f"{pid:>12} {mfg:>4} {cp_conc:>16.3f} {all_conc:>10.3f} {cp_conc-all_conc:>+6.3f}")
    print(f"\nMEAN delta (CP-near - overall): {np.mean(ratios):+.3f}  "
          + ("-> SIGNAL IS USEFUL (CPs are more concave)" if np.mean(ratios) > 0.02 else "-> weak signal, try another approach"))


def detect(mesh, conc_thr=0.08, eps=4.0, min_pts=4, min_depth=1.5):
    """geometrik CP dedektoru: konkav vertexleri kumele -> her aciklik icin agiz-merkezi + eksen + derinlik.
    Filtre: derinlik>=min_depth (duz kontaklari at). ML YOK."""
    V = mesh.vertices.view(np.ndarray); N = mesh.vertex_normals.view(np.ndarray)
    sc = concavity(mesh)
    cand = np.where(sc > conc_thr)[0]
    if len(cand) < min_pts: return []
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(V[cand])
    cps = []
    for c in set(lab) - {-1}:
        idx = cand[lab == c]
        if len(idx) < min_pts: continue
        axis = -N[idx].mean(0); axis /= (np.linalg.norm(axis) + 1e-9)   # iceri = -normal
        ctr = V[idx].mean(0)
        depth = float(((V[idx] - ctr) @ axis).max() - ((V[idx] - ctr) @ axis).min())  # eksen boyu yayilim
        if depth < min_depth: continue
        cps.append({"point": ctr.tolist(), "axis": axis.tolist(), "depth": depth, "n": int(len(idx))})
    return cps


def greedy_match(P, G, tol):
    if not len(P) or not len(G): return 0, len(P), len(G)
    D = np.linalg.norm(np.asarray(P)[:, None] - np.asarray(G)[None], axis=-1)
    order = sorted((D[i, j], i, j) for i in range(len(P)) for j in range(len(G)) if D[i, j] <= tol)
    up, ug = set(), set(); tp = 0
    for d, i, j in order:
        if i in up or j in ug: continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(P) - tp, len(G) - tp


def evaluate(parts, **kw):
    T = Fp = Fn = 0; per = {}
    for jf in parts:
        try:
            V, F, G, Gd = load_json_mesh(jf)
            if F is None or not len(G): continue
            mesh = trimesh.Trimesh(V, F, process=False)
            cps = detect(mesh, **kw)
            P = [c["point"] for c in cps]
            tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))
            tp, fp, fn = greedy_match(P, G, tol)
            mfg = os.path.basename(jf).split(".")[0]
            per.setdefault(mfg, [0, 0, 0])
            per[mfg][0] += tp; per[mfg][1] += fp; per[mfg][2] += fn
            T += tp; Fp += fp; Fn += fn
        except Exception:
            continue
    pr = T / max(T + Fp, 1); rc = T / max(T + Fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    print(f"\nGEOMETRIC detector (NO ML): F1={f1:.3f} P={pr:.3f} R={rc:.3f} (TP{T} FP{Fp} FN{Fn})")
    for m, (t, f, n) in sorted(per.items()):
        p = t / max(t + f, 1); r = t / max(t + n, 1)
        print(f"  {m:>7}: F1={2*p*r/max(p+r,1e-9):.3f} P={p:.3f} R={r:.3f} ({t+n} CP)")


# ============ G3: silindir/tup-fit dedektor (naif detect'ten cok daha az FP) ============
def opening_signal(mesh, R=3.0):
    """G1: konkavlik + convex-hull mesafesi birlesik -> daha temiz aday (naif tek-sinyalden iyi)."""
    V = mesh.vertices.view(np.ndarray)
    sc = concavity(mesh, R=R)                      # yerel-oyukluk (mevcut)
    try:
        hull = mesh.convex_hull
        from trimesh.proximity import ProximityQuery
        hd = -ProximityQuery(hull).signed_distance(V)   # +: hull ICINDE (girinti) = recessed
        hd = np.clip(hd, 0, None); hd = hd/(hd.max()+1e-9)
    except Exception:
        hd = np.zeros(len(V))
    return sc, hd


def detect2(mesh, conc_thr=0.10, eps=4.0, min_pts=6, min_depth=2.0, max_width=16.0, min_ar=0.12):
    """G3: konkav+recessed vertexleri kumele -> her kume icin TUP-fit (eksen/derinlik/genislik) ->
    fiziksel filtre. Cable acikligi = DERIN+DAR tup (yuksek depth/width); snap/kenar = sig -> elenir."""
    V = mesh.vertices.view(np.ndarray); N = mesh.vertex_normals.view(np.ndarray)
    sc, hd = opening_signal(mesh)
    cand = np.where((sc > conc_thr) | (hd > 0.3))[0]
    if len(cand) < min_pts: return []
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(V[cand])
    cps = []
    for c in set(lab) - {-1}:
        idx = cand[lab == c]
        if len(idx) < min_pts: continue
        P = V[idx]; ctr = P.mean(0)
        axis = -N[idx].mean(0); axis /= (np.linalg.norm(axis)+1e-9)   # iceri (tup ekseni)
        al = (P - ctr) @ axis                                          # eksen boyu
        perp = P - ctr - al[:, None]*axis; width = 2*np.median(np.linalg.norm(perp, axis=1))
        depth = float(al.max() - al.min())
        ar = depth/(width+1e-9)                                        # aspect: tup-luk
        if depth < min_depth or width < 1.0 or width > max_width or ar < min_ar: continue
        mouth = P[np.argmax(al)] if abs(al.max())>abs(al.min()) else ctr  # dis-yuzey ucu
        cps.append({"point": mouth.tolist(), "axis": axis.tolist(), "depth": depth, "width": float(width), "n": int(len(idx))})
    return cps


def evaluate2(parts, **kw):
    T=Fp=Fn=0; per={}
    for jf in parts:
        try:
            V,F,G,Gd=load_json_mesh(jf)
            if F is None or not len(G): continue
            import trimesh as _tm; mesh=_tm.Trimesh(V,F,process=False)
            cps=detect2(mesh, **kw); P=[c["point"] for c in cps]
            tol=max(3.0,0.06*float(np.linalg.norm(V.max(0)-V.min(0))))
            tp,fp,fn=greedy_match(P,G,tol)
            mfg=os.path.basename(jf).split(".")[0]; per.setdefault(mfg,[0,0,0])
            per[mfg][0]+=tp; per[mfg][1]+=fp; per[mfg][2]+=fn; T+=tp; Fp+=fp; Fn+=fn
        except Exception: continue
    pr=T/max(T+Fp,1); rc=T/max(T+Fn,1); f1=2*pr*rc/max(pr+rc,1e-9)
    print(f"\nGEOMETRIC G3 (tube-fit, NO ML): F1={f1:.3f} P={pr:.3f} R={rc:.3f} (TP{T} FP{Fp} FN{Fn})")
    for m,(t,f,n) in sorted(per.items()):
        p=t/max(t+f,1); r=t/max(t+n,1); print(f"  {m:>7}: F1={2*p*r/max(p+r,1e-9):.3f} P={p:.3f} R={r:.3f} ({t+n} CP)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="validate", choices=["validate","detect","eval","eval2"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mfg", default="")
    ap.add_argument("--ds", default="_ds1/DataSet")
    ap.add_argument("--conc-thr", type=float, default=0.08)
    ap.add_argument("--min-depth", type=float, default=1.5)
    a = ap.parse_args()
    pat = f"{a.mfg}*ElectricalTerminal*.json" if a.mfg else "*ElectricalTerminal*.json"
    parts = sorted(glob.glob(os.path.join(a.ds, pat)))[:a.n]
    print(f"{len(parts)} parts ({a.ds}, mfg={a.mfg or 'all'})\n")
    if a.mode == "validate":
        validate(parts)
    elif a.mode == "eval":
        evaluate(parts, conc_thr=a.conc_thr, min_depth=a.min_depth)
    elif a.mode == "eval2":
        evaluate2(parts, conc_thr=a.conc_thr, min_depth=a.min_depth)
