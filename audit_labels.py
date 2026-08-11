# -*- coding: utf-8 -*-
"""ETIKET DENETIMI: insan-boyali CableEntry bolgeleri URETICI CP'leriyle ortusuyor mu?
NEDEN: kullanici 'arada yanlis etiket yapmis olabilirim' dedi (_recall_v2 95, _recall_pxc 78 parca).
Bu etiketler URUN modelinde YOK (urun 07-23/24'te donduruldu, etiketler 07-27) -- yani mevcut
sonuclari etkilemiyor. Ama gelecekte kullanilacaklarsa kalitelerini bilmemiz gerek.
OLCUM: her boyali parcada, boyanan CableEntry bileseninin merkezi en yakin uretici CP'ye kac mm?
  yakin (<= tol)  -> etiket DOGRU yerde
  uzak            -> ya gercek listelenmemis aciklik ya HATA
Ayrica: uretici CP'lerinin kaci boyanmis (recall) -> eksik boyama var mi.
Kullanim: python audit_labels.py [_label_targets_recall_v2 ...]"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree

CE = 3           # CableEntry sinif id
DIRS = sys.argv[1:] or ["_label_targets_recall_v2", "_label_targets_recall_pxc", "_label_targets_recall"]


def load_obj(p):
    V, F = [], []
    for ln in open(p):
        if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.array(V, float), np.array(F, int)


def components(V, F, mask):
    """boyali vertexleri baglantili bilesenlere ayir (mesh kenarlari uzerinden)"""
    idx = np.where(mask)[0]
    if not len(idx): return []
    s = set(idx.tolist()); adj = {i: [] for i in idx}
    for tri in F:
        for a in tri:
            if a in s:
                for b in tri:
                    if b in s and b != a: adj[a].append(b)
    seen = set(); out = []
    for i in idx:
        if i in seen: continue
        st = [i]; comp = []
        while st:
            x = st.pop()
            if x in seen: continue
            seen.add(x); comp.append(x); st.extend(adj[x])
        out.append(comp)
    return out


os.environ["BA_ALLOW_SEEN"] = "1"
from big_arbiter import eligible
from cad_eval import align_frames
from infer_step_cp import step_to_mesh
paths = {p: (jf, stp) for m, p, jf, stp in eligible()}

for d in DIRS:
    if not os.path.isdir(d): continue
    parts = sorted(os.path.basename(os.path.normpath(x)) for x in glob.glob(d + "/*/"))
    near_all, far_all, cov_all = [], [], []
    n_ok = 0
    for pid in parts:
        of = os.path.join(d, pid, f"{pid}.obj"); lf = os.path.join(d, pid, f"{pid}.labels.txt")
        if not (os.path.exists(of) and os.path.exists(lf)) or pid not in paths: continue
        try:
            V, F = load_obj(of)
            L = np.array([int(x) for x in open(lf).read().split()], np.int64)
            if len(L) != len(V): continue
            jf, stp = paths[pid]
            j = json.load(open(jf, encoding="utf-8-sig"))
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
            if not len(G): continue
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            Vr, _ = step_to_mesh(stp)
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R                                   # uretici CP -> mesh frame
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            tol = max(6.0, 0.10 * diag)                        # etiket icin GENIS tolerans
            comps = components(V, F, L == CE)
            if not comps: continue
            centers = np.array([V[c].mean(0) for c in comps])
            tr = cKDTree(Gm)
            dist, _ = tr.query(centers)
            near_all += [float(x) for x in dist if x <= tol]
            far_all += [float(x) for x in dist if x > tol]
            # kapsama: uretici CP'lerinin kaci boyanmis bir bilesene yakin
            tr2 = cKDTree(centers); d2, _ = tr2.query(Gm)
            cov_all.append(float((d2 <= tol).mean()))
            n_ok += 1
        except Exception:
            continue
    tot = len(near_all) + len(far_all)
    if not tot:
        print(f"{d:32s} (olculemedi)"); continue
    print(f"{d:32s} {n_ok:3d} parca | boyali bilesen {tot:4d} | "
          f"URETICI CP'ye YAKIN %{100*len(near_all)/tot:5.1f} | UZAK %{100*len(far_all)/tot:5.1f} | "
          f"uretici CP kapsamasi %{100*np.mean(cov_all):5.1f}")
print("\nYORUM: 'UZAK' orani yuksekse ya listelenmemis gercek aciklik ya ETIKET HATASI.")
print("       Kiyas icin ayni olcumu ESKI (guvenilen) turlarda da yap -- fark varsa yeni tur supheli.")
