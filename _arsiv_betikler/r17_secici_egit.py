# -*- coding: utf-8 -*-
"""R1c: EKSEN SECICISINI EGIT + SINAVDA TEK ATISLA OLC.

Egitim verisi: `r16_secici_veri.npz` -- 8543 satir, 900 EGITIM parcasi (sinav kumesi
ve ikizleri DISARIDA). Her satir bir EKSEN ADAYI; etiket = bu aday GT yonune <=10 derece mi.

TEZ SADAKATI: `v_o` KONUMU degismez. Yalniz yon secilir; tezin ham `v_o - v_s`'si
adaylardan biri ("mevcut") olarak kalir ve sonuc onunla YAN YANA raporlanir.
"""
import sys, os, json, io, pickle
os.environ.setdefault("BA_ALLOW_SEEN","1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0,'.')
import numpy as np, trimesh, wire_gate, thesis_remesh
import brep_axes as BX
from sklearn.ensemble import RandomForestClassifier
from f2_12_veri_kolu import egit
from big_arbiter import eligible
from infer_step_cp import step_to_mesh
from r15_eksen_secici import adaylar
from r16_secici_veri import ozn, AD

d = np.load("results/r16_secici_veri.npz")
X, y = d["X"], d["y"]
clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
clf.fit(X, y)
print(f"secici egitildi: {len(X)} satir, dogru aday %{100*y.mean():.0f}")
imp = sorted(zip(clf.feature_importances_, ["ic_cap","ort_cap","ileri","geri","maks","min",
              "n_ileri","n_geri","mevcut_uyum","eksen_hizali","boy/cap","aday_tipi"]),
             reverse=True)
print("  en onemli 4 oznitelik:", [f"{n} {v:.2f}" for v, n in imp[:4]])

sv = json.load(io.open('results/d5_4_sinav_kumesi.json', encoding='utf-8'))
G6 = {r['pid']: r for r in pickle.load(open('results/_der_yeni_g6.pkl','rb'))}
yol = {p: s for m, p, jf, s in eligible()}
m, _ = egit('results/zengin_parite_v3.npz')

def kos(D, ad):
    Am, As, Ak = [], [], []
    for r in D:
        if r['X'] is None or r.get('XR') is None: continue
        M = np.hstack([r['X'], r['XR']]).astype(float)
        if M.shape[1]*2 != m['n_feat']: continue
        k = wire_gate.karar_maskesi(wire_gate.karar_skoru(m, M))
        P = np.asarray(r['P'],float)[k]; Pd = np.asarray(r['Pd'],float)[k]
        G = np.asarray(r['G'],float); Gd = np.asarray(r['Gd'],float)
        if not len(P) or not len(G): continue
        try:
            Vr, Fr = step_to_mesh(yol[r['pid']])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            mesh = trimesh.Trimesh(vertices=np.ascontiguousarray(V,np.float64),
                                   faces=np.ascontiguousarray(F,np.int64), process=False)
            cyl = BX.cylinders(yol[r['pid']])
        except Exception:
            continue
        dd_ = P[:,None,:]-G[None,:,:]; al = (dd_*Gd[None,:,:]).sum(-1)
        pe = np.linalg.norm(dd_-al[...,None]*Gd[None,:,:], axis=-1)
        pe2 = np.where(np.abs(al)<=40, pe, np.inf)
        nP = Pd/(np.linalg.norm(Pd,axis=1,keepdims=True)+1e-9)
        tol = max(3.0, 0.06*float(r['diag'])); up, ug = set(), set()
        for q, a_, b_ in sorted((pe2[a,b],a,b) for a in range(len(P)) for b in range(len(G))):
            if q > tol or a_ in up or b_ in ug: continue
            up.add(a_); ug.add(b_)
            g = Gd[b_]; cand = adaylar(mesh, cyl, P[a_], nP[a_])
            aci = [np.degrees(np.arccos(min(1,abs(float(v@g))))) for _n, v in cand]
            F_ = np.array([ozn(mesh, P[a_], v, nP[a_]) + [float(AD.index(n))]
                           for n, v in cand], float)
            s = clf.predict_proba(F_)[:, 1]
            Am.append(aci[0]); As.append(aci[int(np.argmax(s))]); Ak.append(min(aci))
    A, B, C = np.array(Am), np.array(As), np.array(Ak)
    print(f"\n=== {ad} (n={len(A)}) ===")
    print(f"{'':<18}{'aci<=10':>10}{'DIK>80':>9}")
    for nm, Z in (("MEVCUT (tez zinciri)", A), ("SECICI (ogrenilmis)", B), ("KAHIN (tavan)", C)):
        print(f"{nm:<18}{100*(Z<=10).mean():>9.0f}%{100*(Z>80).mean():>8.0f}%")
    fark = 100*((B<=10).mean()-(A<=10).mean())
    print(f"  fark {fark:+.1f} puan -> {'GECTI' if fark>=5 else 'GECMEDI'} (GO: >=+5)")
    return fark

import olcum_kumesi as OK
SIN = [G6[p] for p in sorted(set(sv['pidler'])) if p in G6][:80]
D194_e, _ = OK.kume('results/_der_tam.pkl')
D194 = [G6[p] for p in sorted({r['pid'] for r in D194_e}) if p in G6][:80]
kos(SIN, "GORULMEMIS sinav"); kos(D194, "194luk (gerileme kontrolu)")
