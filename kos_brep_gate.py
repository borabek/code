# -*- coding: utf-8 -*-
"""GENISLETILMIS HAVUZ + GATE REFIT: D6'da egit, D7'de (marka-disi) olc.

Havuz recall'un baglayici kisit oldugu olculdu (D7 seg-tek 0.6654 -> robot tavani
~0.32). B-rep onerileri eklenince recall 0.7852'ye cikiyor. Bu betik o havuzun
UCTAN UCA ne verdigini olcer.

GATE YENIDEN FIT EDILIR -- zorunlu: mevcut gate ESKI (seg-tek) aday dagiliminda
egitildi; yeni adaylari ona vermek [[gate-refit-minv4]] dersinin ihlali olurdu.
Egitim D6 (468 parca), sinav D7 (835 parca); marka kumeleri AYRIK.

Ozellikler diske ONBELLEKLENIR (`results/_brep_oz/`), is kesilirse ayni komut
kaldigi yerden devam eder. Ekran kapansa da arka planda surer.
"""
import json, os, pickle, sys, time
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import connector3d, wire_gate, brep_havuz, makbuz_hash
import d6_kayit, kanonik_d7 as K
from sina_kume import esle_macar

CE, CT = int(connector3d.CABLE_ENTRY), int(connector3d.CONTACT)
OZ = "results/_brep_oz"; os.makedirs(OZ, exist_ok=True)
S = K.step_haritasi()


def kume(ad, kayitlar, ob, cylf, acf):
    cy = pickle.load(open(cylf, "rb")); ac = pickle.load(open(acf, "rb"))
    cik = []
    t0 = time.time(); atlanan = 0
    for i, (pid, r) in enumerate(sorted(kayitlar.items())):
        G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        yol = f"{OZ}/{ad}_{pid}.npz"
        if os.path.exists(yol):
            try:
                z = np.load(yol)
                cik.append({"pid": pid, "mfg": r["mfg"], "X": z["X"], "y": z["y"],
                            "P": z["P"], "D": z["D"], "kaynak": z["kaynak"],
                            "G": G, "Gd": np.asarray(r["Gd"], float),
                            "diag": r["diag"]})
                continue
            except Exception:
                os.remove(yol)                       # bozuk onbellek -> yeniden uret
        f = f"{ob}/{pid}.npz"
        if not os.path.exists(f):
            atlanan += 1
            continue
        z = np.load(f)
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pb = np.asarray(z["pbs"], float).mean(0)
        P, D, kay = brep_havuz.birlesik_havuz(r["P"], r["Pd"], cy.get(pid),
                                              ac.get(pid))
        if not len(P):
            continue
        cps = [{"point": P[j], "direction": D[j]} for j in range(len(P))]
        X = wire_gate.feats_for(V, F, pb, cps, CE, CT, step_path=S.get(pid))
        X = np.asarray(X, float)
        # ETIKET: aday GT'ye TESPIT toleransinda mi (bire-bir Macar ile atanmis mi)
        tol = max(3.0, 0.06 * r["diag"])
        d = np.linalg.norm(P[:, None] - G[None], axis=-1)
        y = (d.min(1) <= tol).astype(np.int8)
        np.savez_compressed(yol, X=X, y=y, P=P, D=D, kaynak=kay)
        cik.append({"pid": pid, "mfg": r["mfg"], "X": X, "y": y, "P": P, "D": D,
                    "kaynak": kay, "G": G, "Gd": np.asarray(r["Gd"], float),
                    "diag": r["diag"]})
        if (i + 1) % 25 == 0:
            print(f"  {ad} {i+1}/{len(kayitlar)} ({time.time()-t0:.0f}s, "
                  f"atlanan {atlanan})", flush=True)
    print(f"{ad}: {len(cik)} parca hazir (atlanan {atlanan})", flush=True)
    return cik


print("D6 ozellikleri...", flush=True)
tr = kume("d6", d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])),
          "results/_p1_olasilik_g7", "results/_d6_silindirler.pkl",
          "results/_d6_acikliklar.pkl")
print("D7 ozellikleri...", flush=True)
te = kume("d7", K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"]),
          "results/_p1_olasilik_d7", "results/_d7_silindirler.pkl",
          "results/_d7_acikliklar.pkl")

Xtr = np.vstack([d["X"] for d in tr]); ytr = np.concatenate([d["y"] for d in tr])
print(f"\nEGITIM {Xtr.shape} | pozitif {ytr.mean():.4f}", flush=True)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                             class_weight="balanced_subsample", random_state=0)
clf.fit(Xtr, ytr)
pickle.dump({"clf": clf, "n_feat": Xtr.shape[1]},
            open("results/brep_gate_d6.pkl", "wb"))
print("gate egitildi -> results/brep_gate_d6.pkl", flush=True)

import collections
sonuc = {}
for esik in (0.30, 0.40, 0.50, 0.60, 0.70):
    rob = collections.defaultdict(lambda: [0, 0, 0]); tes = []
    for d in te:
        s = clf.predict_proba(d["X"])[:, 1]
        m = s >= esik
        P, D = (d["P"][m], d["D"][m]) if m.any() else (d["P"][:0], d["D"][:0])
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, s[m]); P, D = P[nm], D[nm]
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                                False, isaretli=True)[:3]
        a = rob[d["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                   max(3.0, 0.06*d["diag"]), 180.0, True)[:3])
    pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in rob.items()}
    mi = float(2*sum(a[0] for a in rob.values()) /
               max(sum(2*a[0]+a[1]+a[2] for a in rob.values()), 1))
    sonuc[esik] = {"robot": mi, "tespit": K.mikro(tes),
                   "makro": float(np.mean(list(pm.values()))),
                   "en_kotu": float(min(pm.values())), "marka": pm}
    c = sonuc[esik]
    print(f"esik {esik:.2f}  robot {mi:.4f} | tespit {c['tespit']:.4f} | "
          f"makro {c['makro']:.4f} | en kotu {c['en_kotu']:.4f}", flush=True)
json.dump({"damga": makbuz_hash.damga(), "sonuc": {str(k): v for k, v in sonuc.items()},
           "taban_kanonik": {"robot": 0.2029, "tespit": 0.4523},
           "not": "GENISLETILMIS HAVUZ (seg + B-rep) + D6'da REFIT gate. D7 marka-disi. "
                  "MIKRO. Tez turetmesi DEGISMEDI; B-rep EK aday kaynagi."},
          open("results/brep_gate_d7.json", "w"), indent=1)
print("makbuz -> results/brep_gate_d7.json")
