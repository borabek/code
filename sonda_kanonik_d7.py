# -*- coding: utf-8 -*-
"""KANONIK ZINCIRLE D7: 0.2344 uretilebiliyor mu? Sonra p5-v2 AYNI zincirde.

Once DOGRULAMA: kanonik zincir + eski ag (D7 onbellegi) ~0.2344 vermeli.
Vermezse olcum yolu HALA urunun yaptigi sey DEGILDIR ve devam edilmez.
"""
import glob, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, urun_zinciri, kanonik_zincir as KZ
from sina_kume import esle_macar, f1w
from korpus_kimlik import step_kimlik as SK

YANAL, ACI = 2.0, 10.0
d7p = set(map(str, json.load(open("results/d7_sinav_kumesi.json"))["pidler"]))
kayit = d6_kayit.yukle(d7p)
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
OB = "results/_p1_olasilik_d7"
pidler = sorted({f[:-4] for f in os.listdir(OB) if f.endswith(".npz")} & set(kayit))
print(f"parca {len(pidler)}", flush=True)

T, R = [], []
for pid in pidler:
    r = kayit[pid]
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    if not len(G):
        continue
    d = np.load(f"{OB}/{pid}.npz")
    V = np.ascontiguousarray(d["V"], np.float64)
    F = np.ascontiguousarray(d["F"], np.int64)
    pbs = [np.asarray(q, float) for q in d["pbs"]]
    cps = KZ.urun_cikti(V, F, pbs, S.get(pid))
    P, D = KZ.poz_ver(cps)
    if len(P):
        P, D = urun_zinciri.tam_poz(V, F, np.asarray(d["pbs"], float).mean(0),
                                    P, D, step_path=S.get(pid))
    T.append((len(G),) + esle_macar(P, D, G, Gd, r["diag"], 0., 180., True)[:3])
    R.append((len(G),) + esle_macar(P, D, G, Gd, r["diag"], YANAL, ACI,
                                    False, isaretli=True)[:3])
t, rr = f1w(T), f1w(R)
print(f"\nKANONIK ZINCIR (eski ag): tespit {t:.4f} | robot {rr:.4f}")
print(f"MANSET BEKLENTISI       : tespit 0.4847 | robot 0.2344")
print(f"FARK                    : tespit {t-0.4847:+.4f} | robot {rr-0.2344:+.4f}")
json.dump({"damga": makbuz_hash.damga(), "tespit": t, "robot": rr,
           "manset_tespit": 0.4847, "manset_robot": 0.2344, "n_parca": len(T)},
          open("results/kanonik_d7_dogrulama.json", "w"), indent=1)
print("makbuz -> results/kanonik_d7_dogrulama.json")
