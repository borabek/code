# -*- coding: utf-8 -*-
"""P3-c: UYE-ONERI KAHINI -- ek seed egitmeye deger mi?

SORU: g10 s1/s2 egitmek (saatler) kazanc getirir mi? Kapi: uye havuzunu
BIRLESTIRMEK aday kahinini +0.05 artiriyorsa EGIT, <+0.02 ise KAPAT.

VEKIL: elimizde g7 ve g10 var (farkli egitimler). Bunlarin BIRLESIMI, iki
seed'in birlesimi icin bir ALT SINIRDIR (g7 ve g10 birbirinden g10-s0/s1'den
DAHA farkli, yani gercek seed kazanci bundan KUCUK olur).
"""
import glob, json, os, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import d6_kayit, robot_cp
from sina_kume import esle_macar, f1w
from korpus_kimlik import step_kimlik as SK

A, B = "results/_p1_olasilik_g10", "results/_p1_olasilik_g7"
sv = d6_kayit.sinav(); kayit = d6_kayit.yukle(set(sv["pidler"]))
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
ortak = sorted({f[:-4] for f in os.listdir(A) if f.endswith(".npz")}
               & {f[:-4] for f in os.listdir(B) if f.endswith(".npz")} & set(kayit))
print(f"ortak parca {len(ortak)}", flush=True)


def adaylar(ob, pid):
    d = np.load(f"{ob}/{pid}.npz")
    V = np.ascontiguousarray(d["V"], np.float64); F = np.ascontiguousarray(d["F"], np.int64)
    cps, _o, _c, _p = robot_cp.adaylari_uret(
        V, F, [np.asarray(q, float) for q in d["pbs"]], S.get(pid))
    if not cps:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.asarray([c["point"] for c in cps], float),
            np.asarray([c["direction"] for c in cps], float))


T_a, T_b, T_u = [], [], []
for pid in ortak:
    r = kayit[pid]
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    if not len(G):
        continue
    Pa, Da = adaylar(A, pid); Pb, Db = adaylar(B, pid)
    Pu = np.vstack([Pa, Pb]) if len(Pa) or len(Pb) else np.zeros((0, 3))
    Du = np.vstack([Da, Db]) if len(Da) or len(Db) else np.zeros((0, 3))
    for L, (P, D) in ((T_a, (Pa, Da)), (T_b, (Pb, Db)), (T_u, (Pu, Du))):
        L.append((len(G),) + esle_macar(P, D, G, Gd, r["diag"], 0., 180., True)[:3])
a, b, u = f1w(T_a), f1w(T_b), f1w(T_u)
print(f"g10 tek       : {a:.4f}")
print(f"g7  tek       : {b:.4f}")
print(f"BIRLESIM      : {u:.4f}")
print(f"BIRLESIM KAZANCI (en iyi tekten): {u - max(a, b):+.4f}")
k = u - max(a, b)
print(f"KARAR: {'EGIT (>=+0.05)' if k >= 0.05 else ('KAPAT (<+0.02)' if k < 0.02 else 'BELIRSIZ (0.02-0.05)')}")
json.dump({"damga": makbuz_hash.damga(), "g10": a, "g7": b, "birlesim": u,
           "kazanc": k, "n_parca": len(T_a),
           "not": "g7+g10 birlesimi, IKI SEED birlesimi icin ALT SINIR"},
          open("results/p3c_uye_kahin.json", "w"), indent=1)
print("makbuz -> results/p3c_uye_kahin.json")
