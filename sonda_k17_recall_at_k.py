"""K1.7 sondasi: p3c secenek havuzunda dogru eksen VAR MI, KACINCI SIRADA?

Recall@k dusukse  -> TEMSIL sorunu (havuzda yok), yeni onerici gerekir.
Recall@k yuksek ama rank kotu -> SIRALAMA sorunu, ozellik/skor duzeltilir.

Ayrica dogru secenegin YEREL yaricap sirasini olcer: hipotez, dogru secenegin
komsulugundaki en buyuk silindir olmasi.
"""
import os
import sys
import glob
import pickle
import collections

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")

import d6_kayit                      # noqa: E402
import wire_gate                     # noqa: E402
import urun_zinciri                  # noqa: E402
import p3c_eksen_secici as P3C       # noqa: E402
import brep_snap                     # noqa: E402
from p1c_esik import maske           # noqa: E402
from sina_kume import esle_macar     # noqa: E402
from korpus_kimlik import step_kimlik as SK  # noqa: E402

ACI = 10.0          # dogru sayilma esigi (urun robot toleransi)
KOMSU_MM = 10.0

sv = d6_kayit.sinav()
kayit = d6_kayit.yukle(set(sv["pidler"]))
gate = pickle.load(open("results/wire_gate_v5.pkl", "rb"))
cyl = pickle.load(open("results/_d6_silindirler.pkl", "rb"))
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
OB = "results/_p1_olasilik"

n_es = 0
havuzda = 0
siralar = []
yaricap_sirasi = []          # dogru secenegin yerel yaricap sirasi (0 = en buyuk)
yerel_n = []

for pid, r in kayit.items():
    f = f"{OB}/{pid}.npz"
    if not os.path.exists(f):
        continue
    M = d6_kayit.x58(r)
    if M is None or r.get("P") is None or not len(r["P"]):
        continue
    sk = np.asarray(wire_gate.karar_skoru(gate, M), float)
    k = maske(sk, 0.40, 0.30)
    if not k.any():
        continue
    P = np.asarray(r["P"], float)[k]
    D = np.asarray(r["Pd"], float)[k]
    d = np.load(f)
    V = np.ascontiguousarray(d["V"], np.float64)
    F = np.ascontiguousarray(d["F"], np.int64)
    P, D = urun_zinciri.tam_poz(
        V, F, np.asarray(d["pbs"], float).mean(0), P, D, step_path=S.get(pid)
    )
    G = np.asarray(r["G"], float)
    Gd = np.asarray(r["Gd"], float)
    if not len(P) or not len(G):
        continue
    C = cyl.get(pid) or []
    if not C:
        continue

    A = np.asarray([c["axis"] for c in C], float)
    R = np.asarray([c["radius"] for c in C], float)
    Mo = np.asarray(
        [0.5 * (np.asarray(c["mouth_a"], float) + np.asarray(c["mouth_b"], float))
         for c in C], float
    )

    _t, _f, _n, bi = esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)
    for (pi, gi, _y, _ax, _ac, _b) in bi["eslesme"]:
        n_es += 1
        yak = np.where(np.linalg.norm(Mo - P[pi], axis=1) <= KOMSU_MM)[0]
        if not len(yak):
            continue
        # secenek havuzu: her yakin silindirin +/- ekseni
        ops = []
        for j in yak:
            for s in (1.0, -1.0):
                ops.append((j, s * A[j]))
        # dogru olan(lar)
        dogru = [
            t for t, (j, a) in enumerate(ops)
            if np.degrees(np.arccos(np.clip(float(a @ Gd[gi]), -1, 1))) <= ACI
        ]
        if not dogru:
            continue
        haviuzda = True
        havuzda += 1
        # p3c'nin fiilen sectigi yon D[pi]; onun havuzdaki sirasini,
        # "buyuk yaricap once" siralamasina gore olc
        skor = np.asarray([R[j] for (j, a) in ops], float)
        sira_idx = np.argsort(-skor)
        yer = {t: q for q, t in enumerate(sira_idx)}
        siralar.append(min(yer[t] for t in dogru))
        # dogru secenegin yerel yaricap sirasi
        rr = np.asarray([R[j] for j in yak], float)
        j_dogru = ops[dogru[0]][0]
        yaricap_sirasi.append(int((rr > R[j_dogru]).sum()))
        yerel_n.append(len(yak))

siralar = np.asarray(siralar)
yr = np.asarray(yaricap_sirasi)
print(f"eslesme {n_es} | havuzda dogru eksen VAR: {havuzda} (%{100*havuzda/max(n_es,1):.1f})\n")
print("YARICAP-SIRALI havuzda dogru eksenin recall@k:")
for kk in (1, 2, 3, 5, 10):
    print(f"  recall@{kk:<3} = %{100*(siralar < kk).mean():.1f}")
print(f"\ndogru secenegin YEREL yaricap sirasi (0 = komsulugun EN BUYUGU):")
for q in (0, 1, 2):
    print(f"  sira {q}: %{100*(yr == q).mean():.1f}")
print(f"  sira >2: %{100*(yr > 2).mean():.1f}")
print(f"komsulukta ortalama silindir: {np.mean(yerel_n):.1f}")
