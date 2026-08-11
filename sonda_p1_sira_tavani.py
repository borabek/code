# -*- coding: utf-8 -*-
"""P1 ONCULU: gate ONCE mi SONRA mi? Tavan farki gercek mi?

IDDIA: ham aday + (silindir + planar) poz secenekleriyle bire-bir kahin 0.6097;
ONCE gate uygulanirsa ayni kahin 0.4837. Yani mevcut `gate -> pose` sirasi
tavani BASTAN kirpiyor. Bu sonda o iddiayi KENDI verimizde uretir.

KAHIN = her aday icin seceneklerin EN IYISI secilebilseydi (bire-bir Macar,
robot toleransi 2mm/10 derece ISARETLI). Ulasilabilir degil, TAVAN.

TEZE SADIK: `v_o` (mevcut poz) HER ZAMAN seceneklerden biri.
"""
import glob, json, os, pickle, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, robot_cp, wire_gate, brep_snap
from p1c_esik import maske
from sina_kume import esle_macar, f1w
from korpus_kimlik import step_kimlik as SK

OB = "results/_p1_olasilik_g10"; GATE = "results/wire_gate_v7.pkl"
YANAL, ACI = 2.0, 10.0
sv = d6_kayit.sinav(); kayit = d6_kayit.yukle(set(sv["pidler"]))
gate = pickle.load(open(GATE, "rb"))
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
cyl = pickle.load(open("results/_d6_silindirler.pkl", "rb"))
pidler = sorted({f[:-4] for f in os.listdir(OB) if f.endswith(".npz")} & set(kayit))
print(f"parca {len(pidler)}\n", flush=True)


def secenekler(P, D, cyls, mm=8.0):
    """Her aday icin (konum, yon) secenekleri. ILK oge HER ZAMAN MEVCUT (tez)."""
    uy = [c for c in (cyls or []) if brep_snap.R_MIN <= c["radius"] <= brep_snap.R_MAX]
    out = []
    for i in range(len(P)):
        o = [(P[i], D[i])]
        for c in uy:
            a = np.asarray(c["axis"], float)
            for m in (c["mouth_a"], c["mouth_b"]):
                m = np.asarray(m, float)
                if np.linalg.norm(m - P[i]) <= mm:
                    o.append((m, a)); o.append((m, -a))
        out.append(o)
    return out


def kahin(P, D, cyls, G, Gd, diag):
    """ORTAK (bipartite) kahin -- aday BASINA acgozlu DEGIL.

    ILK SURUMUM YANLISTI: her aday KENDI en iyi secenegini bagimsiz seciyordu.
    Boylece bircok aday AYNI GT'ye yigiliyor, Macar bire-bir atayinca cogu bosa
    gidiyordu ve kahin uctan uca sonucun ancak biraz ustunde cikiyordu (0.151) --
    hatta gate'li hali (0.199) DAHA YUKSEK gorunuyordu, ki bir ust kumede
    imkansizdir. Isaret: kahin ORTAK olmali.

    Dogrusu: (aday i, GT j) icin i'nin O GT'ye gore EN IYI secenegini bul,
    kabul edilebilirse 1 puan; sonra bipartite EN BUYUK ESLESME (Hungarian).
    """
    if not len(P) or not len(G):
        return (len(G), 0.0, 0.0, 0.0)
    S_ = secenekler(P, D, cyls)
    n, m = len(P), len(G)
    C = np.ones((n, m))          # maliyet: 0 = kabul edilebilir, 1 = degil
    for i, o in enumerate(S_):
        for j in range(m):
            g = Gd[j] / (np.linalg.norm(Gd[j]) + 1e-9)
            for (p, d) in o:
                u = d / (np.linalg.norm(d) + 1e-9)
                v = G[j] - p
                lat = np.linalg.norm(v - (v @ u) * u)
                ax = abs(float(v @ u))
                ac = np.degrees(np.arccos(np.clip(float(u @ g), -1, 1)))
                if lat <= YANAL and ac <= ACI and ax <= 40.0:
                    C[i, j] = 0.0
                    break
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(C)
    tp = int(sum(1 for a, b_ in zip(ri, ci) if C[a, b_] == 0.0))
    # NULL SECENEGI: p5-v2 tasariminda bir aday ATILABILIR. Kahin de atabilmeli,
    # yoksa ham havuzun fazla adaylari otomatik FP olur ve GENIS havuz DAHA KOTU
    # gorunur -- ki bu, olculmek istenen seyin (havuz zenginligi) TERSINI olcer.
    # Ikinci surumumde bu yoktu ve gate'li kol daha yuksek cikiyordu.
    fp = 0
    fn = m - tp
    return (m, float(tp), float(fp), float(fn))


R_ham, R_gate = [], []
for pid in pidler:
    r = kayit[pid]
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    if not len(G): continue
    d = np.load(f"{OB}/{pid}.npz")
    V = np.ascontiguousarray(d["V"], np.float64); F = np.ascontiguousarray(d["F"], np.int64)
    cps, _o, _c, _p = robot_cp.adaylari_uret(V, F, [np.asarray(q, float) for q in d["pbs"]],
                                             S.get(pid))
    if not cps:
        R_ham.append((len(G),0.,0.,0.)); R_gate.append((len(G),0.,0.,0.)); continue
    P = np.asarray([c["point"] for c in cps], float)
    D = np.asarray([c["direction"] for c in cps], float)
    cy = cyl.get(pid)
    R_ham.append(kahin(P, D, cy, G, Gd, r["diag"]))
    avg = np.asarray(d["pbs"], float).mean(0)
    Xp = np.asarray(wire_gate.feats_for(V, F, avg, cps, robot_cp.CE, robot_cp.CT,
                                        step_path=S.get(pid)), float)
    k = maske(np.asarray(wire_gate.karar_skoru(gate, Xp), float), 0.40, 0.30)
    if k.any():
        R_gate.append(kahin(P[k], D[k], cy, G, Gd, r["diag"]))
    else:
        R_gate.append((len(G), 0., 0., 0.))
a, b = f1w(R_ham), f1w(R_gate)
print(f"HAM aday + poz secenekleri (gate YOK) : {a:.4f}")
print(f"ONCE gate, sonra poz secenekleri      : {b:.4f}")
print(f"GATE'IN KIRPTIGI TAVAN                : {a-b:+.4f}")
json.dump({"ham_kahin": a, "gate_once_kahin": b, "fark": a-b, "n_parca": len(pidler),
           "not": "D6 (DEV). Kahin = ulasilabilir degil TAVAN."},
          open("results/p1_sira_tavani.json","w"), indent=1)
print("makbuz -> results/p1_sira_tavani.json")
