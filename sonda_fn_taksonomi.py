# -*- coding: utf-8 -*-
"""FN TAKSONOMISI -- TAM zincir + YENI yigin (g10 + gate v7).

Hata bankasi g5 doneminden kalma (GATE_REDDI %43 / ADAY_YOK %41 / KALABALIK %15).
Yigin degisti; kovalarin YENIDEN olculmesi lazim ki kalan kollar dogru yere baksin.

KOVALAR (oncelik sirasiyla):
  ADAY_YOK    : GT'nin tespit toleransinda HIC aday yok (temsil)
  GATE_REDDI  : aday VAR ama gate elemis (karar)
  KALABALIK   : aday var, gate gecmis, ama Macar baska GT'ye vermis (rekabet)
  POZ         : eslesmis ama robot toleransini gecemiyor (yanal/aci)
"""
import collections, glob, json, os, pickle, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, robot_cp, wire_gate, urun_zinciri
from p1c_esik import maske
from sina_kume import esle_macar
from korpus_kimlik import step_kimlik as SK

OB = "results/_p1_olasilik_g10"; GATE = "results/wire_gate_v7.pkl"
sv = d6_kayit.sinav(); kayit = d6_kayit.yukle(set(sv["pidler"]))
gate = pickle.load(open(GATE, "rb"))
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
pidler = sorted({f[:-4] for f in os.listdir(OB) if f.endswith(".npz")} & set(kayit))

kova = collections.Counter(); n_gt = 0
for pid in pidler:
    r = kayit[pid]
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    if not len(G): continue
    d = np.load(f"{OB}/{pid}.npz")
    V = np.ascontiguousarray(d["V"], np.float64); F = np.ascontiguousarray(d["F"], np.int64)
    cps, _o, _c, _p = robot_cp.adaylari_uret(V, F, [np.asarray(q, float) for q in d["pbs"]],
                                             S.get(pid))
    n_gt += len(G)
    if not cps:
        kova["ADAY_YOK"] += len(G); continue
    P0 = np.asarray([c["point"] for c in cps], float)
    D0 = np.asarray([c["direction"] for c in cps], float)
    avg = np.asarray(d["pbs"], float).mean(0)
    Xp = np.asarray(wire_gate.feats_for(V, F, avg, cps, robot_cp.CE, robot_cp.CT,
                                        step_path=S.get(pid)), float)
    k = maske(np.asarray(wire_gate.karar_skoru(gate, Xp), float), 0.40, 0.30)
    P1, D1 = (P0[k], D0[k]) if k.any() else (P0[:0], D0[:0])
    if len(P1):
        P1, D1 = urun_zinciri.tam_poz(V, F, avg, P1, D1, step_path=S.get(pid))
    tol = max(3.0, 0.06 * r["diag"])
    es_rob = set()
    if len(P1):
        _t,_f,_n,bi = esle_macar(P1, D1, G, Gd, r["diag"], 2.0, 10.0, False, isaretli=True)
        es_rob = {gi for (_pi, gi, *_x) in bi["eslesme"]}
        _t2,_f2,_n2,bi2 = esle_macar(P1, D1, G, Gd, r["diag"], 0.0, 180.0, True)
        es_tes = {gi for (_pi, gi, *_x) in bi2["eslesme"]}
    else:
        es_tes = set()
    for i in range(len(G)):
        if i in es_rob: continue                      # robot-TP, FN degil
        yakin_ham = np.min(np.linalg.norm(P0 - G[i], axis=1)) <= tol if len(P0) else False
        yakin_gate = np.min(np.linalg.norm(P1 - G[i], axis=1)) <= tol if len(P1) else False
        if not yakin_ham:      kova["ADAY_YOK"] += 1
        elif not yakin_gate:   kova["GATE_REDDI"] += 1
        elif i not in es_tes:  kova["KALABALIK"] += 1
        else:                  kova["POZ"] += 1       # eslesti ama robot toleransi yok
print(f"parca {len(pidler)} | GT {n_gt} | robot-FN {sum(kova.values())}\n")
for k_, v in kova.most_common():
    print(f"  {k_:<12} {v:>5}  %{100*v/max(sum(kova.values()),1):.1f}")
json.dump(dict(kova), open("results/fn_taksonomi_g10v7.json","w"), indent=1)
print("\nmakbuz -> results/fn_taksonomi_g10v7.json")
