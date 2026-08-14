# -*- coding: utf-8 -*-
"""U7: parca-ici z-skoru YALNIZ COKUS halinde uygula (takasi yonlendirme ile kaldir).

U6 DESENI (7 seri-disi bolme + 2 uretici-disi bolme, hepsi uctan uca):
    D, TABAN ZAYIFKEN kazaniyor, TABAN SAGLIKLIYKEN kaybediyor.
      WEI disarida  taban 0.4832 (COKUS) -> D +0.0869
      seri 25       taban 0.5654         -> D +0.0639
      seri 10       taban 0.5980         -> D +0.0086
      seri 17/15/30/32/16 taban 0.66-0.78 -> D -0.008 .. -0.060
      PXC disarida  taban 0.7203         -> D -0.0375
Yani D genel bir iyilestirme DEGIL, bir KURTARMA. Her parcaya uygulamak, cokmeyen parcalarda
vergi odemek demek.

FIKIR: cokusu CALISMA ANINDA sez, yalniz orada z-skor gate'e gec. Sezgi metadata GEREKTIRMEZ --
ham gate'in KENDI skor dagilimi zaten cokusu gosteriyordu: olculmus teshis, cokus halinde model
adaylarin %10.3'une pozitif diyor, gercek %24.1. Yani parcanin en yuksek ham skoru DUSUKSE,
gate o parcada kararsizdir.

YONLENDIRICI: parcanin ham-gate maksimum skoru, EGITIM korpusunun q'inci yuzdeligininin ALTINDA
ise -> z-skor gate. Esik EGITIM parcalarindan (test geometrisi haric) hesaplanir; test tarafina
BAKMAZ.

KILL (onceden yazili, dorduncu sart dahil):
  (a) WEI-disi (cokus) kazancinin EN AZ %80'ini korumali          (>= A + 0.0695)
  (b) PXC-disi kaybi en fazla 0.01 olmali                          (>= A - 0.01)
  (c) tanidik >= A - 0.01
  (d) 7 seri-disi bolmede ORTALAMA fark >= -0.002                  (vergiyi KALDIRMALI)
Dordu birden saglanmazsa yonlendirme DAGITILMAZ ve D ile A arasindaki secim ACIKCA raporlanir.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
QLER = [0.10, 0.25, 0.40]
ONEK = 2


def main():
    import wire_gate
    from big_arbiter import eligible
    from sina_kume import esle, f1w
    from sklearn.ensemble import RandomForestClassifier

    with open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    ORAN = float(cfg["gate_goreli_oran"]); TABAN = float(cfg["gate_goreli_taban"])
    mfg_of = {p: m for m, p, jf, s in eligible()}

    with open("results/_u4_der.pkl", "rb") as f:
        DER = pickle.load(f)
    for r in DER:
        r["mfg"] = mfg_of.get(r["pid"], "?")
    d = np.load("results/gate_regrow_data_topo.npz", allow_pickle=True)
    with open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    tr_mfg = np.array([str(x) for x in d["mfg"]])
    tr_onek = np.array([p[:ONEK] for p in tr_pid])
    Xtr = np.asarray(d["X"], float); ytr = np.asarray(d["y"])
    tg = {gk.get(r["pid"], "yok:" + r["pid"]) for r in DER}
    kod = {k: collections.Counter(mfg_of.get(p, "?") for p in tr_pid[tr_mfg == k]).most_common(1)[0][0]
           for k in np.unique(tr_mfg)}

    Ztr = np.zeros((len(Xtr), Xtr.shape[1] * 2))
    for u in np.unique(tr_pid):
        i = np.where(tr_pid == u)[0]
        Ztr[i] = wire_gate.parca_ici(Xtr[i], "zskor")

    def rf(M, keep):
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                      random_state=0).fit(M[keep], ytr[keep])

    def maske(s):
        return (s >= ORAN * max(float(s.max()), 1e-9)) & (s >= TABAN)

    def olc(alt, keep):
        """A / D / yonlendirilmis(q) kollarini AYNI egitim bolmesiyle olc."""
        ca = rf(Xtr, keep); cd = rf(Ztr, keep)
        # ESIK: yalniz EGITIM parcalarinin ham-gate maks skorlarindan. Test tarafina BAKMAZ.
        egit_maks = []
        for u in np.unique(tr_pid[keep]):
            i = np.where((tr_pid == u) & keep)[0]
            if len(i):
                egit_maks.append(float(ca.predict_proba(Xtr[i])[:, 1].max()))
        esik = {q: float(np.quantile(egit_maks, q)) for q in QLER}
        det = {k: [] for k in ["A", "D"] + [f"R{q}" for q in QLER]}
        yon_say = {q: 0 for q in QLER}
        for r in alt:
            sa = sd = None
            if r["X"] is not None:
                sa = ca.predict_proba(r["X"])[:, 1]
                sd = cd.predict_proba(wire_gate.parca_ici(r["X"], "zskor"))[:, 1]
            for ad in det:
                if sa is None:
                    s = None
                elif ad == "A":
                    s = sa
                elif ad == "D":
                    s = sd
                else:
                    q = float(ad[1:])
                    cok = float(sa.max()) < esik[q]        # COKUS SEZGISI
                    yon_say[q] += int(cok)
                    s = sd if cok else sa
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
                if s is not None and maske(s).any():
                    m = maske(s); P = r["P"][m]; Pd = r["Pd"][m]
                det[ad].append(("cok" if r["n"] >= 8 else "dusuk",)
                               + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
        return {k: f1w(v) for k, v in det.items()}, yon_say, esik

    print("=== URETICI-DISI (asil eksen) ===")
    print(f"{'bolme':<16}{'n':>4}{'A':>9}{'D':>9}" + "".join(f"{'R'+str(q):>9}" for q in QLER)
          + "  yonlendirilen")
    MF = {}
    for k, mad in kod.items():
        alt = [x for x in DER if x["mfg"] == mad]
        if len(alt) < 10:
            continue
        v, ys, es = olc(alt, (tr_mfg != k) & ~np.isin(tr_grp, list(tg)))
        MF[mad] = v
        print(f"{mad + ' disarida':<16}{len(alt):>4}{v['A']:>9.4f}{v['D']:>9.4f}"
              + "".join(f"{v['R'+str(q)]:>9.4f}" for q in QLER)
              + "  " + "/".join(f"{ys[q]}" for q in QLER), flush=True)
    v, _, _ = olc(DER, ~np.isin(tr_grp, list(tg)))
    MF["tanidik"] = v
    print(f"{'TANIDIK':<16}{len(DER):>4}{v['A']:>9.4f}{v['D']:>9.4f}"
          + "".join(f"{v['R'+str(q)]:>9.4f}" for q in QLER))

    print("\n=== SERI-DISI (7 bolme, vergi kontrolu) ===")
    te_say = collections.Counter(r["pid"][:ONEK] for r in DER)
    tr_say = collections.Counter(p[:ONEK] for p in np.unique(tr_pid))
    seriler = sorted(k for k in te_say if te_say[k] >= 10 and tr_say[k] >= 30)
    SR = {k: [] for k in ["A", "D"] + [f"R{q}" for q in QLER]}
    for s_ in seriler:
        alt = [r for r in DER if r["pid"][:ONEK] == s_]
        v, _, _ = olc(alt, (tr_onek != s_) & ~np.isin(tr_grp, list(tg)))
        for k in SR:
            SR[k].append(v[k])
        print(f"  seri {s_}: A {v['A']:.4f} | D {v['D']:+.4f}->{v['D']-v['A']:+.4f} | "
              + " ".join(f"R{q} {v['R'+str(q)]-v['A']:+.4f}" for q in QLER), flush=True)
    ort = {k: float(np.mean(np.array(SR[k]) - np.array(SR["A"]))) for k in SR}
    kaz = {k: int((np.array(SR[k]) > np.array(SR["A"])).sum()) for k in SR}

    print(f"\n{'kol':<8}{'WEI-disi':>10}{'PXC-disi':>10}{'tanidik':>10}"
          f"{'seri ORT':>10}{'seri kazanan':>14}")
    A = MF["WEI"]["A"], MF["PXC"]["A"], MF["tanidik"]["A"]
    for k in ["A", "D"] + [f"R{q}" for q in QLER]:
        print(f"{k:<8}{MF['WEI'][k]:>10.4f}{MF['PXC'][k]:>10.4f}{MF['tanidik'][k]:>10.4f}"
              f"{ort[k]:>+10.4f}{str(kaz[k]) + '/' + str(len(seriler)):>14}")

    print("\nKILL (onceden yazili):")
    hedef = A[0] + 0.80 * (MF["WEI"]["D"] - A[0])
    kazanan = None
    for q in QLER:
        k = f"R{q}"
        a_ = MF["WEI"][k] >= hedef
        b_ = MF["PXC"][k] >= A[1] - 0.01
        c_ = MF["tanidik"][k] >= A[2] - 0.01
        d_ = ort[k] >= -0.002
        print(f"  {k}: (a) cokus {MF['WEI'][k]:.4f}>={hedef:.4f} {'OK' if a_ else 'X'} | "
              f"(b) PXC {MF['PXC'][k]:.4f}>={A[1]-0.01:.4f} {'OK' if b_ else 'X'} | "
              f"(c) tanidik {'OK' if c_ else 'X'} | (d) seri ort {ort[k]:+.4f}>=-0.002 "
              f"{'OK' if d_ else 'X'} -> {'GECTI' if all([a_,b_,c_,d_]) else 'GECMEDI'}")
        if all([a_, b_, c_, d_]) and (kazanan is None or MF["WEI"][k] > MF["WEI"][kazanan]):
            kazanan = k
    print(f"\nSONUC: {kazanan + ' DAGITILABILIR' if kazanan else 'YONLENDIRME GECMEDI'}")
    with open("results/u7_yonlendirme.json", "w", encoding="utf-8") as fh:
        json.dump({"uretici_disi": MF, "seri_ort_fark": ort, "seri_kazanan": kaz,
                   "n_seri": len(seriler), "kazanan": kazanan}, fh, indent=1)
    print("makbuz -> results/u7_yonlendirme.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
