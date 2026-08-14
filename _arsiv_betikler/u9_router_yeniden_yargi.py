# -*- coding: utf-8 -*-
"""U9: COKUS YONLENDIRICISINI YENIDEN YARGILA -- temiz protokol, tek kural.

2026-08-01 DENETIM BULGUSU (hakli): yonlendirici KENDI on-yazili testini gecmemisti --
`results/u7_yonlendirme.json` kazanan=null, `results/u8_yonlendirme_dogrula.json` hakimiyet=false.
Buna ragmen "D'yi her eksende domine ediyor" gerekcesiyle dagitildi. Yani cubuk sonradan
degistirildi. Bu betik karari SIFIRDAN, duzeltilmis protokolle verir:

  * KUME: `olcum_kumesi` (split3.json'dan DEV/VAL, PID dedup, dogrudan LOCKED CIKARILMIS)
  * BOOTSTRAP: GEOMETRI GRUBU birimli (parca degil -- ikizler bagimsiz sayilmaz)
  * KARAR: `karar_olcutu.degerlendir` (GA artik KARARA KATILIYOR; kanitsiz kol GECMEZ)

UC KOL (dagitim merdiveninin tamami):
  A  ham 22 sutun                    -> results/wire_gate.pkl.pre_parca_ici
  D  her parcaya parca-ici z-skor    -> results/wire_gate.pkl.pre_yonlendirme
  R  cokus yonlendirmeli (DAGITIK)   -> results/wire_gate.pkl

Kural gecmezse SONUC: bir alt basamaga DON. Bu betigin cikti verdigi sey bir tavsiye degil,
uygulanacak karardir.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _egitim_verisi():
    """Dagitilan gate hangi veriyle egitildiyse olcum de ONUNLA yapilmali."""
    try:
        with io.open("cp_config.json", encoding="utf-8") as f:
            return json.load(f)["current_product"]["wire_gate"].get(
                "egitim_verisi", "results/gate_regrow_data_topo.npz")
    except Exception:
        return "results/gate_regrow_data_topo.npz"


def main():
    import karar_olcutu
    import olcum_kumesi
    import wire_gate
    from big_arbiter import eligible
    from sina_kume import esle, f1w
    from sklearn.ensemble import RandomForestClassifier

    cfg = json.load(io.open("cp_config.json", encoding="utf-8"))
    ORAN = float(cfg["gate_goreli_oran"]); TABAN = float(cfg["gate_goreli_taban"])
    mfg_of = {p: m for m, p, jf, s in eligible()}
    DER, rap = olcum_kumesi.kume()
    olcum_kumesi.rapor_bas(rap)
    for r in DER:
        r["mfg"] = mfg_of.get(r["pid"], "?")

    d = np.load(_egitim_verisi(), allow_pickle=True)
    gk = olcum_kumesi.geo_anahtarlari()
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    tr_mfg = np.array([str(x) for x in d["mfg"]])
    Xtr = np.asarray(d["X"], float); ytr = np.asarray(d["y"])
    tg = {r["geo"] for r in DER}
    hepsi = ~np.isin(tr_grp, list(tg))
    kod = {k: collections.Counter(mfg_of.get(p, "?") for p in tr_pid[tr_mfg == k]).most_common(1)[0][0]
           for k in np.unique(tr_mfg)}
    Ztr = np.zeros((len(Xtr), Xtr.shape[1] * 2))
    for u in np.unique(tr_pid):
        i = np.where(tr_pid == u)[0]
        Ztr[i] = wire_gate.parca_ici(Xtr[i], "zskor")

    def kur(keep, kol):
        rf = lambda M: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                              random_state=0).fit(M[keep], ytr[keep])
        if kol == "A":
            return {"clf": rf(Xtr), "n_feat": Xtr.shape[1], "donusum": None}
        if kol == "D":
            return {"clf": rf(Ztr), "n_feat": Ztr.shape[1], "donusum": "zskor"}
        m = {"clf": rf(Xtr), "clf_z": rf(Ztr), "n_feat": Xtr.shape[1],
             "donusum": None, "donusum_z": "zskor"}
        mx = [float(m["clf"].predict_proba(Xtr[np.where((tr_pid == u) & keep)[0]])[:, 1].max())
              for u in np.unique(tr_pid[keep]) if ((tr_pid == u) & keep).any()]
        m["esik_cokus"] = float(np.quantile(mx, 0.10))
        return m

    def puanla(m, alt):
        det, rob = [], []
        for r in alt:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if r["X"] is not None:
                s = wire_gate.karar_skoru(m, r["X"])
                k = wire_gate.karar_maskesi(s)
                if k.any():
                    P = r["P"][k]; Pd = r["Pd"][k]
            rj = "cok" if r["n"] >= 8 else "dusuk"
            det.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        return det, rob

    BOLME = [("tanidik", hepsi, DER)]
    for k, mad in kod.items():
        alt = [x for x in DER if x["mfg"] == mad]
        if len(alt) >= 10:
            BOLME.append((mad, (tr_mfg != k) & hepsi, alt))

    SON, PARCA = {}, {}
    print(f"\n{'kol':<6}" + "".join(f"{b:>12}" for b, _, _ in BOLME) + f"{'robot(tanidik)':>16}")
    for kol in ("A", "D", "R"):
        SON[kol] = {}
        sat = f"{kol:<6}"
        for b, keep, alt in BOLME:
            det, rob = puanla(kur(keep, kol), alt)
            SON[kol][b] = float(f1w(det))
            PARCA[(kol, b)] = (det, rob, [x["geo"] for x in alt])
            sat += f"{f1w(det):>12.4f}"
        sat += f"{f1w(PARCA[(kol,'tanidik')][1]):>16.4f}"
        print(sat, flush=True)

    def ga(kol_a, kol_b):
        """GRUP bootstrap ile eslestirilmis fark GA'si."""
        out = {}
        for b, _, _ in BOLME:
            if b == "tanidik":
                continue
            da, _, g = PARCA[(kol_a, b)]
            db, _, _ = PARCA[(kol_b, b)]
            cift = list(zip(da, db))
            fn = lambda rows: f1w([y for _, y in rows]) - f1w([x for x, _ in rows])
            _, lo, hi = olcum_kumesi.grup_bootstrap(cift, g, fn, n=2000)
            out[b] = (lo, hi)
        return out

    print("\n=== KARAR (karar_olcutu, GA KARARA KATILIYOR) ===")
    kararlar = {}
    for taban_kol, aday_kol in (("A", "D"), ("A", "R"), ("D", "R")):
        k = karar_olcutu.degerlendir(SON[taban_kol], SON[aday_kol],
                                     ga=ga(taban_kol, aday_kol))
        kararlar[f"{taban_kol}->{aday_kol}"] = bool(k)
        print(f"\n{taban_kol} -> {aday_kol}:  {k}")

    print("\n=== UYGULANACAK KARAR ===")
    # ILKE: GECEN EN BASIT KOL secilir. Daha karmasik bir kol ancak daha basitini KANITLI
    # sekilde gecerse tercih edilir. (Ilk surumde "A->R gectiyse R kalir" yaziyordu -- bu,
    # R'nin D'yi gecip gecmedigini HIC sormuyordu ve yonlendiricinin dagitilmasinin
    # gerekcesindeki kusurun ta kendisiydi.)
    if kararlar["D->R"]:
        sonuc = "R KALIR (D'yi kanitli sekilde geciyor)"
    elif kararlar["A->D"]:
        sonuc = "D'ye DON -> results/wire_gate.pkl.pre_yonlendirme (R, D'yi gecemiyor)"
    elif kararlar["A->R"]:
        sonuc = "R KALIR (D gecmedi ama R gecti)"
    else:
        sonuc = "A'ya DON -> results/wire_gate.pkl.pre_parca_ici"
    print(f"  {sonuc}")
    with io.open("results/u9_router_yargi.json", "w", encoding="utf-8") as f:
        json.dump({"kollar": SON, "kararlar": kararlar, "sonuc": sonuc,
                   "kume": {k: rap[k] for k in ("puanlanan_parca", "geometri_grubu",
                                                "kume_dagilimi")}}, f, indent=1)
    print("makbuz -> results/u9_router_yargi.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
