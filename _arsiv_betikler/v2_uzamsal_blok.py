# -*- coding: utf-8 -*-
"""V2: UZAMSAL BLOK -- temel gate'e parca-ici DUZEN bilgisi.

TESPIT 0.85 OPERASYONU, birinci kol.

TESHIS (v1): 0.85 yapisal olarak MUMKUN (aday tavani 0.9354) ama kahin gate'in %91'i gerekiyor;
su an %80.5'teyiz. Yani yeni BILGI lazim.

KULLANILMAMIS KAYNAK: `wire_gate.spatial_feats` ZATEN VAR ama yalnizca `apply_spatial`
(metadata-destekli top-N modu) icinde kullaniliyor -- TEMEL GATE'TE YOK. Oysa klemenste
CP'ler DUZENLI IZGARADA dizilir: ayni sirada esit araliklarla, cogu zaman ayna simetrisiyle.
Bu, tek bir acikligin geometrisinden GORULEMEYEN bir sinyaldir ve gate'in butun 58 sutunu
tek-aday bilgisidir.

ALTI SUTUN (op2_spatial ile ayni, cerceve-bagimsiz):
    sp_mir   ayna-esi var mi (parcanin ana ekseninde yansimasi baska bir adaya denk mi)
    sp_nnws  en yakin komsunun gate skoru
    sp_cons  ayni SATIR/SUTUNDAKI adaylarin skor toplami (dizilim tutarliligi)
    sp_dens  yerel yogunluk (skor agirlikli)
    sp_cen   parca merkezine goreli uzaklik
    sp_nnd   en yakin komsu mesafesi / parca yayilimi

CIFT GECISLI: sutunlar mevcut gate skorunu kullaniyor, yani once bir gecis yapilir, skorlar
uzamsal sutunlara beslenir, ikinci gate karar verir. Bu, tezdeki hicbir seyi degistirmez --
ag, remesh, sinif tanimi ve CP turetmesi aynen kalir.

KILL: karar_olcutu bes sarti (tanidik >= -0.01, bolme ortalamasi >= +0.01, en kotu bolme
kotulesmesin, hicbir bolmede > 0.05 kayip, GA ile KANITLI kazanc).
"""
import collections
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import karar_olcutu
    import olcum_kumesi
    import wire_gate
    from big_arbiter import eligible
    from sina_kume import esle, f1_rejim, f1w
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    mfg_of = {p: m for m, p, jf, s in eligible()}
    DER, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    for r in DER:
        r["mfg"] = mfg_of.get(r["pid"], "?")
    gk = olcum_kumesi.geo_anahtarlari(); tg = {r["geo"] for r in DER}

    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    X58 = np.hstack([np.asarray(zen["X22"], float), np.asarray(zen["XR"], float)])
    y = np.asarray(zen["y"]); pid = np.array([str(x) for x in zen["pids"]])
    mfg = np.array([str(x) for x in zen["mfg"]])
    pts = None
    par = np.load("results/gate_regrow_data_parite.npz", allow_pickle=True)
    ppid = np.array([str(x) for x in par["pids"]])
    PTS = np.asarray(par["pts"], float)
    ortak = {p for p in np.unique(pid) if (ppid == p).sum() == (pid == p).sum()}
    print(f"aday sayisi ESLESEN parca: {len(ortak)}/{len(np.unique(pid))}", flush=True)
    grp = np.array([gk.get(p, "yok:" + p) for p in pid])
    dag = wire_gate._load(wire_gate.MODEL_PATH); DON = dag.get("donusum")

    def donustur(X, pidler):
        if not DON:
            return X
        Z = np.zeros((len(X), X.shape[1] * 2))
        for u in np.unique(pidler):
            i = np.where(pidler == u)[0]
            Z[i] = wire_gate.parca_ici(X[i], DON)
        return Z

    # --- 1. GECIS: mevcut gate skorlari (grup-capraz OOF, sizintisiz)
    M58 = donustur(X58, pid)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(M58, y, grp):
        oof[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                         random_state=0).fit(M58[tr], y[tr]).predict_proba(M58[te])[:, 1]
    print("1. gecis skorlari hazir", flush=True)

    # --- UZAMSAL sutunlar (parca basina)
    SP = np.zeros((len(y), 6))
    for u in np.unique(pid):
        i = np.where(pid == u)[0]
        j = np.where(ppid == u)[0]
        if len(j) != len(i):
            continue
        SP[i] = wire_gate.spatial_feats(PTS[j], oof[i])
    dolu = float((np.abs(SP).sum(1) > 0).mean())
    print(f"uzamsal sutunlar: {SP.shape} | dolu {dolu:.1%}", flush=True)

    # --- OLCUM tarafi: ayni sekilde iki gecis
    kod = {k: collections.Counter(mfg_of.get(p, "?") for p in pid[mfg == k]).most_common(1)[0][0]
           for k in np.unique(mfg)}
    BOLME = [("tanidik", None, DER)]
    for k, mad in kod.items():
        alt = [x for x in DER if x["mfg"] == mad]
        if len(alt) >= 10:
            BOLME.append((mad, k, alt))

    SON, PARCA = {}, {}
    print(f"\n{'kol':<16}" + "".join(f"{b:>12}" for b, _, _ in BOLME) + f"{'dusuk':>9}{'cok':>9}")
    for ad, ek in (("A 58", False), ("B 58+uzamsal", True)):
        Xt = np.hstack([X58, SP]) if ek else X58
        Mt = donustur(Xt, pid)
        SON[ad] = {}; sat = f"{ad:<16}"
        for b, mk, alt in BOLME:
            keep = ~np.isin(grp, list(tg))
            if mk is not None:
                keep = keep & (mfg != mk)
            m1 = {"clf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                                random_state=0).fit(M58[keep], y[keep]),
                  "n_feat": M58.shape[1], "donusum": DON}
            m2 = {"clf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                                random_state=0).fit(Mt[keep], y[keep]),
                  "n_feat": Mt.shape[1], "donusum": DON}
            det, rob = [], []
            for r in alt:
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
                Xr = np.hstack([r["X"], r["XR"]]) if r.get("XR") is not None else None
                if Xr is not None:
                    if ek:
                        s1 = wire_gate.karar_skoru(m1, Xr)          # 1. gecis
                        sp = wire_gate.spatial_feats(r["P"], np.asarray(s1, float))
                        Xr = np.hstack([Xr, sp])
                    s = wire_gate.karar_skoru(m2, Xr)
                    k2 = wire_gate.karar_maskesi(s)
                    if k2.any():
                        P = r["P"][k2]; Pd = r["Pd"][k2]
                rj = "cok" if r["n"] >= 8 else "dusuk"
                det.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
                rob.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
            SON[ad][b] = float(f1w(det))
            PARCA[(ad, b)] = (det, rob, [x["geo"] for x in alt])
            sat += f"{f1w(det):>12.4f}"
        rr = f1_rejim(PARCA[(ad, "tanidik")][0])
        sat += f"{rr['F1']['dusuk']:>9.4f}{rr['F1']['cok']:>9.4f}"
        print(sat, flush=True)

    ga = {}
    for b, _, _ in BOLME:
        if b == "tanidik":
            continue
        da, _, g = PARCA[("A 58", b)]; db, _, _ = PARCA[("B 58+uzamsal", b)]
        fn = lambda rows: f1w([y2 for _, y2 in rows]) - f1w([x for x, _ in rows])
        _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(da, db)), g, fn, n=2000)
        ga[b] = (lo, hi)
    print("\n=== KARAR ===")
    k = karar_olcutu.degerlendir(SON["A 58"], SON["B 58+uzamsal"], ga=ga)
    print(k)
    with io.open("results/v2_uzamsal.json", "w", encoding="utf-8") as f:
        json.dump({"kollar": SON, "gecti": bool(k)}, f, indent=1)
    print("makbuz -> results/v2_uzamsal.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
