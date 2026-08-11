# -*- coding: utf-8 -*-
"""T8: UYE SECICIYI guclendir -- kahinin kacan %48'i.

DURUM: kahin (GT'ye bakip en iyi uye yonunu secmek) robot-haziri 0.5523 -> 0.6059 yapiyor.
Dagitilan ogrenilmis secici 0.5801'de, yani kahinin %52'si. Kalan %48 = +0.0258.

SECICIDE EKSIK OLAN NE: su an her uye TEK BASINA degerlendiriliyor (kendi guveni, birlesime
uzakligi, ortalamaya acisi). Ama "KAC UYE BU YONDE HEMFIKIR" sorusu hic sorulmuyor -- oysa
bir yonu iki model bagimsiz bulduysa o yon muhtemelen dogrudur. Bu, topluluk yonteminin en
temel sinyali ve havuzda YOK.

EKLENEN OZELLIKLER (5):
    uzl_n       bu uyeye 10 derece icinde olan DIGER uye sayisi
    uzl_conf    o uyelerin guven toplami
    uzl_sira    uzlasma sayisina gore sira
    eksen_hiz   yonun koordinat eksenine hizaliligi (|en buyuk bilesen|)
    baskin_aci  parcanin BASKIN yonune aci (tum uyelerden hesaplanir)

Ayrica HAVUZ genisletiliyor: aci-secicinin duzelttigi yon de bir ADAY olarak havuza girer.

OLCUM: r9 ile AYNI protokol (194 parca, geometri grubuna gore capraz dogrulama) -> yakalama
oranlari dogrudan karsilastirilabilir. Gecerse tam korpusta yeniden uretilir.
"""
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
YAKIN = 5.0


def main():
    import olcum_kumesi
    import wire_gate
    from sina_kume import esle, f1w
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    DER, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    gk = olcum_kumesi.geo_anahtarlari(); tg = {r["geo"] for r in DER}

    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    Xt = np.hstack([np.asarray(zen["X22"], float), np.asarray(zen["XR"], float)])
    ytr = np.asarray(zen["y"]); tpid = np.array([str(x) for x in zen["pids"]])
    keep = ~np.isin(np.array([gk.get(p, "yok:" + p) for p in tpid]), list(tg))
    dag = wire_gate._load(wire_gate.MODEL_PATH); DON = dag.get("donusum")
    Z = np.zeros((len(Xt), Xt.shape[1] * 2))
    for u in np.unique(tpid):
        i = np.where(tpid == u)[0]
        Z[i] = wire_gate.parca_ici(Xt[i], DON)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(Z[keep], ytr[keep])
    gate = {"clf": clf, "n_feat": Z.shape[1], "donusum": DON}
    print("gate hazir", flush=True)

    aci_m = wire_gate._load(wire_gate.ACI_PATH)

    # --- her parca: kabul edilen CP'ler + HAVUZ (uye yonleri + aci-duzeltilmis yon)
    KAY = []
    for r in DER:
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); HAV = []
        if r["X"] is not None and r.get("XR") is not None and r.get("UYE"):
            X58 = np.hstack([r["X"], r["XR"]])
            s = wire_gate.karar_skoru(gate, X58); k = wire_gate.karar_maskesi(s)
            if k.any():
                P = r["P"][k].copy(); Pd = r["Pd"][k].copy()
                cps = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                cps = wire_gate.pose_duzelt(X58[k], cps)
                P = np.array([c["point"] for c in cps], float)
                Pd0 = np.array([c["direction"] for c in cps], float)
                # aci-secicinin onerdigi yon (havuza ADAY olarak girecek)
                aci_yon = Pd0.copy()
                if aci_m is not None:
                    try:
                        pr = aci_m["reg"].predict(X58[k])
                        for i in range(len(P)):
                            d, u, v = wire_gate._yerel_cerceve(Pd0[i])
                            gg = d + float(pr[i, 2]) * u + float(pr[i, 3]) * v
                            aci_yon[i] = gg / (np.linalg.norm(gg) + 1e-9)
                    except Exception:
                        pass
                Pd = Pd0
                Xk = X58[k]
                for i in range(len(P)):
                    uy = [(Pd0[i], 1.0, 0.0), (aci_yon[i], 0.9, 0.0)]
                    for lst in r["UYE"]:
                        for m in lst:
                            q = np.asarray(m["point"], float)
                            dd = float(np.linalg.norm(q - P[i]))
                            if dd <= YAKIN:
                                uy.append((np.asarray(m["direction"], float),
                                           float(m.get("confidence", 1.0)), dd))
                    HAV.append((uy, Xk[i]))
        KAY.append(dict(geo=r["geo"], rj="cok" if r["n"] >= 8 else "dusuk", P=P, Pd=Pd,
                        HAV=HAV, G=np.asarray(r["G"], float), Gd=np.asarray(r["Gd"], float),
                        diag=float(r["diag"])))
    print(f"{len(KAY)} parca | {sum(len(k['HAV']) for k in KAY)} CP", flush=True)

    def ozellik(uy, X58i, zengin):
        DIR = np.array([u[0] for u in uy])
        DIR = DIR / (np.linalg.norm(DIR, axis=1, keepdims=True) + 1e-9)
        CONF = np.array([u[1] for u in uy]); MES = np.array([u[2] for u in uy])
        ort = DIR.mean(0); ort /= np.linalg.norm(ort) + 1e-9
        a_ort = np.degrees(np.arccos(np.clip(np.abs(DIR @ ort), 0, 1)))
        a_bir = np.degrees(np.arccos(np.clip(np.abs(DIR @ DIR[0]), 0, 1)))
        sira = np.argsort(np.argsort(-CONF))
        F = [[CONF[u_], MES[u_], a_ort[u_], a_bir[u_], float(len(DIR)),
              float(np.mean(a_ort)), float(sira[u_])] for u_ in range(len(DIR))]
        if zengin:
            # UZLASMA: bu yone 10 derece icinde kac uye var (KENDISI HARIC)
            M = np.degrees(np.arccos(np.clip(np.abs(DIR @ DIR.T), 0, 1)))
            uzl = (M <= 10.0).sum(1) - 1
            uzl_c = np.array([float(CONF[(M[u_] <= 10.0)].sum() - CONF[u_])
                              for u_ in range(len(DIR))])
            s2 = np.argsort(np.argsort(-uzl))
            # BASKIN yon: isaretten bagimsiz en buyuk ozvektor
            Cm = (DIR[:, :, None] * DIR[:, None, :]).sum(0)
            bas = np.linalg.eigh(Cm)[1][:, -1]
            a_bas = np.degrees(np.arccos(np.clip(np.abs(DIR @ bas), 0, 1)))
            for u_ in range(len(DIR)):
                F[u_] += [float(uzl[u_]), float(uzl_c[u_]), float(s2[u_]),
                          float(np.abs(DIR[u_]).max()), float(a_bas[u_])]
        return DIR, np.array([f + X58i.tolist() for f in F], float)

    SON = {}
    for ad, zengin in (("A mevcut (7+58)", False), ("B zengin (12+58)", True)):
        RX, RY, RG, MAP = [], [], [], []
        for ki, r in enumerate(KAY):
            for i, (uy, X58i) in enumerate(r["HAV"]):
                if not len(r["G"]):
                    continue
                DIR, F = ozellik(uy, X58i, zengin)
                b = int(np.argmin(np.linalg.norm(r["G"] - r["P"][i], axis=1)))
                a = np.degrees(np.arccos(np.clip(np.abs(DIR @ r["Gd"][b]), 0, 1)))
                for u_ in range(len(DIR)):
                    RX.append(F[u_]); RY.append(1 if a[u_] <= 10 else 0); RG.append(r["geo"])
                MAP.append((ki, i, len(DIR), DIR))
        RX = np.array(RX, float); RY = np.array(RY); RG = np.array(RG)
        oof = np.zeros(len(RY))
        for tr, te in GroupKFold(n_splits=5).split(RX, RY, RG):
            oof[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1,
                                             random_state=0).fit(RX[tr], RY[tr]).predict_proba(RX[te])[:, 1]
        # geri dagit
        PD = {ki: r["Pd"].copy() for ki, r in enumerate(KAY)}
        idx = 0
        for ki, i, n, DIR in MAP:
            PD[ki][i] = DIR[int(np.argmax(oof[idx:idx + n]))]
            idx += n
        det, rob = [], []
        for ki, r in enumerate(KAY):
            det.append((r["rj"],) + esle(r["P"], PD[ki], r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((r["rj"],) + esle(r["P"], PD[ki], r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        SON[ad] = (float(f1w(det)), float(f1w(rob)), rob, RX.shape[1])
        print(f"{ad:<20} tespit {SON[ad][0]:.4f} | ROBOT {SON[ad][1]:.4f} "
              f"({RX.shape[1]} sutun, {len(RY)} satir)", flush=True)

    # taban: secicisiz
    det0, rob0 = [], []
    for r in KAY:
        det0.append((r["rj"],) + esle(r["P"], r["Pd"], r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
        rob0.append((r["rj"],) + esle(r["P"], r["Pd"], r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
    t0 = f1w(rob0)
    g = [r["geo"] for r in KAY]
    print(f"\n{'kol':<20}{'robot':>9}{'tabana gore':>13}{'kahinin':>10}")
    print(f"{'taban (secicisiz)':<20}{t0:>9.4f}")
    for ad in SON:
        d = SON[ad][1] - t0
        print(f"{ad:<20}{SON[ad][1]:>9.4f}{d:>+13.4f}{d/0.0536:>10.0%}")
    en = max(SON, key=lambda a: SON[a][1])
    fn = lambda rows: f1w([y for _, y in rows]) - f1w([x for x, _ in rows])
    _, lo, hi = olcum_kumesi.grup_bootstrap(
        list(zip(SON["A mevcut (7+58)"][2], SON["B zengin (12+58)"][2])), g, fn, n=2000)
    dd = SON["B zengin (12+58)"][1] - SON["A mevcut (7+58)"][1]
    print(f"\nA -> B: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] -> "
          f"{'GERCEK' if (lo > 0 or hi < 0) else 'GURULTU'}")
    print(f"KILL: zengin surum >= +0.01 VE GA sifiri dislamali -> "
          f"{'GECTI' if (dd >= 0.01 and lo > 0) else 'GECMEDI'}")
    with io.open("results/t8_uye_zengin.json", "w", encoding="utf-8") as f:
        json.dump({"taban": float(t0),
                   "A": float(SON["A mevcut (7+58)"][1]), "B": float(SON["B zengin (12+58)"][1]),
                   "fark": float(dd), "ga": [float(lo), float(hi)]}, f, indent=1)
    print("makbuz -> results/t8_uye_zengin.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
