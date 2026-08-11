# -*- coding: utf-8 -*-
"""U10: KAHIN ORANINI DURUST OLC -- taban, kol ve kahin AYNI KOSUDAN.

DUZELTILEN HATA (2026-08-02): T8'de "kahinin %66'si yakalandi" yazdim; kahin payini SABIT
0.0536'ya boldum. O sabit BASKA BIR ZINCIRDE olculmustu (uye secici dagitilmadan once,
taban 0.5523). T8'in kendi tabani ise 0.5280 idi. Taban degisince hem pay hem payda kayar ve
oran ANLAMSIZLASIR. Nitekim tek degiskenle olcunce "havuz bilesimi" farki yok oldu.

Bu, `f1w` tuzagiyla ayni sinifta: FARKLI KOSULARDAN gelen sayilari ayni formulde kullanmak.

BU BETIK ucunu de TEK KOSUDA olcer:
    T  taban        : uye secici KAPALI (pose + aci)
    K  kol          : uye secici ACIK (dagitilan)
    O  kahin        : GT'ye bakip havuzdaki EN IYI yonu sec (ulasilabilir DEGIL, ust sinir)
ve oranI (K-T)/(O-T) olarak verir -- payda AYNI kosudan.

Ayrica GUNCEL bosslugu verir: dagitilan zincir su an nerede, kahin nerede.
"""
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
YAKIN = 5.0


def kahin_orani(taban, kol, kahin):
    """(kol - taban) / (kahin - taban). UCU DE AYNI KOSUDAN gelmeli.

    Sabit bir kahin degerine bolmek YANLISTIR: taban degisince oran anlamini yitirir
    (2026-08-02'de tam bu hata yapildi). Bu fonksiyon ucunu birden ister ki ayrisamasinlar.
    """
    pay = kol - taban
    payda = kahin - taban
    if payda <= 1e-9:
        return float("nan")
    return pay / payda


def main():
    import olcum_kumesi
    import wire_gate
    from sina_kume import esle, f1w
    from sklearn.ensemble import RandomForestClassifier

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    DER, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    gk = olcum_kumesi.geo_anahtarlari(); tg = {r["geo"] for r in DER}
    g = [r["geo"] for r in DER]

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
    uye_m = wire_gate._load(wire_gate.UYE_PATH)
    print(f"gate + uye secici hazir", flush=True)

    def havuz_kur(P, Pd, i, uyeler):
        hav = [(Pd[i], 1.0, 0.0)]
        for lst in uyeler:
            for m in lst:
                q = np.asarray(m["point"], float)
                dd = float(np.linalg.norm(q - P[i]))
                if dd <= YAKIN:
                    hav.append((np.asarray(m["direction"], float),
                                float(m.get("confidence", 1.0)), dd))
        return hav

    # SECIM URUNUN KENDI FONKSIYONUNDAN. Ilk surumde burada UCUNCU kez elle yeniden
    # yazilmisti ve 0.5704 veriyordu; dagitilan `wire_gate.uye_yonu_sec` 0.5801 veriyor
    # (iki bagimsiz betikte dogrulandi). Yani sapma URUNDE degil OLCUMDE idi -- bu gece
    # konulan 'olcum urunu TAKLIT ETMEZ, AYNI KODU cagirir' kuralinin ihlaliydi.

    def puanla(mod):
        """mod: 'taban' (secici yok) | 'kol' (dagitilan secici) | 'kahin' (GT'ye bakar)."""
        det, rob = [], []
        for r in DER:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if r["X"] is not None and r.get("XR") is not None:
                X58 = np.hstack([r["X"], r["XR"]])
                s = wire_gate.karar_skoru(gate, X58); k = wire_gate.karar_maskesi(s)
                if k.any():
                    P = r["P"][k].copy(); Pd = r["Pd"][k].copy()
                    cps = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                    cps = wire_gate.pose_duzelt(X58[k], cps)
                    cps = wire_gate.aci_duzelt(X58[k], cps)
                    P = np.array([c["point"] for c in cps], float)
                    Pd = np.array([c["direction"] for c in cps], float)
                    if mod == "kol" and r.get("UYE"):
                        cps = wire_gate.uye_yonu_sec(X58[k], cps, r["UYE"])
                        Pd = np.array([c["direction"] for c in cps], float)
                    elif mod == "kahin" and r.get("UYE"):
                        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
                        for i in range(len(P)):
                            hav = havuz_kur(P, Pd, i, r["UYE"])
                            if len(hav) < 2 or not len(G):
                                continue
                            DIR = np.array([h[0] for h in hav])
                            DIR = DIR / (np.linalg.norm(DIR, axis=1, keepdims=True) + 1e-9)
                            b = int(np.argmin(np.linalg.norm(G - P[i], axis=1)))
                            a = np.degrees(np.arccos(np.clip(np.abs(DIR @ Gd[b]), 0, 1)))
                            Pd[i] = DIR[int(np.argmin(a))]
            rj = "cok" if r["n"] >= 8 else "dusuk"
            det.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        return det, rob

    R = {}
    for mod, ad in (("taban", "T taban (secici KAPALI)"), ("kol", "K dagitilan secici"),
                    ("kahin", "O KAHIN (ust sinir)")):
        d, rr = puanla(mod)
        R[mod] = (float(f1w(d)), float(f1w(rr)), rr)
        print(f"{ad:<26}tespit {R[mod][0]:.4f} | ROBOT {R[mod][1]:.4f}", flush=True)

    T, K, O = R["taban"][1], R["kol"][1], R["kahin"][1]
    oran = kahin_orani(T, K, O)
    print(f"\n=== AYNI KOSUDAN ===")
    print(f"  taban {T:.4f} -> kol {K:.4f} -> kahin {O:.4f}")
    print(f"  kolun kazanci {K-T:+.4f} | kahinin toplami {O-T:+.4f} | YAKALAMA {oran:.0%}")
    print(f"  KALAN BOSLUK: {O-K:+.4f}")
    fn = lambda rows: f1w([y for _, y in rows]) - f1w([x for x, _ in rows])
    _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(R["taban"][2], R["kol"][2])), g, fn, n=2000)
    print(f"  kolun GA'si [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nESKI (YANLIS) HESAP: kahin payini SABIT 0.0536'ya bolmustum; o sabit")
    print(f"BASKA bir zincirden geliyordu (taban 0.5523). Dogru payda: {O-T:.4f}")
    with io.open("results/u10_kahin_orani.json", "w", encoding="utf-8") as f:
        json.dump({"taban": T, "kol": K, "kahin": O, "yakalama": float(oran),
                   "kalan_bosluk": float(O - K), "ga": [float(lo), float(hi)]}, f, indent=1)
    print("makbuz -> results/u10_kahin_orani.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
