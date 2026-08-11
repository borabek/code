# -*- coding: utf-8 -*-
"""Q1: CESITLILIK mi HACIM mi? -- "ucuncu uretici" tavsiyesinin BELIRLEYICI deneyi.

P1 ogrenme egrisi "daha cok veri" dedi ama asil soru cevapsiz kaldi: AYNI TOPLAM BOYUTTA,
tek ureticiden gelen veri mi yoksa KARISIK veri mi daha iyi? Cunku:
  * cevap "hacim" ise -> hangi ureticiden geldigi onemsiz, en ucuz kaynaktan al
  * cevap "cesitlilik" ise -> UCUNCU URETICI, ayni sayida PXC/WEI parcasindan DAHA
    degerlidir ve talep buna gore yazilmalidir

TASARIM (esit hacim, farkli bilesim): her boyut N icin
    A) yalniz PXC'den N grup
    B) yalniz WEI'den N grup
    C) yari PXC + yari WEI, toplam N grup
Uc kol da AYNI N. Fark yalniz BILESIM. Olcum kumesi sabit; 3 tohum.

BEKLENTI: C, A ve B'nin ortalamasindan BELIRGIN yuksekse cesitliligin hacimden BAGIMSIZ
bir degeri vardir ve ucuncu uretici talebi hakli cikar.

Hicbir sey dagitilmaz.
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

BOYUTLAR = [120, 200, 300]
TOHUMLAR = (0, 1, 2)


def main():
    import gate_tezgah as T
    import olcum_kumesi
    import wire_gate
    from sklearn.ensemble import RandomForestClassifier
    from gece_kilit import bekci

    bekci("q1")
    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    X, y, pid, mfg, keep = D["X"], D["y"], D["pid"], D["mfg"], D["keep"]
    gk = D["gk"]
    grup = np.array([gk.get(p, "yok:" + p) for p in pid])
    Z = np.zeros((len(X), X.shape[1] * 2))
    for u in np.unique(pid):
        i = np.where(pid == u)[0]
        Z[i] = wire_gate.parca_ici(X[i], D["donusum"])

    kod_ters = {v: k for k, v in D["kod"].items()}
    kP, kW = kod_ters.get("PXC"), kod_ters.get("WEI")
    gP = np.unique(grup[keep & (mfg == kP)])
    gW = np.unique(grup[keep & (mfg == kW)])
    print(f"\nkullanilabilir grup: PXC {len(gP)} | WEI {len(gW)}")
    ust = min(len(gP), len(gW))
    boyutlar = [b for b in BOYUTLAR if b <= ust]
    print(f"denenecek boyutlar: {boyutlar} (tek-uretici kollari icin ust sinir {ust})")

    def kol_yap(secim_fn, tohum_alt):
        def kol(X_, y_, pid_, mfg_, kp, th):
            sec = secim_fn(tohum_alt)
            m = kp & np.isin(grup, sec)
            if len(np.unique(y_[m])) < 2 or m.sum() < 200:
                m = kp
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                         random_state=th).fit(Z[m], y_[m])
            return {"clf": clf, "n_feat": Z.shape[1], "donusum": D["donusum"]}
        return kol

    SON = {}
    print(f"\n{'kol':<26}{'havuz':>9}{'PXC-disi':>10}{'WEI-disi':>10}{'URET-ORT':>10}")
    for N in boyutlar:
        for ad, fn in (
            (f"A yalniz PXC N={N}", lambda t, N=N: np.random.default_rng(t).permutation(gP)[:N]),
            (f"B yalniz WEI N={N}", lambda t, N=N: np.random.default_rng(t).permutation(gW)[:N]),
            (f"C KARISIK   N={N}", lambda t, N=N: np.concatenate([
                np.random.default_rng(t).permutation(gP)[:N // 2],
                np.random.default_rng(t + 77).permutation(gW)[:N - N // 2]])),
        ):
            h, u, px, we = [], [], [], []
            for th in TOHUMLAR:
                S, _ = T.calistir(D, kol_yap(fn, th), ad, tohumlar=(th,), ayrinti=False)
                h.append(S["havuzlanmis"]["tespit"]); u.append(S["_URETICI_DISI_ORT"])
                px.append(S.get("PXC-disi", {}).get("tespit", np.nan))
                we.append(S.get("WEI-disi", {}).get("tespit", np.nan))
            SON[ad] = {"havuz": float(np.mean(h)), "uretici": float(np.mean(u)),
                       "pxc_disi": float(np.nanmean(px)), "wei_disi": float(np.nanmean(we)),
                       "sapma": float(np.std(h))}
            s = SON[ad]
            print(f"{ad:<26}{s['havuz']:>9.4f}{s['pxc_disi']:>10.4f}"
                  f"{s['wei_disi']:>10.4f}{s['uretici']:>10.4f}", flush=True)
            bekci("q1 " + ad)

    print(f"\n{'N':<7}{'A(PXC)':>9}{'B(WEI)':>9}{'ort(A,B)':>10}{'C(karisik)':>12}{'CESITLILIK':>12}")
    KAZANC = {}
    for N in boyutlar:
        a = SON[f"A yalniz PXC N={N}"]["havuz"]; b = SON[f"B yalniz WEI N={N}"]["havuz"]
        c = SON[f"C KARISIK   N={N}"]["havuz"]
        KAZANC[N] = c - (a + b) / 2
        print(f"{N:<7}{a:>9.4f}{b:>9.4f}{(a+b)/2:>10.4f}{c:>12.4f}{KAZANC[N]:>+12.4f}")
    ort = float(np.mean(list(KAZANC.values())))
    print(f"\nORTALAMA CESITLILIK KAZANCI: {ort:+.4f}")
    if ort >= 0.02:
        print("  -> CESITLILIGIN HACIMDEN BAGIMSIZ DEGERI VAR. Ucuncu uretici talebi HAKLI:")
        print("     ayni sayida PXC/WEI parcasindan DAHA degerlidir.")
    elif ort >= 0.005:
        print("  -> Cesitliligin ZAYIF ama pozitif degeri var; talep hacim uzerinden yazilmali,")
        print("     cesitlilik ikincil gerekce olarak eklenmeli.")
    else:
        print("  -> CESITLILIK EK DEGER VERMIYOR: onemli olan HACIM. Ucuncu uretici yerine")
        print("     EN UCUZ kaynaktan parca almak dogru strateji.")
    with io.open("results/q1_cesitlilik.json", "w", encoding="utf-8") as f:
        json.dump({"sonuc": SON, "cesitlilik_kazanci": {str(k): float(v)
                                                        for k, v in KAZANC.items()},
                   "ortalama": ort}, f, indent=1)
    print("makbuz -> results/q1_cesitlilik.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
