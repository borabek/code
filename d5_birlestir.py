# -*- coding: utf-8 -*-
"""D5-2b: 4 VARDIYANIN CIKTISINI BIRLESTIR + KAYIP PARCALARI SAY (D5-5b'nin girdisi).

Turetme 4 paralel vardiyada kostu (`d5_turet_yeni.py --vardiya k --toplam 4`); her biri
kendi `_der_yeni_<k>.pkl` ve `_geo_yeni_<k>.json` dosyasina yazdi. Bu betik onlari TEK
dosyada toplar ve **kimin dustugunu** cikarir.

NEDEN KAYIP LISTESI AYRI CIKARILIYOR: turetme betigi hatali parcalari yalnizca EKRANA
yaziyordu (ilk 5'ini). Kosan surecleri bozmamak icin betigi degistirmedim; kayip listesi
zaten `eligible()` ile turetilenlerin FARKI olarak tam ve kesin sekilde geri alinabilir.

TEKILLIK: ayni pid iki vardiyada olmamali (indeks % 4 ile bolundu). Yine de kontrol edilir
ve cakisma varsa YUKSELTILIR -- sessiz cift-agirlik [[geometry-twin-leakage]] kadar sinsi.
"""
import collections
import glob
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CIKTI = "results/_der_yeni.pkl"
GEO = "results/_geo_yeni.json"
KAYIP = "results/_d5_kayip.json"


def main():
    import protokol
    protokol.tez_dogrula()
    from big_arbiter import eligible

    OUT, GEOD, nerede = [], {}, {}
    kirpilan = 0
    kopya_top = [0]
    # VARDIYA SAYISI SABIT DEGIL (2026-08-05): `range(4)` yaziliydi; o gun turetme once
    # 4, sonra 8 vardiyayla, ardindan iki ayri KURTARMA turuyla (`_dev_*`) kosuldu.
    # Sabit 4 ile birlestirme shard'larin YARISINDAN COGUNU sessizce dusururdu -- hata
    # vermeden, yalnizca "veri kolu ise yaramadi" gibi gorunerek. Artik diskte NE VARSA
    # o okunur; `nerede` sozlugu kopyalari zaten eliyor.
    kaynaklar = sorted(glob.glob("results/_der_yeni_*.pkl"))
    kaynaklar = [f for f in kaynaklar if "_g6" not in os.path.basename(f)]
    print(f"BIRLESTIRILECEK SHARD: {len(kaynaklar)}")
    for f in kaynaklar:
        k = os.path.basename(f)[len("_der_yeni_"):-len(".pkl")]
        if not os.path.exists(f):
            print(f"  vardiya {k}: DOSYA YOK"); continue
        with open(f, "rb") as h:
            R = pickle.load(h)
        # CAKISMA: onceden `assert` idi ve GOREVINI YAPTI -- 2026-08-04'te gercekten atesledi.
        # Sebep: vardiya bolme hatasi duzeltilmeden ONCEKI kosularda her vardiya yalniz kendi
        # ciktisini "zaten var" saydigi icin listeler farklilasmis ve bazi parcalar IKI
        # vardiyada turetilmisti. Kayitlar yanlis DEGIL (ayni parca, ayni boru hatti) ama
        # kopya kayit egitimde CIFT AGIRLIK yaratir -- [[geometry-twin-leakage]] kadar sinsi.
        # Artik: ilk gelen tutulur, kopya ATILIR ve sayisi GURULTULU sekilde raporlanir.
        cak = [r["pid"] for r in R if r["pid"] in nerede]
        if cak:
            print(f"    KOPYA: {len(cak)} parca zaten baska vardiyada var, atiliyor "
                  f"(ornek {cak[:3]})")
            kopya_top[0] += len(cak)
            R = [r for r in R if r["pid"] not in nerede]
        for r in R:
            nerede[r["pid"]] = k
            # OZNITELIK GENISLIGI NORMALIZASYONU (2026-08-04'te yakalandi):
            # `d5_turet_yeni.py` WG_ZENGIN=1 ile kostugu icin `feats_for` zengin sutunlari
            # da X'e EKLIYOR -> X 58 sutun cikiyor. Oysa olcum korpusu (`_der_tam.pkl`) ve
            # DAGITILAN gate 22 bekliyor (n_feat 116 = (22+36)*2). Normalize edilmezse yeni
            # korpus ne eskisiyle birlestirilebilir ne de puanlanabilir.
            # OLCULDU: 984/984 kayitta `X[:, 22:]` ile `XR` BIREBIR AYNI (maks fark 0.0),
            # yani 58 sutun 22'nin UST KUMESI ve dilimleme KAYIPSIZ -- yeniden turetme YOK.
            X = r.get("X")
            if X is not None and X.shape[1] > 22:
                assert X.shape[1] == 22 + r["XR"].shape[1], (
                    f"{r['pid']}: beklenmeyen genislik {X.shape[1]}")
                assert np.abs(X[:, 22:] - r["XR"]).max() < 1e-9, (
                    f"{r['pid']}: X kuyrugu XR ile ayni DEGIL -- dilimleme KAYIPLI olur")
                r["X"] = np.ascontiguousarray(X[:, :22])
                kirpilan += 1
        OUT += R
        g = f"results/_geo_yeni_{k}.json"
        if os.path.exists(g):
            GEOD.update(json.load(io.open(g, encoding="utf-8")))
        print(f"  vardiya {k}: {len(R)} parca")

    with open(CIKTI, "wb") as f:
        pickle.dump(OUT, f)
    with io.open(GEO, "w", encoding="utf-8") as f:
        json.dump(GEOD, f)
    print(f"\nBIRLESIK: {len(OUT)} parca -> {CIKTI}")
    print(f"  geometri anahtari: {len(GEOD)} parca -> {GEO}")

    # --- KAYIP: hedeflenmis ama turetilememis parcalar
    var = set()
    d = np.load("results/zengin_parite_w2.npz", allow_pickle=True)
    var |= {str(x) for x in d["pids"]}
    for y in ("results/_der_tam.pkl", "results/_der_kontrol.pkl"):
        if os.path.exists(y):
            with open(y, "rb") as f:
                var |= {r["pid"] for r in pickle.load(f)}
    turetilen = {r["pid"] for r in OUT}
    E = eligible()
    kayip = [(m, p, s) for m, p, _, s in E if p not in var and p not in turetilen]
    c = collections.Counter(m for m, _, _ in kayip)
    hedef = len([1 for m, p, _, s in E if p not in var])
    print(f"\nKAYIP: {len(kayip)} / {hedef} hedef parca "
          f"(%{100*len(kayip)/max(hedef,1):.1f})")
    print(f"  uretici: {dict(c.most_common(8))}")
    with io.open(KAYIP, "w", encoding="utf-8") as f:
        json.dump([{"mfg": m, "pid": p, "stp": s} for m, p, s in kayip], f, indent=1)
    print(f"  liste -> {KAYIP}  (D5-5b bunu ucuncu mesh kademesiyle yeniden dener)")

    cm = collections.Counter(r["mfg"] for r in OUT)
    print(f"\n  uretici dagilimi: {dict(cm.most_common(10))}")
    print(f"  GT toplam {sum(r['n'] for r in OUT)} | aday {sum(len(r['P']) for r in OUT)}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
