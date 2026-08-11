# -*- coding: utf-8 -*-
"""OLCUM KUMESI -- puanlanacak parcalarin TEK ve DENETLENEBILIR kaynagi.

2026-08-01 DENETIMI dort ayri kusur buldu ve dordu de burada kapatiliyor:

1. **DEV/VAL ADLARI YANLIS BAGLIYDI.** `manset.py` kumeleri `results/_probs_{dev,val}.pkl`
   onbelleklerinden cikariyordu ve "dev" aslinda ESKI `_h_probs.pkl` onbellegiydi. Gercek bolme
   `results/split3.json`'da (2026-07-31, keskin geometri anahtari). Iki atama %100 UYUSMUYORDU:
   benim "dev 97 / val 100" dedigim kume, gercekte **54 DEV / 100 VAL / 40 ATANMAMIS / 3 LOCKED**.

2. **TEKRAR EDEN PARCA.** 200 satir aslinda 197 parca; uc parca (3214360, 2428950000,
   1010880000) iki kez sayiliyordu -- eski DEV ve VAL onbellekleri kesisiyor.

3. **LOCKED KIRLENMESI.** Uc LOCKED parca DOGRUDAN puanlanmis; geometri grubu uzerinden
   toplam bes LOCKED parca dort grupta kirlenmis. Geriye **95 grup-temiz LOCKED** kaliyor ve
   bunlar TEK ATIS icin saklanmalidir.

4. **PARCA BOOTSTRAP'I.** 197 parca 174 GEOMETRI GRUBUNA dusuyor; parca ornekleyen bir
   bootstrap, ayni geometrinin ikizlerini bagimsiz sayarak guven araligini DARALTIR.
   Dogru birim GRUP'tur.

Bu modul kumeyi kurar, kirliligi ROPORTAJ EDER ve grup-bootstrap birimini verir.
Puanlama yapan her betik BURAYI cagirir; kume tanimi bir daha kopyalanmaz.
"""
import collections
import json
import io
import os
import pickle

import numpy as np

SPLIT = "results/split3.json"
GEO = "results/_strict_geometry_keys.json"


def _oku(yol):
    with io.open(yol, encoding="utf-8") as f:
        return json.load(f)


def bolme_atamasi():
    """pid -> 'dev' | 'val' | 'locked' (split3.json, TEK dogruluk kaynagi)."""
    s3 = _oku(SPLIT)
    ata = {}
    for k in ("dev", "val", "locked"):
        for p in s3[k]["parts"]:
            ata[str(p)] = k
    return ata


def geo_anahtarlari():
    return _oku(GEO)


def kume(der_yolu="results/_u4_der.pkl", locked_cikar=True, tekrar_cikar=True):
    """Puanlanacak kaydi dondur + kirlilik raporu.

    Doner: (KAYITLAR, rapor). Her kayda `kume` (dev/val/atanmamis) ve `geo` eklenir.
    """
    with open(der_yolu, "rb") as f:
        DER = pickle.load(f)
    ata = bolme_atamasi()
    gk = geo_anahtarlari()

    gorulen, temiz, atilan_tekrar, atilan_locked = set(), [], [], []
    for r in DER:
        pid = str(r["pid"])
        if tekrar_cikar and pid in gorulen:
            atilan_tekrar.append(pid)
            continue
        gorulen.add(pid)
        b = ata.get(pid, "atanmamis")
        if locked_cikar and b == "locked":
            atilan_locked.append(pid)
            continue
        r = dict(r)
        r["kume"] = b
        r["geo"] = gk.get(pid, "yok:" + pid)
        temiz.append(r)

    # KIRLILIK, PUANLAMADAN CIKARILANLARI DA KAPSAR (2026-08-02 denetimi).
    # Onceki surum `kullanilan_geo`yu yalniz KALAN parcalardan hesapliyordu; boylece
    # DOGRUDAN kullanilip cikarilan LOCKED parcalarin KENDI GRUBU "dokunulmamis" gorunuyor
    # ve o parcalar yeniden TEMIZ sayiliyordu. Ama onlar KULLANILDI -- olcum onbelleginde
    # varlar ve tarihsel olarak puanlandilar. Bir kez dokunulan parca kalici olarak kirlidir.
    # Etkisi: "temiz LOCKED 98" -> DOGRUSU 95.
    kullanilan_geo = ({r["geo"] for r in temiz}
                      | {gk.get(p, "yok:" + p) for p in atilan_locked}
                      | {gk.get(p, "yok:" + p) for p in atilan_tekrar})
    s3 = _oku(SPLIT)
    locked_kirli = {}
    for p in s3["locked"]["parts"]:
        g = gk.get(str(p), "yok:" + str(p))
        if g in kullanilan_geo:
            locked_kirli.setdefault(g, []).append(str(p))
    locked_temiz = [str(p) for p in s3["locked"]["parts"]
                    if gk.get(str(p), "yok:" + str(p)) not in kullanilan_geo]

    rapor = {
        "girdi_satir": len(DER),
        "puanlanan_parca": len(temiz),
        "atilan_tekrar": atilan_tekrar,
        "atilan_locked": atilan_locked,
        "kume_dagilimi": dict(collections.Counter(r["kume"] for r in temiz)),
        "geometri_grubu": len(kullanilan_geo),
        "locked_kirli_grup": {g: v for g, v in locked_kirli.items()},
        "locked_kirli_parca": sorted({p for v in locked_kirli.values() for p in v}),
        "locked_temiz_n": len(locked_temiz),
        "locked_temiz": locked_temiz,
    }
    return temiz, rapor


def grup_bootstrap(satirlar, gruplar, f1_fn, n=4000, tohum=0):
    """GRUP birimli bootstrap. Parca degil GEOMETRI GRUBU ornekler.

    NEDEN: 197 parca 174 gruba dusuyor ve gruplarin bir kismi IKIZ parcalar iceriyor.
    Parca ornekleyen bootstrap ikizleri bagimsiz gozlem sayar -> guven araligi SAHTE DARALIR.
    Grup ornekleyince bagimlilik korunur.
    """
    rng = np.random.default_rng(tohum)
    grup_ind = collections.defaultdict(list)
    for i, g in enumerate(gruplar):
        grup_ind[g].append(i)
    anahtar = list(grup_ind)
    v = []
    for _ in range(n):
        sec = rng.integers(0, len(anahtar), len(anahtar))
        idx = [i for k in sec for i in grup_ind[anahtar[k]]]
        v.append(f1_fn([satirlar[i] for i in idx]))
    v = np.asarray(v, float)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rapor_bas(rapor):
    print(f"OLCUM KUMESI: {rapor['girdi_satir']} satir -> {rapor['puanlanan_parca']} parca "
          f"({rapor['geometri_grubu']} geometri grubu)")
    print(f"  kume: {rapor['kume_dagilimi']}")
    if rapor["atilan_tekrar"]:
        print(f"  ATILAN TEKRAR ({len(rapor['atilan_tekrar'])}): {rapor['atilan_tekrar']}")
    if rapor["atilan_locked"]:
        print(f"  ATILAN LOCKED ({len(rapor['atilan_locked'])}): {rapor['atilan_locked']}")
    print(f"  LOCKED kirlenmesi: {len(rapor['locked_kirli_grup'])} grup / "
          f"{len(rapor['locked_kirli_parca'])} parca -> {rapor['locked_kirli_parca']}")
    print(f"  GRUP-TEMIZ LOCKED: {rapor['locked_temiz_n']} parca (TEK ATIS icin saklanir)")




def locked_gruplari():
    """95 grup-temiz LOCKED parcanin GEOMETRI GRUPLARI."""
    _, rap = kume("results/_der_tam.pkl")
    gk = geo_anahtarlari()
    return {gk.get(p, "yok:" + p) for p in rap["locked_temiz"]}


def sinav_egitim_maskesi(pidler):
    """FINAL SINAVI icin egitim maskesi: LOCKED'in GEOMETRI GRUPLARINI disla.

    2026-08-03 BULGUSU (q2_sinav_butunlugu.py): dagitilan egitim verisi
    `zengin_parite_w2.npz` 95 grup-temiz LOCKED parcasinin **77'sini** iceriyor
    (grup uzerinden 80). Sinav dagitilan gate'le kosulsaydi 95 parcanin 80'i KIRLI
    olurdu ve sonuc SISIK cikardi.

    NEDEN GOZDEN KACTI: "LOCKED'a dokunulmadi" hep "puanlamiyoruz" diye anlasildi.
    Olcum betikleri yalnizca OLCUM KUMESININ gruplarini egitimden atiyor; LOCKED'in
    gruplari hicbir zaman atilmadi cunku LOCKED hic puanlanmamisti.

    KURAL: LOCKED uzerinde olculecek her gate, bu maskeyle egitilmek ZORUNDADIR.
    Aksi halde sinav GECERSIZDIR ve tek atislik hakki bosa gider.
    """
    import numpy as np
    gk = geo_anahtarlari()
    lg = locked_gruplari()
    g = np.array([gk.get(str(p), "yok:" + str(p)) for p in pidler])
    return ~np.isin(g, list(lg))


if __name__ == "__main__":
    sat, rap = kume()
    rapor_bas(rap)
    with io.open("results/olcum_kumesi.json", "w", encoding="utf-8") as f:
        json.dump(rap, f, indent=1, ensure_ascii=False)
    print("makbuz -> results/olcum_kumesi.json")
