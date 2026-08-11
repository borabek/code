# -*- coding: utf-8 -*-
"""P6 KARAR KODU -- egitim de urun de BU fonksiyonlari cagirir.

Bu projede olcum yolu urunden IKI KEZ ayristi ve iki gece kaybettirdi
([[olcum-zaafiyetleri-kapatildi]], [[olcum-yolu-ve-secim-kusurlari]]). Sebep her
seferinde ayniydi: karar kurali iki yerde YAZILMISTI. Burada tek yerde duruyor.

`donustur` : oznitelik donusumu (A+B parca-ici z-skor, C+D ham)
`sec`      : acgozlu skor sirasi + konum NMS -> (P, D)
"""
import numpy as np

AB = 67          # A(58 havuz) + B(9 agiz, adayin kendi yonuyle)
NMS_MM = 5.0     # dagitilan `wire_gate.kalabalik_maskesi` ile ayni olcek


def donustur(X, zskor="ab"):
    """A+B blogu parca-ici z-skor, C+D ham.

    NEDEN AYRI: A+B mutlak buyukluklerdir (olasilik, mm, sayim) ve markadan
    markaya olcegi kayar -- parca-ici z-skor gate tarihindeki en buyuk tek
    kazancti. C+D ZATEN gorelidir (aci, oran, destek yuzdesi); onlari bir daha
    normalize etmek p5-v2'de uctan uca 0.1649 -> 0.1465 DUSURMUSTU.
    """
    import wire_gate
    X = np.asarray(X, float)
    if zskor == "hepsi":
        return wire_gate.parca_ici(X, "zskor")
    if zskor == "yok":
        return X
    if zskor == "sira":
        # PARCA-ICI SIRA (yuzdelik). Z-skor parcanin ORTALAMA ve SAPMASINA
        # baglidir; aday sayisi ve karisimi degisince (egitim ~232 aday/parca,
        # NIT 407 ve cogunlugu mesh) ayni fiziksel aday farkli bir z-skora
        # dusuyor. Sira bu kaymadan ETKILENMEZ: "bu parcadaki en derin 3. delik"
        # ifadesi aday sayisindan bagimsizdir.
        return np.hstack([wire_gate.parca_ici(X[:, :AB], "sira"), X[:, AB:]])
    if zskor == "ikisi":
        # Hem z-skor hem sira: model hangisine nerede guvenecegine karar versin.
        return np.hstack([wire_gate.parca_ici(X[:, :AB], "zskor"),
                          wire_gate.parca_ici(X[:, :AB], "sira")[:, AB:],
                          X[:, AB:]])
    return np.hstack([wire_gate.parca_ici(X[:, :AB], "zskor"), X[:, AB:]])


KAYNAK_AD = ["kay_seg", "kay_brep", "kay_mesh"]


def kaynak_blok(kaynak):
    """Aday kaynagi -> 3 sutunluk gosterge (0 seg / 1 B-rep / 2 mesh tepesi).

    NEDEN GEREKLI: mesh tepeleri havuzun cogunlugunu olusturur (parca basina
    ~250 aday) ve buyuk cogunlugu yanlistir; B-rep agizlari cok daha az ama cok
    daha zengindir; segmentasyonun `v_o` adayi en azdir. Bu ON OLASILIK farkini
    modele soylememek, ona ayni isi ogrenmeyi oznitelikler uzerinden zorlamak
    demek. Onbellek `kaynak` alanini zaten tasiyor -- yeniden cikarim gerekmez.
    """
    k = np.asarray(kaynak, int).reshape(-1)
    return np.stack([(k == 0), (k == 1), (k == 2)], axis=1).astype(float)


def kabul_maskesi(s, kural):
    """Skorlardan KABUL maskesi. `kural` = ("mutlak", e) ya da ("goreli", oran, taban).

    GORELI kural urunun kendi kuralidir (`p1c_esik.maske`): aday, KENDI
    PARCASINDAKI en yuksek skorun `oran` katini gecmeli VE `taban`i asmali.
    Parcalar arasi skor olcegi kaydigi icin mutlak esik bazi parcalarda hicbir
    seyi, bazilarinda her seyi geciriyor.
    """
    s = np.asarray(s, float)
    if not len(s):
        return np.zeros(0, bool)
    if kural[0] == "mutlak":
        return s >= kural[1]
    return (s >= kural[1] * float(np.max(s))) & (s >= kural[2])


def sec_ayrintili(P, idx, YD, s, esik, nms_mm=NMS_MM):
    """`sec` ile AYNI karar, ama secilen ADAY indekslerini de dondurur.

    Teshis icin: "konumu dogru sectik ama yonu mu kacirdik?" sorusu ancak
    secilen adaylarin kimligi bilinerek sorulabilir.
    Doner: (P_sec, D_sec, aday_idx)
    """
    P = np.asarray(P, float).reshape(-1, 3)
    idx = np.asarray(idx, int)
    YD = np.asarray(YD, float).reshape(-1, 3)
    s = np.asarray(s, float)
    kural = ("mutlak", float(esik)) if np.isscalar(esik) else tuple(esik)
    k = np.where(kabul_maskesi(s, kural))[0]
    if not len(k):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
    sira = k[np.argsort(-s[k])]
    ap, ad, ai, kapali = [], [], [], set()
    for j in sira:
        i = int(idx[j])
        if i in kapali:
            continue
        p = P[i]
        if ap and float(np.min(np.linalg.norm(
                np.asarray(ap) - p, axis=1))) < nms_mm:
            kapali.add(i)
            continue
        ap.append(p)
        ad.append(YD[j])
        ai.append(i)
        kapali.add(i)
    return (np.asarray(ap, float).reshape(-1, 3),
            np.asarray(ad, float).reshape(-1, 3), np.asarray(ai, int))


def sec(P, idx, YD, s, esik, nms_mm=NMS_MM):
    """Acgozlu secim: en yuksek skordan basla, konum NMS uygula.

    Bir konum kabul edilince (ya da NMS'e takilinca) o konumun DIGER yon
    secenekleri kapanir -- yani bir konuma EN IYI yon secilir ve bir konum
    en fazla bir tahmin uretir.

    Doner: (P_sec, D_sec)
    """
    P = np.asarray(P, float).reshape(-1, 3)
    idx = np.asarray(idx, int)
    YD = np.asarray(YD, float).reshape(-1, 3)
    s = np.asarray(s, float)
    kural = ("mutlak", float(esik)) if np.isscalar(esik) else tuple(esik)
    k = np.where(kabul_maskesi(s, kural))[0]
    if not len(k):
        return np.zeros((0, 3)), np.zeros((0, 3))
    sira = k[np.argsort(-s[k])]
    ap, ad, kapali = [], [], set()
    for j in sira:
        i = int(idx[j])
        if i in kapali:
            continue
        p = P[i]
        if ap and float(np.min(np.linalg.norm(
                np.asarray(ap) - p, axis=1))) < nms_mm:
            kapali.add(i)
            continue
        ap.append(p)
        ad.append(YD[j])
        kapali.add(i)
    return (np.asarray(ap, float).reshape(-1, 3),
            np.asarray(ad, float).reshape(-1, 3))
