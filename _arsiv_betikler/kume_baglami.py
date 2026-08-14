# -*- coding: utf-8 -*-
"""KUME BAGLAMI: "bu secenek, parcanin DIGER secenekleri arasinda nerede duruyor?"

SORUN. Bugunku secici NOKTASALDIR: her (konum, yon) secenegini tek basina
puanlar. Oysa karar kurali GORELI (parca-maksimumunun %85'i) ve gercek soru
her zaman karsilastirmalidir: "bu aday, ayni delige bakan digerlerinden iyi mi?",
"bu yon, ayni adayin obur yonlerinden iyi mi?", "5mm otede daha guclu bir
rakip var mi?". Noktasal model bunlarin HICBIRINI goremez.

Bu, D2 (aday-kumesi transformer) kolunun UCUZ yaklasimidir: dikkat mekanizmasi
yerine elle secilmis kume ozetleri. Olculen secici verimliligi %36.5 ve hedefe
1.62x gerekiyor; oznitelik bloklari tek basina bunu vermez ama bu blok
"adaylar arasi baglam" hipotezini UCUZA sinar. Kazanc varsa D2'ye yatirim
gerekcelenir; yoksa hipotez zayiflar.

Sutunlar (8):
  log_secenek   ln(1 + parcadaki secenek sayisi)
  log_aday      ln(1 + parcadaki AYRIK aday sayisi)
  aday_secenek  bu adayin kac secenegi var
  aday_sira     bu secenegin AYNI ADAY icindeki skor sirasi (0 = en iyi)
  aday_fark     skor - (ayni adayin en yuksek skoru)      [<= 0]
  en_iyi_uzak   parcanin EN IYI secenegine mesafe / kosegen
  en_iyi_aci    en iyi secenegin yonuyle aci (derece/180)
  rakip_5mm     5mm icinde, skoru BUNDAN YUKSEK ayrik aday sayisi
"""
import numpy as np

OZ_AD = ["log_secenek", "log_aday", "aday_secenek", "aday_sira", "aday_fark",
         "en_iyi_uzak", "en_iyi_aci", "rakip_5mm"]

RAKIP_R = 5.0        # yerel rekabet yaricapi (mm) -- NMS ile ayni olcek


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def oznitelik(P, D, idx, s, diag):
    """(n, 8). `P`,`D` SECENEK konum/yonu; `idx` her secenegin aday indeksi."""
    P = np.asarray(P, float).reshape(-1, 3)
    D = _birim(D)
    idx = np.asarray(idx, int).ravel()
    s = np.asarray(s, float).ravel()
    n = len(P)
    X = np.zeros((n, len(OZ_AD)))
    if not n:
        return X
    diag = max(float(diag), 1e-6)

    X[:, 0] = np.log1p(n)
    ayrik = np.unique(idx)
    X[:, 1] = np.log1p(len(ayrik))

    # --- ADAY ICI: ayni adaya ait secenekler arasinda sira ve fark
    # `idx` ARDISIK OLMAYABILIR; sirali gruplama icin yeniden etiketle.
    _, ters = np.unique(idx, return_inverse=True)
    n_ad = ters.max() + 1
    en_iyi = np.full(n_ad, -np.inf)
    np.maximum.at(en_iyi, ters, s)
    sayi = np.bincount(ters, minlength=n_ad)
    X[:, 2] = sayi[ters]
    X[:, 4] = s - en_iyi[ters]

    # aday ici sira: once adaya, sonra skora gore sirala
    duz = np.lexsort((-s, ters))
    sira = np.empty(n, float)
    k = 0
    while k < n:
        j = k
        a = ters[duz[k]]
        while j < n and ters[duz[j]] == a:
            j += 1
        sira[duz[k:j]] = np.arange(j - k)
        k = j
    X[:, 3] = sira

    # --- PARCANIN EN IYI SECENEGI ile iliski
    b = int(np.argmax(s))
    X[:, 5] = np.linalg.norm(P - P[b], axis=1) / diag
    X[:, 6] = np.degrees(np.arccos(np.clip(D @ D[b], -1, 1))) / 180.0

    # --- YEREL REKABET: 5mm icinde skoru daha yuksek AYRIK aday sayisi
    # aday basina en yuksek skor ve temsili konum
    ad_p = np.zeros((n_ad, 3))
    ad_p[ters] = P                      # ayni adayin secenekleri ayni konumda
    if n_ad <= 4000:
        dm = np.linalg.norm(ad_p[:, None, :] - ad_p[None, :, :], axis=-1)
        yakin = dm <= RAKIP_R
        np.fill_diagonal(yakin, False)
        daha_iyi = en_iyi[None, :] > en_iyi[:, None]
        X[:, 7] = (yakin & daha_iyi).sum(1)[ters]
    return X
