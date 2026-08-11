# -*- coding: utf-8 -*-
"""YON SECENEK BANKASI -- bir konuma BIRDEN COK yon onerir.

NEDEN. Dagitilan urun her konuma TEK yon bagliyor: havuzdan gelen `D`, uzerine
isaret duzeltmesi (`urun_genis.sec`). Oysa tavan merdiveni (`results/tavan_085.json`,
D7 marka-disi, mukemmel secici) sunu soyluyor:

    P2+Y0  yalniz adayin kendi yonu     recall 0.4820   F1 tavani 0.6505
    P2+Y3  + komsu + silindir + ana     recall 0.6391   F1 tavani 0.7798

Yani konum ZATEN dogruyken yalnizca yon yanlis oldugu icin kaybedilen GT sayisi
D7'de **485** (Y1 komsu 197 + Y2 silindir 170 + Y3 ana eksen 118). Kazanan
yapilandirmanin hata bankasinda (`results/kazanan_hata_bankasi_v2.json`) buna
karsilik gelen kova `YON_YOK = 539`. Iki bagimsiz sayim ayni yeri gosteriyor.

BU MODUL o seceneklerin DAGITILABILIR halini uretir. Tavan betigi
(`sonda_tavan_085.py`) "bilgi var mi" diye soruyordu ve parca basina binlerce
konum kullaniyordu; burada secenek sayisi aday basina `MAX_SEC` ile sinirli ve
her secenek FIZIKSEL bir kaynaktan geliyor.

KAYNAKLAR (hepsi +/- ciftli -- olcum ISARETLI aci kullanir):
  0 KENDI     adayin kendi yonu. HER ZAMAN 0. secenek ve gercek fallback.
  1 KOMSU     10mm icindeki komsu adaylarin yonleri (sira mutabakati)
  2 SILINDIR  parcadaki B-rep silindir eksenleri
  3 ANA       parcanin ana eksenleri (SVD) -- klemenste giris yonu cogunlukla
              govde eksenlerinden biridir, bedava bilgi

TEZ: `v_o` turetmesi, 5 sinif ve remesh DEGISMEDI. Bu modul yalniz PUANLANACAK
secenekleri cogaltir; tezin cevabi her zaman kaynak 0 ile havuzda.
"""
import os

import numpy as np

# SERBEST-DERINLIK YELPAZESI (`YB_FAN` ile acilir).
# OLCULDU: 64 yonlu sonda NIT'te +0.0000 verdi ve kolu "OLU" ilan etmistim.
# Oysa 64 yonun kure uzerindeki komsuluk araligi ~25 derece, olcum toleransi
# 10 derece -- sonda etkiyi FIZIKSEL OLARAK olcemezdi. 256 yonle ayni kol
# NIT'te +0.0676 recall verdi (BANKA 0.5676 -> 0.6351).
# Fizik: gercek giris yonu, agizdan DISARI en uzun bos yolu olan yondur.
FAN_N = int(os.environ.get("YB_FAN", "0"))      # 0 = kapali, 256 onerilen
FAN_K = int(os.environ.get("YB_FAN_K", "3"))    # aday basina kac yon onerisi

KOMSU_R = 10.0          # komsu yonu toplama yaricapi (mm)
DEDUPE_DER = 8.0        # bu aciyla ayni sayilan yonler tek temsilciye iner
# ADAY BASINA SECENEK TAVANI. OLCULDU (2026-08-12, 30 parca, yelpaze 256):
#   tavan 12 -> yonlu recall 0.8462 (maliyet 1.00x)   <- bugunku
#   tavan 24 -> yonlu recall 0.9077 (maliyet 1.20x)   <- DIZ NOKTASI
#   tavan 40 -> yonlu recall 0.9077 (maliyet 1.23x)   kazanc yok
# Tavan BAGLIYOR: 256 isinlik yelpaze onlarca yon uretiyor ama cogu bu tavanda
# eleniyor -- yelpaze yeni yon EKLEMIYOR, mevcut kaynaklarin yerini ALIYOR.
# Sabit oldugu icin yeniden cikarimla denenemiyordu; artik cevreden ayarlanir.
MAX_SEC = int(os.environ.get("YB_MAX_SEC", "12"))
DESTEK_DER = 10.0       # "bu yonu destekliyor" esigi -- olcumun ACI'siyla ayni

KAYNAK_AD = ["kendi", "komsu", "silindir", "ana"]
OZ_AD = ["k_kendi", "k_komsu", "k_silindir", "k_ana",
         "aci_kendi", "yerel_destek", "yerel_destek_n", "parca_destek",
         "sil_hiza", "sil_yaricap", "ana_hiza", "eksen_hiza",
         "gate_s", "votes", "n_aday", "n_secenek"]


def birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def dedupe(Y, kaynak, tol_der=DEDUPE_DER):
    """ISARETLI dedupe. +u ve -u AYRI yonlerdir; olcum isaretli aci kullaniyor.

    Ilk gelen kazanir; kaynak sirasi (kendi<komsu<silindir<ana) cagiranda kurulur,
    boylece ayni yonun en 'ucuz' kaynagi temsilci olur.
    """
    if not len(Y):
        return np.zeros((0, 3)), np.zeros(0, int)
    Y = birim(Y)
    cos_tol = np.cos(np.radians(tol_der))
    tut, tk = [], []
    for i in range(len(Y)):
        if tut and float(np.max(np.asarray(tut) @ Y[i])) >= cos_tol:
            continue
        tut.append(Y[i])
        tk.append(int(kaynak[i]))
    return np.asarray(tut, float), np.asarray(tk, int)


def parca_yonleri(cyl, V):
    """Parca duzeyinde yon kaynaklari. Doner: (Ysil, Yana) -- ikisi de +/- ciftli."""
    eks = []
    for c in (cyl or []):
        a = np.asarray(c.get("axis", [0, 0, 0]), float)
        if np.linalg.norm(a) > 1e-9:
            eks.append(a)
    Ysil = np.zeros((0, 3))
    if eks:
        e = birim(eks)
        Ysil = np.vstack([e, -e])
    Yana = np.zeros((0, 3))
    V = np.asarray(V, float).reshape(-1, 3)
    if len(V) > 3:
        a = birim(np.linalg.svd(V - V.mean(0), full_matrices=False)[2])
        Yana = np.vstack([a, -a])
    return Ysil, Yana


def _silindir_olcu(cyl, p, y):
    """Bu yone en cok uyan silindirin (hiza, yaricap). Yoksa (0,0)."""
    en = (0.0, 0.0)
    for c in (cyl or []):
        a = np.asarray(c.get("axis", [0, 0, 0]), float)
        n = np.linalg.norm(a)
        if n < 1e-9:
            continue
        h = abs(float((a / n) @ y))
        if h > en[0]:
            en = (h, float(c.get("radius", 0.0) or 0.0))
    return en


def fibonacci_kure(n):
    """n yonun kure uzerinde MUMKUN OLDUGUNCA ESIT dagilimi."""
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    th = phi * i
    return np.stack([np.cos(th) * r, y, np.sin(th) * r], axis=1)


def yelpaze_yonleri(P, mesh, diag, n_yon=None, k=None):
    """`n_yon`/`k` None ise MODUL GLOBALLERI okunur.

    Varsayilan argumanlari `FAN_N`'e baglamak, `YB.FAN_N = 256` yazan bir
    cagriyi SESSIZCE etkisiz birakiyordu (varsayilan tanim aninda baglanir).
    """
    n_yon = FAN_N if n_yon is None else n_yon
    k = FAN_K if k is None else k
    """Her aday icin EN DERIN `k` serbest yon. Doner: (n_aday, k, 3).

    Isinlar TEK cagrida topluca atilir; `agiz_tanimlayici._ilk_mesafe` zaten
    topakli oldugu icin bellek patlamaz.
    """
    if n_yon <= 0 or mesh is None or not len(P):
        return np.zeros((len(P), 0, 3))
    import agiz_tanimlayici as AT
    F = fibonacci_kure(n_yon)
    eps = max(1e-3, 1e-4 * diag)
    nc = len(P)
    O = np.repeat(P, n_yon, axis=0) + eps * np.tile(F, (nc, 1))
    Dv = np.tile(F, (nc, 1))
    der = AT._ilk_mesafe(mesh, O, Dv, diag).reshape(nc, n_yon)
    top = np.argsort(-der, axis=1)[:, :k]
    return F[top]


def secenekler(P, D, cyl, V, gate_s=None, votes=None,
               komsu_r=KOMSU_R, max_sec=MAX_SEC, mesh=None, diag=None,
               fan_maske=None):
    """Aday basina yon secenekleri.

    Doner: (idx, YD, OZ)
      idx (k,)   -> hangi adayin secenegi
      YD  (k,3)  -> secenek yonu (birim)
      OZ  (k,K)  -> `OZ_AD` sirasinda oznitelikler

    Konum DEGISMEZ: `P[idx[j]]`. Bu modul yalniz YON cogaltir; konum havuzu
    cagiranin isidir.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    D = birim(D)
    n = len(P)
    if n == 0:
        return (np.zeros(0, int), np.zeros((0, 3)),
                np.zeros((0, len(OZ_AD))))
    gate_s = np.zeros(n) if gate_s is None else np.asarray(gate_s, float)
    votes = np.zeros(n) if votes is None else np.asarray(votes, float)
    Ysil, Yana = parca_yonleri(cyl, V)
    cos_destek = np.cos(np.radians(DESTEK_DER))
    # YELPAZE: aday basina en derin serbest yonler.
    # `fan_maske` verilirse YALNIZ o adaylara uygulanir. Mesh tepelerine
    # yelpaze atmak cok pahali (600 aday x 256 isin) ve gereksiz: mesh adayi
    # zaten kendi TEPE NORMALINI tasiyor ve parca duzeyi yonleri bankadan
    # aliyor. Yelpazenin olculen degeri B-REP AGIZLARINDA idi.
    Yfan = np.zeros((n, 0, 3))
    if FAN_N > 0 and mesh is not None:
        if fan_maske is None:
            Yfan = yelpaze_yonleri(P, mesh, diag)
        else:
            fm = np.asarray(fan_maske, bool)
            Yfan = np.zeros((n, FAN_K, 3))
            if fm.any():
                Yfan[fm] = yelpaze_yonleri(P[fm], mesh, diag)

    # komsuluk: her aday icin 10mm icindeki adaylarin indeksleri
    d2 = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    idx, YD, OZ = [], [], []
    for i in range(n):
        yakin = np.where(d2[i] <= komsu_r)[0]
        aday = [(D[i], 0)]
        for j in yakin:
            if j != i:
                aday.append((D[j], 1))
        for y in Ysil:
            aday.append((y, 2))
        for y in Yana:
            aday.append((y, 3))
        if Yfan.shape[1]:
            for y in Yfan[i]:
                if float(np.dot(y, y)) > 0.5:   # maske disi adayda sifir vektor
                    aday.append((y, 3))         # ana eksenlerle AYNI kaynak kodu:
                                                # C blogunun genisligi degismesin

        Y, KY = dedupe(np.asarray([a[0] for a in aday], float),
                       np.asarray([a[1] for a in aday], int))
        if len(Y) > max_sec:                 # kendi (0. sira) HER ZAMAN kalir
            Y, KY = Y[:max_sec], KY[:max_sec]
        # destek olculeri -- toplu
        cy = D[yakin] @ Y.T if len(yakin) else np.zeros((0, len(Y)))
        yerel_n = (cy >= cos_destek).sum(0) if len(yakin) else np.zeros(len(Y))
        yerel = yerel_n / max(len(yakin), 1)
        parca = (D @ Y.T >= cos_destek).mean(0)
        aci0 = np.degrees(np.arccos(np.clip(Y @ D[i], -1.0, 1.0)))
        for t in range(len(Y)):
            y = Y[t]
            sh, sr = _silindir_olcu(cyl, P[i], y)
            ah = float(np.max(np.abs(Yana @ y))) if len(Yana) else 0.0
            idx.append(i)
            YD.append(y)
            OZ.append([float(KY[t] == 0), float(KY[t] == 1),
                       float(KY[t] == 2), float(KY[t] == 3),
                       float(aci0[t]), float(yerel[t]), float(yerel_n[t]),
                       float(parca[t]), sh, sr, ah,
                       float(np.max(np.abs(y))),
                       float(gate_s[i]), float(votes[i]), float(n),
                       float(len(Y))])
    return (np.asarray(idx, int), np.asarray(YD, float).reshape(-1, 3),
            np.asarray(OZ, float).reshape(-1, len(OZ_AD)))


def etiketle(Pp, Dd, G, Gd, yanal=2.0, aci=10.0, eksenel=40.0):
    """Secenek etiketi: bu (konum, yon) cifti BIR GT icin KABUL EDILEBILIR mi?

    Kabul kutusu `sina_kume.esle_macar` ile BIREBIR ayni: yanal <= 2mm,
    |eksenel| <= 40mm, ISARETLI aci <= 10 derece.

    BIRE-BIR KISITI BURADA UYGULANMAZ. Sebep: ayni GT'ye uyan uc gecerli yon
    varsa, Macar atamasi keyfi olarak birini pozitif digerlerini NEGATIF yapar
    ve model "gecerli poz" yerine "atama kurasini" ogrenmeye calisir. Bire-bir
    SECIM aninda (bipartite) uygulanir -- nesne tespitinde standart ayrim.
    """
    Pp = np.asarray(Pp, float).reshape(-1, 3)
    Dd = birim(Dd)
    G = np.asarray(G, float).reshape(-1, 3)
    if not len(Pp) or not len(G):
        return np.zeros(len(Pp), int), np.full(len(Pp), -1, int)
    Gd = birim(Gd)
    diff = Pp[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    yan = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    an = np.degrees(np.arccos(np.clip(Dd @ Gd.T, -1.0, 1.0)))
    kabul = (yan <= yanal) & (np.abs(al) <= eksenel) & (an <= aci)
    y = kabul.any(1).astype(int)
    hangi = np.where(y > 0, np.argmax(kabul, axis=1), -1)
    return y, hangi
