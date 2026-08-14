# -*- coding: utf-8 -*-
"""YON ODUNC ALMA: konum `v_o`'da KALIR, yalnizca YON secilir.

NEDEN AYRI BIR KOL: bugune kadar denenen bes mimari ADAY EKLIYORDU ve hepsi
kesinligi seyreltip tabanin altinda kaldi
([[brep-havuz-tavan-acildi-secici-acamadi]]). Bu kol **aday sayisini
DEGISTIRMEZ** -- her adayin yalnizca yonunu degistirebilir. Yapisal olarak
farklidir: FP sayisi artamaz, yalnizca var olan bir aday robot-hazir hale gelir.

OLCULDU (`results/konum_yon_{val,d7}.json`, mukemmel secici tavani, TEZ-SAF havuz):
  VAL  robot F1 tavani 0.5928 -> 0.7008   (yalniz yon odunc alarak)
  D7   robot F1 tavani 0.3412 -> 0.4401
Kaynak kirilimi (recall): komsu aday yonu VAL +0.027 / D7 +0.030,
B-rep silindir ekseni VAL +0.086 / D7 +0.040, baskin yon +0.005.

TEZE SADIK: `v_o` agiz-ortasi KONUMU, 5 sinif segmentasyon ve ~6000 remesh
DEGISMEZ. Tezin kendi yonu HER ZAMAN 0 numarali secenektir ve gercek fallback'tir:
hicbir secenek daha iyi gorunmezse aday tezin yonuyle cikar.
"""
import numpy as np

KOMSU_R = 10.0        # komsu adayin yonunu odunc alma yaricapi (mm)
EKSEN_R = 10.0        # B-rep silindir eksenini odunc alma yaricapi (mm)
AYIRT_ACI = 5.0       # bu aciya daha yakin secenekler AYNI sayilir (tekrar yok)

OZ_AD = ["mevcut", "aci_mevcuda", "kaynak_komsu", "kaynak_eksen", "kaynak_baskin",
         "mesafe", "mesafe_diag", "aci_baskina", "eksen_hizasi", "gate_skoru",
         "n_aday", "destek"]


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def _aci(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def baskin_yon(D):
    """Parcanin isaret-hizali baskin yonu (tum adaylarin ortalamasi)."""
    D = _birim(D)
    if len(D) == 0:
        return None
    if len(D) == 1:
        return D[0]
    B = D * np.sign(D @ D[0])[:, None]
    return _birim([B.mean(0)])[0]


def secenekler(P, D, i, cyl, gate_s, bask, komsu_r=KOMSU_R, eksen_r=EKSEN_R):
    """`i` numarali aday icin (yon, oznitelik) listesi. Ilk oge HER ZAMAN MEVCUT.

    Konum DEGISMEZ -- doner degerde konum yok, yalnizca yon.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    D = _birim(D)
    p, d = P[i], D[i]
    n = len(P)
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) if n > 1 else 1.0

    adaylar = [(d, "mevcut", 0.0)]
    if n > 1:
        uz = np.linalg.norm(P - p, axis=1)
        for j in np.argsort(uz):
            if j == i or uz[j] > komsu_r:
                continue
            adaylar.append((D[j], "komsu", float(uz[j])))
    for c in cyl or []:
        a = np.asarray(c["axis"], float)
        na = np.linalg.norm(a)
        if na < 1e-9:
            continue
        a = a / na
        m = np.asarray(c.get("center", p), float)
        u = float(np.linalg.norm(m - p))
        if u <= eksen_r:
            adaylar.append((a, "eksen", u))
            adaylar.append((-a, "eksen", u))
    if bask is not None:
        adaylar.append((bask, "baskin", 0.0))

    # TEKRAR AYIKLAMA: birbirine `AYIRT_ACI`'dan yakin yonler AYNI secenektir.
    # Mevcut HER ZAMAN korunur (ilk sirada oldugu icin dogal olarak kazanir).
    secili, oz = [], []
    for v, kaynak, mes in adaylar:
        v = _birim([v])[0]
        if any(_aci(v, w) < AYIRT_ACI for w, _, _ in secili):
            continue
        secili.append((v, kaynak, mes))
    for v, kaynak, mes in secili:
        destek = int(sum(1 for w in D if _aci(v, w) < AYIRT_ACI))
        oz.append([
            float(kaynak == "mevcut"),
            _aci(v, d),
            float(kaynak == "komsu"),
            float(kaynak == "eksen"),
            float(kaynak == "baskin"),
            mes,
            mes / max(diag, 1e-6),
            _aci(v, bask) if bask is not None else 0.0,
            float(np.max(np.abs(v))),
            float(gate_s),
            float(n),
            float(destek),
        ])
    return [v for v, _, _ in secili], np.asarray(oz, float)


def uygula(P, D, cyl, gate_skorlari, puanla):
    """Her aday icin en iyi yonu sec. Doner: yeni D (konum DOKUNULMAZ).

    `puanla(X)` -> her secenek icin skor. MEVCUT (0. secenek) esitlikte KAZANIR:
    yeni yon ancak KESIN daha iyiyse alinir, boylece tezin cevabi varsayilan kalir.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    D = _birim(D)
    if len(P) == 0:
        return D
    bask = baskin_yon(D)
    yeni = D.copy()
    for i in range(len(P)):
        V, X = secenekler(P, D, i, cyl, float(gate_skorlari[i]), bask)
        if len(V) < 2:
            continue
        s = np.asarray(puanla(X), float)
        j = int(np.argmax(s))
        if j != 0 and s[j] > s[0]:
            yeni[i] = V[j]
    return yeni
