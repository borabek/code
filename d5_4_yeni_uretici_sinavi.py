# -*- coding: utf-8 -*-
"""D5-4: GORULMEMIS URETICI SINAV KUMESI -- urunun ASIL sorusunu olcen kume.

NEDEN: olcum kumesi 194 parca ve dagilimi **WEI 103 / PXC 90 / WAGO 1**. Yani "gorulmemis
bir WSCAD parcasinda CP bul" hedefi ([[robot-goal-and-training-sources]]) bu kumeyle
ciddi olcekte SINANAMIYOR. [[gate-uretici-disi-cokusu]]: gate gorulmemis ureticide
0.7422 -> 0.2799 cokuyordu; korpus 22 ureticiye ciktigi icin bu ilk kez olculebilir.

194'LUK KUME DONDURULMUS KALIR -- karsilastirilabilirlik icin. Bu AYRI bir kumedir.

UC KATMANLI TEMIZLIK (sinav kumesi kirlenirse hicbir sey ifade etmez):
  1. URETICI: yalnizca egitim korpusunda OLMAYAN ureticiler
  2. GEOMETRI GRUBU: adayin anahtari egitim korpusunun, olcum kumesinin ya da LOCKED'in
     HICBIR anahtariyla carpismayacak ([[geometry-twin-leakage]]: parcalarin %80'inin
     ikizi var; parca duzeyi ayrim YETMEZ)
  3. GRUP-ICI TEKILLIK: ayni geometri grubundan yalniz BIR parca alinir, yoksa kume
     kendi icinde tekrar eder ve etkin N sisikmis gorunur

DENGE: uretici x CP kovasi (1-3 / 4-7 / 8+). Kova bilgisi GT'den gelir, tahminden degil.

MUHURLENIR: pid listesi + SHA256 yazilir. Sonradan "kume degisti mi" sorusu yanitlanabilir.
Bu kume BIR KEZ kurulur; her kol onu AYNI haliyle kullanir.

TEZ DEGISMEZ: hicbir sey egitilmez; bu bir SECIM betigidir.
"""
import collections
import hashlib
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YENI = "results/_der_yeni.pkl"
GEO_YASAK = "results/_geo_yasak.json"
# DIKKAT -- DAIRE TUZAGI (2026-08-04, kendi hatam): ilk surum burada `zengin_parite_v3.npz`
# kullaniyordu. Ama v3 YENI parcalari ZATEN iceriyor, dolayisiyla "egitimde olmayan uretici"
# kumesi bosaliyordu (20 uretici -> 2 parca -> 0). Sinav kumesi ONCE tanimlanir, egitim
# SONRA onu disarida birakir. Referans bu yuzden ORIJINAL korpus olmali.
EGITIM = "results/zengin_parite_w2.npz"
CIKTI = "results/d5_4_sinav_kumesi.json"
DER_CIKTI = "results/_der_sinav_yeni.pkl"

# CERCEVE ESIGI: parcanin GT noktalarinin en yakin adaya ORTANCA yanal mesafesi.
# 15mm ustu = hizalama tamamen kaymis. 5-15mm supheli ama SINAVDA KALIR (o parcalar
# gercek zorluk da olabilir); yalniz acik bozukluk atilir. Esik gevsek TUTULDU ki
# "sonucu guzellestirmek icin zor parcalari atmis" olmayalim.
CERCEVE_ESIK = 15.0

KOVA = ((1, 3), (4, 7), (8, 10 ** 9))


def kova_adi(n):
    for lo, hi in KOVA:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10**9 else '8+'}"
    return "0"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinir", type=int, default=250,
                    help="sinav kumesi ust siniri (dengeli ornekleme). 0 = sinirsiz")
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()
    import olcum_kumesi as OK

    with open(YENI, "rb") as f:
        R = pickle.load(f)
    print(f"yeni turetme: {len(R)} parca")

    # --- KATMAN 1: egitim korpusunda OLMAYAN ureticiler
    d = np.load(EGITIM, allow_pickle=True)
    egitim_mfg = set(map(str, d["mfg"]))
    egitim_pid = set(map(str, d["pids"]))
    aday = [r for r in R if r["mfg"] not in egitim_mfg]
    print(f"KATMAN 1 uretici: egitimde {len(egitim_mfg)} uretici var "
          f"({sorted(egitim_mfg)}) -> {len(aday)} parca kaldi")

    # --- KATMAN 2: geometri grubu carpismasi (yasak anahtarlar + egitim anahtarlari)
    YASAK = set(json.load(io.open(GEO_YASAK, encoding="utf-8")).values())
    egitim_anah = {r["geo"] for r in R if r["pid"] in egitim_pid and r.get("geo")}
    gk = OK.geo_anahtarlari()
    egitim_anah |= {gk[p] for p in egitim_pid if p in gk}
    once = len(aday)
    aday = [r for r in aday
            if r.get("geo") and r["geo"] not in YASAK and r["geo"] not in egitim_anah]
    print(f"KATMAN 2 geometri: {once} -> {len(aday)} parca "
          f"({once - len(aday)} carpisma atildi)")

    # --- KATMAN 3: grup-ici tekillik (ayni gruptan yalniz BIR parca)
    gor, tekil = set(), []
    for r in sorted(aday, key=lambda x: (x["mfg"], x["pid"])):
        if r["geo"] in gor:
            continue
        gor.add(r["geo"]); tekil.append(r)
    print(f"KATMAN 3 tekillik: {len(aday)} -> {len(tekil)} parca "
          f"({len(aday) - len(tekil)} ikiz atildi)")

    # --- GT'si olmayanlari at (sinav icin etiket sart)
    tekil = [r for r in tekil if int(r["n"]) > 0 and r.get("X") is not None]
    print(f"GT'si + adayi olan: {len(tekil)} parca")

    # --- KATMAN 4: CERCEVE SAGLIGI (2026-08-04, ZORUNLU DUZELTME)
    #
    # BULGU: AL ureticisinde canli urun de, TABAN da, v3 de TAM 0.000 aldi. Sebep model
    # DEGIL: AL'in 81 GT noktasinin EN YAKIN adaya yanal mesafesi ortanca **49.92 mm**
    # (min 35.68), 3mm icinde SIFIR tane. Eksenel mesafe her parcada TAM 0.00.
    # Yani GT YANLIS YERDE -- `cad_eval.align_frames(Vr, Vj)` bu ureticilerde JSON'un
    # Graphic3d noktalarini STEP mesh'ine hizalayamiyor.
    #
    # OLCULDU (sinav kumesi geneli): GT noktalarinin **%12.5'i** cerceve-bozuk
    #   AL 49.92mm · WEG 18.86mm · TOGI 17.04mm · ABB 15.76mm   -> BOZUK
    #   WIE 13.60 · CWT 8.62 · DIN 6.65 · KLM 6.36 · EFX 5.49   -> SUPHELI
    #   ELMEX 0.46 · C3 1.78 · A-B 2.33 · DEG 2.49 · CCD 2.71   -> SAGLAM
    # F1 bu siralamayla neredeyse BIREBIR ortusuyor (ELMEX 0.607 ... AL 0.000).
    #
    # NEDEN FILTRE SART: cerceve bozuksa olculen sey URUNUN BASARISI DEGIL, hizalamanin
    # basarisizligidir. Bu parcalar sinavda kalirsa "gorulmemis ureticide cokuyor"
    # sonucu SAHTE olarak agirlasir.
    # NOT: bu parcalar KAYBEDILMIYOR -- hizalama duzeltilince geri alinabilirler.
    once = len(tekil)
    saglam = []
    for r in tekil:
        P = np.asarray(r["P"], float); G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        if not len(P) or not len(G):
            continue
        dd = P[:, None, :] - G[None, :, :]
        aa = (dd * Gd[None, :, :]).sum(-1)
        yanal = np.linalg.norm(dd - aa[..., None] * Gd[None, :, :], axis=-1).min(0)
        if float(np.median(yanal)) <= CERCEVE_ESIK:
            saglam.append(r)
    tekil = saglam
    print(f"KATMAN 4 cerceve: {once} -> {len(tekil)} parca "
          f"({once - len(tekil)} hizalamasi BOZUK, esik {CERCEVE_ESIK}mm)")

    # --- SINIRLAMA: DENGELI ORNEKLEME (2026-08-04'te eklendi)
    #
    # NEDEN: sinirsiz halde kume 514 parca cikti ve EGITIM CESITLILIGINI YIYORDU --
    # 514 sinav + 393 ikizi = 1476 yeni parcanin **%61'i** egitim disi kaliyor, v3 20
    # ureticiden 8'e dusuyordu. Sinav kumesinin buyuk olmasi gerekmiyor; TEMSILI ve
    # DENGELI olmasi gerekiyor. Uretici x CP-kovasi dongusel secimle kuculturuz:
    # her turda her ureticiden, o ureticinin en az temsil edilen kovasindan bir parca.
    # Deterministik (pid'e gore sirali), yani muhur tekrarlanabilir.
    if a.sinir and len(tekil) > a.sinir:
        havuz = collections.defaultdict(list)
        for r in sorted(tekil, key=lambda x: x["pid"]):
            havuz[(r["mfg"], kova_adi(int(r["n"])))].append(r)
        mfgler = sorted({k[0] for k in havuz})
        kovalar = [f"{lo}-{hi if hi < 10**9 else '8+'}" for lo, hi in KOVA]
        sec, i = [], 0
        while len(sec) < a.sinir:
            eklendi = False
            for m in mfgler:
                for kv_ in kovalar:
                    L = havuz[(m, kv_)]
                    if i < len(L) and len(sec) < a.sinir:
                        sec.append(L[i]); eklendi = True
            if not eklendi:
                break
            i += 1
        print(f"\nSINIRLAMA: {len(tekil)} -> {len(sec)} parca "
              f"(uretici x CP-kovasi dongusel, deterministik)")
        tekil = sec

    kv = collections.Counter(kova_adi(int(r["n"])) for r in tekil)
    mf = collections.Counter(r["mfg"] for r in tekil)
    print(f"\nCP kovasi: {dict(kv)}")
    print(f"uretici  : {dict(mf.most_common())}")

    pid = sorted(r["pid"] for r in tekil)
    imza = hashlib.sha256("|".join(pid).encode()).hexdigest()[:16]
    with open(DER_CIKTI, "wb") as f:
        pickle.dump(tekil, f)
    with io.open(CIKTI, "w", encoding="utf-8") as f:
        json.dump({"n_parca": len(pid), "sha16": imza,
                   "uretici": dict(mf), "cp_kovasi": dict(kv),
                   "gt_toplam": int(sum(r["n"] for r in tekil)),
                   "aday_toplam": int(sum(len(r["P"]) for r in tekil)),
                   "pidler": pid,
                   "not": "GORULMEMIS URETICI sinavi. 194'luk olcum kumesi AYRI ve "
                          "DONDURULMUS kalir. Bu kume egitimde ASLA kullanilmaz."},
                  f, indent=1, ensure_ascii=False)
    print(f"\nSINAV KUMESI: {len(pid)} parca / {len(mf)} uretici | "
          f"GT {sum(r['n'] for r in tekil)} | aday {sum(len(r['P']) for r in tekil)}")
    print(f"  MUHUR sha16 = {imza}")
    print(f"  -> {CIKTI}  ve  {DER_CIKTI}")
    print("\n  UYARI: bu kume EGITIMDE kullanilamaz. Her kol onu AYNI haliyle kullanir;")
    print("         korpus buyudukce YENIDEN URETILIRSE muhur degisir ve karsilastirma bozulur.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
