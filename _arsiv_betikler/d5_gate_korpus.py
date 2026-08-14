# -*- coding: utf-8 -*-
"""D5-3b: GATE EGITIM KORPUSU v3 -- eski korpus + yeni turetme, SIZINTI FILTRELI.

YENIDEN TURETME YOK: `d5_turet_yeni.py` her parcanin `X` (22 sutun) ve `XR` (36 sutun)
matrislerini zaten sakladi; `d5_birlestir.py` de X'i 58 -> 22'ye normalize etti (kayipsiz
oldugu 984/984 kayitta olculdu). Burada yalnizca ETIKET uretilir ve iki korpus birlestirilir.

ETIKET (`y`) `build_zengin_parite.py` ile BIREBIR AYNI kuralla uretilir:
  tol = max(3.0, 0.06*diag) | eksenel |al| <= 40 | TEKIL greedy esleme
Baska bir kural kullanilirsa yeni satirlar eskilerden FARKLI bir hedefe egitilir ve gate
sessizce bozulur -- bu yuzden kural burada TEK YERDE ve yorumlu duruyor.

UC KATMANLI SIZINTI FILTRESI:
  1. `protokol.dogrula()`        -- LOCKED + olcum gruplari (ESKI anahtar uzayi)
  2. YASAK ANAHTAR capraz kontrolu -- yeni parcalarin anahtarlari `_geo_yasak.json` ile
     karsilastirilir. **Bu sart cunku iki anahtar uzayi string olarak ASLA eslesmez**:
     eski `_strict_geometry_keys.json` bicimi ile `geometri_anahtar.anahtar()` bicimi
     farkli. 2026-08-04 olcumu: **126 yeni parca** yasak anahtarla carpisiyor -- yani
     olcum/LOCKED parcalarinin IKIZLERI. Bu filtre olmadan egitime girip mansedi
     sessizce sisirirlerdi.
  3. Aday uretilememis parcalar (X is None) atlanir.

TEZ DEGISMEZ: hicbir ag egitilmez, hicbir esik degismez, `v_o` turetmesi aynen kalir.
Degisen TEK sey korpus buyuklugu ve uretici cesitliligi -- F2-12'nin sarti budur.

!!! F2-12 ICIN KRITIK UYARI -- FILTRE KARISTIRMASI !!!
Bu betik protokol bekcisini `olcum_da=True` ile uygular ve ESKI korpustan da parca cikarir
(olculdu: 7983 aday / 519 parca). Yani v3, w2'nin "buyutulmus" hali DEGIL; **farkli
filtreden gecmis** bir korpus. Dagitilan gate w2 ile egitildiyse, v3'le egitilmis bir
gate'i onunla kiyaslamak "veri artti"yi DEGIL "filtre degisti"yi olcer.

F2-12 DOGRU TASARIM: IKI gate AYNI filtreyle egitilir --
    (a) yalniz ESKI parcalar + ayni protokol filtresi
    (b) eski + YENI parcalar + ayni protokol filtresi
Fark ancak o zaman SADECE VERIDIR. `--yalniz-eski` bayragi (a)'yi uretir.
[[gate-refit-minv4]] dersi birebir bu: iki gate AYNI dagilimda egitilmeli.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ESKI = "results/zengin_parite_w2.npz"
YENI = os.environ.get("D5_YENI", "results/_der_yeni.pkl")
GEO_YENI = "results/_geo_yeni.json"
GEO_YASAK = "results/_geo_yasak.json"
SINAV = "results/d5_4_sinav_kumesi.json"   # D5-4 gorulmemis uretici sinavi -- EGITIME GIRMEZ
# CIKTI da ORTAM DEGISKENI (2026-08-08): g10 ile yeniden turetilen korpus AYRI bir
# dosyaya yazilmali. Sabit kalsaydi `zengin_parite_v3` -- yani DAGITILAN gate'in
# egitildigi korpus -- uzerine yazilir ve A/B kiyasi imkansizlasirdi.
CIKTI = os.environ.get("D5_CIKTI", "results/zengin_parite_v3.npz")
CIKTI_TABAN = CIKTI.replace(".npz", "_taban.npz")
MAKBUZ = "results/d5_gate_korpus.json"

VOTES_SUTUN = 11        # X22 icinde `votes` sutunu -- %100 dogrulandi (X22[:,11] == votes)


def etiketle(P, G, Gd, diag):
    """build_zengin_parite ile BIREBIR AYNI tekil esleme -> aday basina 0/1."""
    n = len(P)
    y = np.zeros(n, int)
    if not n or not len(G):
        return y
    tol = max(3.0, 0.06 * float(diag))
    d = P[:, None, :] - G[None, :, :]
    al = (d * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(d - al[..., None] * Gd[None, :, :], axis=-1)
    pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
    up, ug = set(), set()
    for dd, a_, b_ in sorted((pe[a, b], a, b)
                             for a in range(n) for b in range(len(G))):
        if dd > tol or a_ in up or b_ in ug:
            continue
        up.add(a_); ug.add(b_); y[a_] = 1
    return y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # F2-12'nin (a) kolu: yeni parcalari EKLEMEDEN, AYNI filtreyle taban korpus uret.
    # Boylece (a) ile (b) arasindaki TEK fark VERI olur (bkz. yukaridaki kritik uyari).
    ap.add_argument("--yalniz-eski", action="store_true",
                    help="yeni parcalari EKLEME -- F2-12 taban kolu")
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()

    d = np.load(ESKI, allow_pickle=True)
    X22 = [d["X22"]]; XR = [d["XR"]]; Y = [d["y"]]
    PID = [np.asarray(d["pids"], str)]; MFG = [np.asarray(d["mfg"], str)]
    VOT = [d["votes"]]
    eski_parca = len(set(map(str, d["pids"])))
    print(f"ESKI korpus: {len(d['y'])} aday / {eski_parca} parca")

    if a.yalniz_eski:
        R = []
        print("YALNIZ ESKI: yeni parcalar eklenmiyor (F2-12 taban kolu)")
    else:
        with open(YENI, "rb") as f:
            R = pickle.load(f)
    GY = json.load(io.open(GEO_YENI, encoding="utf-8")) if os.path.exists(GEO_YENI) else {}
    YASAK = set(json.load(io.open(GEO_YASAK, encoding="utf-8")).values())
    # D5-4 SINAV KUMESI EGITIME GIREMEZ. Kume anahtarlari da yasak listesine eklenir --
    # parca kimligi yetmez, IKIZI de girmemeli ([[geometry-twin-leakage]]).
    SINAV_PID, SINAV_MFG = set(), set()
    # IKI SINAV KUMESI birden dislanir (2026-08-05): eski d5_4 kumesi karsilastirilabilirlik
    # icin duruyor, YENI d6 kumesi ise gercek gorulmemis-uretici sinavi. Biri atlanirsa o
    # kumede olculen her sayi sisik olur.
    for _sf in (SINAV, "results/d6_sinav_kumesi.json"):
        if not os.path.exists(_sf):
            continue
        _sv = json.load(io.open(_sf, encoding="utf-8"))
        SINAV_PID |= set(_sv.get("pidler") or [])
        SINAV_MFG |= set(_sv.get("uretici") or {})
        print(f"  SINAV KUMESI {_sf}: {len(_sv.get('pidler') or [])} parca, "
              f"{len(_sv.get('uretici') or {})} uretici (muhur {_sv.get('sha16')})")
    if os.path.exists(SINAV):
        sv = json.load(io.open(SINAV, encoding="utf-8"))
        # DIKKAT -- burada bir zamanlar `SINAV_MFG = set(...)` yaziyordu ve yukaridaki
        # IKI-KUMELI birlesimi EZIYORDU: d6'nin 8 ureticisi sessizce egitime giriyordu
        # (belirti: korpusta SE 74 / UTL 36 / SUPU 24 aday). Atama DEGIL, BIRLESIM.
        #
        # URETICI DUZEYI DISLAMA (2026-08-05'te BULUNAN KIRLILIK):
        # Dislama yalniz PARCA + IKIZ duzeyindeydi. Sinav kumesi "gorulmemis URETICI"
        # sinavi olarak tanimlandigi halde, korpus buyudukce ayni ureticilerin BASKA
        # parcalari egitime giriyordu. Olculdu: `zengin_parite_v3` sinav ureticilerinden
        # **1502 tekil parca** iceriyordu (CWT 691, A-B 358, WIE 226, ...). Sinav pid'leri
        # temizdi, ama URETICI artik gorulmemis DEGILDI -- yani kumenin ADI yalan olmustu.
        # ([[locked-sinav-kirliligi-yakalandi]] ile ayni desen.)
        SINAV_MFG |= set(sv.get("uretici") or {})
        print(f"SINAV (iki kume): {len(SINAV_PID)} parca egitim disi")
        print(f"  URETICI DUZEYI dislama ({len(SINAV_MFG)}): {sorted(SINAV_MFG)}")
    print(f"YENI turetme: {len(R)} parca | yasak anahtar {len(YASAK)}")

    SINAV_ANAH = {r["geo"] for r in R if r["pid"] in SINAV_PID and r.get("geo")}
    atlanan = collections.Counter()
    n_yeni = 0
    for r in R:
        if r.get("X") is None or r.get("XR") is None:
            atlanan["aday_yok"] += 1; continue
        if r["X"].shape[1] != 22 or r["XR"].shape[1] != 36:
            atlanan["genislik"] += 1; continue
        # GT GECERLILIK (2026-08-04): `eligible()` kapisi turetmeden SONRA eklendi, yani
        # `_der_yeni.pkl` hala dejenere GT'li parcalari iceriyor (AL vakasi: yon (0,0,0),
        # tum CP ayni noktada). Onlarda HICBIR aday eslesemez -> hepsi y=0 ile egitime
        # girer ve gate'e "gecerli agizlari REDDET" ogretir. Burada da kapatilir.
        Gq = np.asarray(r["G"], float); Dq = np.asarray(r["Gd"], float)
        if (len(Dq) and (np.linalg.norm(Dq, axis=1) < 1e-6).all()) or            (len(Gq) > 1 and float(np.abs(Gq.max(0) - Gq.min(0)).max()) < 1e-6):
            atlanan["gt_dejenere"] += 1; continue
        if r["pid"] in SINAV_PID:
            atlanan["sinav_kumesi"] += 1; continue
        if str(r["mfg"]) in SINAV_MFG:
            atlanan["sinav_ureticisi"] += 1; continue
        anah = r.get("geo") or GY.get(r["pid"])
        if anah in SINAV_ANAH:
            atlanan["sinav_ikizi"] += 1; continue
        if anah in YASAK:
            # OLCUM ya da LOCKED parcasinin IKIZI -- egitime GIREMEZ
            atlanan["yasak_anahtar"] += 1; continue
        P = np.asarray(r["P"], float)
        y = etiketle(P, np.asarray(r["G"], float), np.asarray(r["Gd"], float), r["diag"])
        X22.append(r["X"]); XR.append(r["XR"]); Y.append(y)
        PID.append(np.array([str(r["pid"])] * len(P)))
        MFG.append(np.array([str(r["mfg"])] * len(P)))
        VOT.append(np.asarray(r["X"], float)[:, VOTES_SUTUN])
        n_yeni += 1

    print(f"\nATLANAN: {dict(atlanan)}")
    print(f"  -> yasak_anahtar = olcum/LOCKED IKIZI, egitime ALINMADI")

    X22 = np.vstack(X22); XR = np.vstack(XR); Y = np.concatenate(Y)
    PID = np.concatenate(PID); MFG = np.concatenate(MFG); VOT = np.concatenate(VOT)

    # KATMAN 1: eski anahtar uzayinda protokol bekcisi (LOCKED + olcum gruplari)
    m = protokol.dogrula(PID.tolist(), ad="gate_korpus_v3", olcum_da=True, sert=False)
    if (~m).any():
        atl = len(set(PID[~m].tolist()))
        print(f"  protokol bekcisi: {int((~m).sum())} aday / {atl} parca CIKARILDI")
        X22, XR, Y, PID, MFG, VOT = (a[m] for a in (X22, XR, Y, PID, MFG, VOT))

    # CIKTI ve MAKBUZ IKISI DE MODA BAGLI. Ilk surumde makbuz sabitti ve taban kosusu
    # v3'un makbuzunun UZERINE yaziyordu; ekrana da yanlis dosya adi basiyordu. Iki kolun
    # makbuzu karisirsa F2-12 karsilastirmasi sessizce yanlis sayilara dayanir.
    out = CIKTI_TABAN if a.yalniz_eski else CIKTI
    mak = MAKBUZ.replace(".json", "_taban.json") if a.yalniz_eski else MAKBUZ
    np.savez(out, X22=X22, XR=XR, y=Y, pids=PID, mfg=MFG, votes=VOT,
             zengin_ad=d["zengin_ad"])
    parca = len(set(PID.tolist()))
    c = collections.Counter(MFG.tolist())
    print(f"\nKORPUS {'TABAN (yalniz eski)' if a.yalniz_eski else 'v3'} -> {out}")
    # ORAN HANGI TABANA GORE (2026-08-04'te duzeltildi): ilk surum v3'u FILTRELENMEMIS
    # w2 (1709 parca) ile kiyasliyordu ve "x1.73" yaziyordu. Ama F2-12'nin kontrol kolu
    # TABAN korpusudur (ayni protokol filtresinden gecmis, 1190 parca) -- dogru oran
    # ona gore x2.48. Iki farkli tabana gore iki farkli sayi yazmak, kolun buyuklugunu
    # oldugundan KUCUK gosteriyordu.
    tb = 0
    if os.path.exists(CIKTI_TABAN):
        tb = len(set(np.load(CIKTI_TABAN, allow_pickle=True)["pids"].tolist()))
    print(f"  {len(Y)} aday / {parca} parca / {len(c)} uretici")
    print(f"  buyume: filtresiz w2 ({eski_parca}) -> x{parca/max(eski_parca,1):.2f}  |  "
          f"TABAN ({tb}) -> x{parca/max(tb,1):.2f}   <- F2-12 icin GECERLI olan bu")
    print(f"  pozitif %{100*Y.mean():.1f} | X22 {X22.shape} | XR {XR.shape}")
    print(f"  uretici: {dict(c.most_common(10))}")
    with io.open(mak, "w", encoding="utf-8") as f:
        json.dump({"aday": int(len(Y)), "parca": int(parca), "uretici": len(c),
                   "eski_parca": eski_parca, "yeni_parca_eklendi": n_yeni,
                   "atlanan": dict(atlanan), "pozitif_oran": float(Y.mean()),
                   "uretici_dagilimi": dict(c)}, f, indent=1, ensure_ascii=False)
    print(f"  makbuz -> {mak}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
