# -*- coding: utf-8 -*-
"""P0-5: D7 FINAL SINAVI -- muhurlenir, TUNING BITENE KADAR ACILMAZ.

NEDEN AYRI BIR KUME: D6 (468 parca) uzerinde uc kol ayarlandi (gate esigi, B-rep
oturtma, eksen secici). Ayarlar D6'nin yarisinda secilip diger yarisinda tek atis
olculdu -- yani D6 artik bir DEV kumesidir, manset kumesi degil.

D7 NEDEN GECERLI: D5'in 13 ureticisi (A-B, C3, CCD, CEM, CWT, DEG, DIN, EFX, ELMEX,
KLM, MOR, WEG, WIE) bugun itibariyle
  * gate v5 egitim korpusunda YOK   (uretici duzeyi dislama, 2026-08-05)
  * yeni oto-etiket korpusunda YOK  (958 kirli parca karantinaya alindi)
Yani YENI boru hatti (g6 seg + gate v5) bu ureticilerden TEK parca gormemistir.

SINIRLILIK -- ACIKCA YAZILIYOR: bu ureticiler ESKI g5 aglarinin egitiminde VARDI.
Dolayisiyla D7 ancak parcalar **g6 aglariyla YENIDEN TURETILDIKTEN** sonra
puanlanabilir. Eski turetmelerle puanlamak sonucu sisirir.

UC KATMANLI TEMIZLIK: uretici / geometri anahtari (egitim+olcum+LOCKED ikizleri) /
grup-ici tekillik. MUHURLENIR: pid listesi + SHA256.
"""
import collections
import hashlib
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d6_kayit

CIKTI = "results/d7_sinav_kumesi.json"
GEO_YASAK = "results/_geo_yasak.json"
MIN_URETICI = 5
KOVA = ((1, 3), (4, 7), (8, 10 ** 9))


def kova_adi(n):
    for lo, hi in KOVA:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10**9 else '8+'}"
    return "0"


def main():
    import protokol
    protokol.tez_dogrula()
    from big_arbiter import eligible

    d5 = json.load(io.open("results/d5_4_sinav_kumesi.json", encoding="utf-8"))
    d6 = json.load(io.open("results/d6_sinav_kumesi.json", encoding="utf-8"))
    HEDEF = set(d5["uretici"]) - set(d6["uretici"])      # MOR D6'da, cakismasin
    print(f"D7 hedef ureticileri ({len(HEDEF)}): {sorted(HEDEF)}")

    # DOGRULAMA: bu ureticiler gercekten HICBIR yeni egitim yapitinda olmamali
    v3 = np.load("results/zengin_parite_v3.npz", allow_pickle=True)
    kirli_gate = HEDEF & set(map(str, v3["mfg"]))
    import glob
    pids = [os.path.basename(os.path.normpath(p)) for p in glob.glob("_label_auto_obj/*/")]
    mp = {p: m for m, p, _jf, _s in eligible()}
    kirli_seg = HEDEF & {mp.get(p) for p in pids}
    if kirli_gate or kirli_seg:
        print(f"  DUR: kume KIRLI -- gate {sorted(kirli_gate)} seg {sorted(kirli_seg)}")
        return
    print("  dogrulama: gate v5 ve yeni oto-etiket korpusunda YOK -- TEMIZ")

    E = eligible()
    n_mfg = collections.Counter(m for m, _p, _jf, _s in E)
    HEDEF = {m for m in HEDEF if n_mfg[m] >= MIN_URETICI}
    kayit = d6_kayit.yukle()
    YASAK = set()
    if os.path.exists(GEO_YASAK):
        YASAK = set(json.load(io.open(GEO_YASAK, encoding="utf-8")).values())
    # D6 pid'leri ve anahtarlari da yasak: iki sinav kumesi CAKISMAMALI
    D6P = set(d6["pidler"])
    D6A = {kayit[p].get("geo") for p in D6P if p in kayit and kayit[p].get("geo")}

    secili, gruplar, atlanan = [], set(), collections.Counter()
    for m, p, _jf, _s in sorted(E, key=lambda t: (t[0], t[1])):
        if m not in HEDEF:
            continue
        r = kayit.get(p)
        if r is None:
            atlanan["turetme_yok"] += 1; continue
        _g = r.get("G")
        G = np.asarray([] if _g is None else _g, float)
        if not len(G):
            atlanan["gt_yok"] += 1; continue
        anah = r.get("geo")
        if p in D6P or (anah and anah in D6A):
            atlanan["d6_cakismasi"] += 1; continue
        if anah and anah in YASAK:
            atlanan["yasak_anahtar"] += 1; continue
        if anah and anah in gruplar:
            atlanan["grup_ici_tekrar"] += 1; continue
        if anah:
            gruplar.add(anah)
        secili.append({"pid": p, "mfg": m, "cp": int(len(G)), "kova": kova_adi(len(G))})
    print(f"ATLANAN: {dict(atlanan)}")

    up = collections.Counter(s["mfg"] for s in secili)
    kv = collections.Counter(s["kova"] for s in secili)
    pidler = sorted(s["pid"] for s in secili)
    muhur = hashlib.sha256("\n".join(pidler).encode()).hexdigest()[:16]
    out = {"n_parca": len(secili), "sha16": muhur, "uretici": dict(up),
           "cp_kovasi": dict(kv), "gt_toplam": sum(s["cp"] for s in secili),
           "pidler": pidler, "parcalar": secili, "min_uretici": MIN_URETICI,
           "durum": "MUHURLU -- ACILMADI",
           "kosul": ("Puanlama YALNIZ g6 aglariyla YENIDEN TURETILDIKTEN sonra gecerlidir. "
                     "Bu ureticiler ESKI g5 aglarinin egitiminde VARDI."),
           "not": ("D7 FINAL sinavi. D6 artik DEV'dir (uzerinde uc kol ayarlandi). "
                   "D7 gorulduKten sonra tuning yapilirsa D7 de DEV olur ve D8 gerekir.")}
    with io.open(CIKTI, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nD7: {len(secili)} parca | {len(up)} uretici | GT {out['gt_toplam']} CP "
          f"| muhur {muhur}")
    print(f"  uretici: {dict(sorted(up.items(), key=lambda kv: -kv[1]))}")
    print(f"  CP kovasi: {dict(kv)}")
    print(f"-> {CIKTI}   [MUHURLU]")


if __name__ == "__main__":
    main()
