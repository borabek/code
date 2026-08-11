# -*- coding: utf-8 -*-
"""O: UC AYRI KUME kur -- gelistirme / dogrulama / KILITLI.

ZAAFIYET: bugune kadar TEK bir 100-parcalik kume hem KARAR VERMEK (J, L2, esik, ozellik secimi)
hem de OLCMEK icin kullanildi. Bu, secilen her seyin o kumeye bir miktar uydurulmasi demek --
"holdout" adi hak edilmiyor.

COZUM: keskin geometri anahtariyla (bbox 0.5mm + B-rep silindir/duzlem imzasi) UC AYRIK grup:
  DEV    -- kararlar burada verilir (mevcut 100 parcanin devami)
  VAL    -- verilen kararlar burada SINANIR (asiri-uydurma yakalanir, kilitli harcanmaz)
  LOCKED -- tek atislik final. Ustunde HICBIR ayar yapilmaz.

Ek kural: LOCKED, segmentasyon egitim+val geometrileriyle de ayni grupta OLMAZ. Boylece
"gorulmemis geometri" iddiasi ilk kez gercekten dogru olur.

Cikti: results/split3.json
"""
import os, sys, json
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KEYS = "results/_strict_geometry_keys.json"
OUT = "results/split3.json"


def main():
    from big_arbiter import eligible

    keys = json.load(open(KEYS))
    parts = []
    for m, p, jf, s in eligible():
        try:
            n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception:
            continue
        if n > 0 and p in keys:
            parts.append((m, p, jf, s, n))
    seg = set()
    for sp in ("train", "val"):
        f = f"wscad_corpus_scheffler_exact/splits/{sp}.txt"
        if os.path.exists(f):
            seg |= {x.strip() for x in open(f) if x.strip()}
    seg_groups = {keys[p] for p in seg if p in keys}
    print(f"{len(parts)} parca | {len({keys[p[1]] for p in parts})} keskin geometri grubu")
    print(f"segmentasyonun kapladigi grup: {len(seg_groups)}")

    # eski 100'luk kume (kararlar orada verildi) -> DEV'e sabitlenir
    LOCK = set(json.load(open("results/split_lock.json"))["locked_parts"])
    pool = [p for p in parts if p[1] not in LOCK]
    rng = np.random.RandomState(202)
    lo = [x for x in pool if x[4] < 8]; hi = [x for x in pool if x[4] >= 8]
    eski = ([lo[i] for i in rng.choice(len(lo), 70, replace=False)] +
            [hi[i] for i in rng.choice(len(hi), 30, replace=False)])
    dev_groups = {keys[p[1]] for p in eski}
    kirli = sum(1 for p in eski if keys[p[1]] in seg_groups)
    print(f"\nESKI 100'luk kume: {kirli}/100 parca segmentasyon geometrisiyle ayni grupta")
    print("  -> bu kume DEV olur; uzerinde karar verilebilir, MANSET olarak kullanilamaz")

    # kalan gruplar: segmentasyona ve DEV'e degmeyenler
    kalan = [p for p in pool
             if keys[p[1]] not in dev_groups and keys[p[1]] not in seg_groups]
    klo = [x for x in kalan if x[4] < 8]; khi = [x for x in kalan if x[4] >= 8]
    print(f"\ntemiz havuz (ne DEV ne segmentasyon): {len(kalan)} parca "
          f"({len(klo)} dusuk / {len(khi)} cok-CP)")

    # gruplari ikiye bol: VAL ve LOCKED (grup duzeyinde, parca duzeyinde DEGIL)
    grp = sorted({keys[p[1]] for p in kalan})
    r2 = np.random.RandomState(31071)
    r2.shuffle(grp)
    half = len(grp) // 2
    val_g, lock_g = set(grp[:half]), set(grp[half:])

    def pick(groups, nlo, nhi, seed):
        c = [p for p in kalan if keys[p[1]] in groups]
        a = [x for x in c if x[4] < 8]; b = [x for x in c if x[4] >= 8]
        r = np.random.RandomState(seed)
        s = ([a[i] for i in r.choice(len(a), min(nlo, len(a)), replace=False)] +
             [b[i] for i in r.choice(len(b), min(nhi, len(b)), replace=False)])
        return s

    val = pick(val_g, 70, 30, 7)
    lock = pick(lock_g, 70, 30, 11)
    assert not ({keys[p[1]] for p in val} & {keys[p[1]] for p in lock}), "VAL ve LOCKED cakisiyor"
    assert not ({keys[p[1]] for p in lock} & dev_groups), "LOCKED ile DEV cakisiyor"
    assert not ({keys[p[1]] for p in lock} & seg_groups), "LOCKED ile segmentasyon cakisiyor"

    json.dump({
        "created": "2026-07-31",
        "anahtar": "keskin geometri: bbox 0.5mm + B-rep silindir/duzlem imzasi + kose/yuz kovalari",
        "kural": {
            "DEV": "kararlar burada verilir (J, L2, esik, ozellik secimi). MANSET DEGIL.",
            "VAL": "verilen kararlar burada SINANIR. Asiri-uydurma buradan gorulur.",
            "LOCKED": "TEK ATIS. Uzerinde hicbir ayar yapilmaz; gate egitimi bu gruplari gormez.",
        },
        "dev": {"n": len(eski), "parts": [p[1] for p in eski],
                "uyari": f"{kirli}/100 parcasi segmentasyon egitim geometrisiyle ayni grupta"},
        "val": {"n": len(val), "parts": [p[1] for p in val],
                "groups": sorted({keys[p[1]] for p in val})},
        "locked": {"n": len(lock), "parts": [p[1] for p in lock],
                   "groups": sorted({keys[p[1]] for p in lock})},
    }, open(OUT, "w"), indent=1)
    print(f"\nDEV {len(eski)} | VAL {len(val)} | LOCKED {len(lock)}")
    print(f"  VAL gruplari {len({keys[p[1]] for p in val})} | "
          f"LOCKED gruplari {len({keys[p[1]] for p in lock})}")
    print(f"  ucu de AYRIK dogrulandi (assert gecti)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
