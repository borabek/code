# -*- coding: utf-8 -*-
"""KORPUS TEMIZLIGI: eslesmeyen dosyalar ve kopyalar (SILMEZ -- ARSIVE TASIR).

UC BULGU (2026-08-04 butunluk denetimi):
  1. STEP'i OLMAYAN JSON      367 dosya (321'i ABB = kapsam disi, yalniz 16'si klemens)
  2. JSON'u OLMAYAN STEP     2699 kimlik / 2843 dosya -- ESKI spekulatif indirmeler,
     bugunku entegrasyondan DEGIL. `eligible()` JSON uzerinden dondugu icin ZARARSIZ,
     ayrica bazilari gorsellestirmede kullanildi (0270018 gibi). SILINMEZ, raporlanir.
  3. KOPYA STEP               164 kimlik -- hepsi 2026-07-13 tarihli, saniyeler arayla
     cift indirilmis, iceriKleri **md5 duzeyinde BIREBIR AYNI**. Dogruluk riski YOK
     (hangisi secilirse secilsin ayni dosya) ama israf; fazlasi arsive alinir.

SILME YOK: her sey `_arsiv/` altina TASINIR. Geri almak icin tersine kopyalamak yeter.
OLCUM KUMESI KORUMASI: 194 olcum parcasindan HICBIRI tasinamaz (assert ile korunur).
"""
import io
import json
import os
import shutil
import sys
import glob
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")
from korpus_kimlik import kimlik, step_kimlik

ARSIV_JSON = "_arsiv/json_stepsiz"
ARSIV_STEP = "_arsiv/step_kopya"
ARSIV_OKSUZ = "_arsiv/step_oksuz"     # JSON'u olmayan STEP'ler (2026-08-04)


def main():
    from big_arbiter import eligible as _eligible
    import olcum_kumesi as OK
    D, _ = OK.kume("results/_der_tam.pkl")
    korunan = {r["pid"] for r in D}
    print(f"KORUNAN (olcum kumesi): {len(korunan)} parca -- bunlar ASLA tasinmaz")

    js = glob.glob("_ds1/DataSet/*.json")
    st = glob.glob("all_wscad_stp/*.stp")
    sk = collections.defaultdict(list)
    for s in st:
        sk[step_kimlik(s)].append(s)
    jk = collections.defaultdict(list)
    for f in js:
        m, p = kimlik(os.path.basename(f))
        if p:
            jk[p].append(f)

    # --- 1) STEP'i olmayan JSON -> arsiv
    os.makedirs(ARSIV_JSON, exist_ok=True)
    n1 = 0
    tasinan_uretici = collections.Counter()
    for p, fs in jk.items():
        if p in sk:
            continue
        assert p not in korunan, f"OLCUM PARCASI tasinmak uzereydi: {p}"
        for f in fs:
            m, _ = kimlik(os.path.basename(f))
            tasinan_uretici[m] += 1
            shutil.move(f, os.path.join(ARSIV_JSON, os.path.basename(f)))
            n1 += 1
    print(f"\n1) STEP'i olmayan JSON -> {ARSIV_JSON}: {n1} dosya")
    print("   uretici:", dict(tasinan_uretici.most_common(8)))

    # --- 3) KOPYA STEP -> arsiv (en eskisi kalir)
    os.makedirs(ARSIV_STEP, exist_ok=True)
    n3 = 0
    for k, v in sk.items():
        if len(v) < 2:
            continue
        v = sorted(v)
        for x in v[1:]:
            shutil.move(x, os.path.join(ARSIV_STEP, os.path.basename(x)))
            n3 += 1
    print(f"3) KOPYA STEP -> {ARSIV_STEP}: {n3} dosya (her kimlikten biri KALDI)")

    # --- 2) OKSUZ STEP -> ARSIV (2026-08-04'te RAPOR'dan TASIMA'ya cevrildi)
    #
    # ONCE yalniz raporlaniyordu. Kullanici kaldirilmasini istedi; F1 etkisi YOK cunku
    # `eligible()` JSON'lar uzerinden doner ve oksuz kimlige HIC bakmaz -- yani bu dosyalar
    # ne egitime ne olcume giriyor, sadece 1.84 GB disk tutuyorlar.
    # SILINMIYOR, arsive tasiniyor: bir kismi gorsellestirmede kullanilmisti (0270018 gibi),
    # geri almak icin tersine kopyalamak yeter.
    js2 = glob.glob("_ds1/DataSet/*.json")
    jk2 = {kimlik(os.path.basename(f))[1] for f in js2}
    st2 = glob.glob("all_wscad_stp/*.stp")
    sk2 = collections.defaultdict(list)
    for s in st2:
        sk2[step_kimlik(s)].append(s)
    oksuz = sorted(set(sk2) - jk2)
    # UC KORUMA: oksuz kimlik ne olcum kumesinde ne de eligible()'da olabilir.
    uygun = {p for _, p, _, _ in _eligible()}
    for k in oksuz:
        assert k not in korunan, f"OLCUM PARCASI oksuz sanildi: {k}"
        assert k not in uygun, f"UYGUN KORPUS parcasi oksuz sanildi: {k}"
    os.makedirs(ARSIV_OKSUZ, exist_ok=True)
    n2 = 0
    for k in oksuz:
        for x in sk2[k]:
            shutil.move(x, os.path.join(ARSIV_OKSUZ, os.path.basename(x)))
            n2 += 1
    print(f"\n2) JSON'u olmayan STEP -> {ARSIV_OKSUZ}: {n2} dosya ({len(oksuz)} kimlik)")
    print("   F1 etkisi YOK: eligible() JSON uzerinden doner, bu kimliklere hic bakmaz.")
    st2 = glob.glob("all_wscad_stp/*.stp")
    sk2 = {step_kimlik(s): [s] for s in st2}

    print(f"\nSON DURUM: JSON {len(js2)} | STEP {len(st2)} | STEP kimligi {len(sk2)}")
    with io.open("results/korpus_temizlik.json", "w", encoding="utf-8") as f:
        json.dump({"json_tasinan": n1, "step_kopya_tasinan": n3,
                   "oksuz_step_tasinan": n2, "oksuz_step_kimlik": len(oksuz),
                   "json_kalan": len(js2), "step_kalan": len(st2),
                   "tasinan_uretici": dict(tasinan_uretici)}, f, indent=1)
    print("makbuz -> results/korpus_temizlik.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
