# -*- coding: utf-8 -*-
"""MANSET: bir olcum makbuzundan DURUST rapor uretir.

Kullanim:  python manset_050.py results/<makbuz>.json [results/<taban>.json]

Tek bir sayi vermez. Bir sayinin sisik olup olmadigini anlamak icin gereken
her seyi yan yana koyar:

  1. MIKRO robot F1 (manset) ve tespit -- havuzlanmis TP/FP/FN uzerinden.
     `f1w` (rejim-agirlikli) KULLANILMAZ: ayni veride 0.0838 vs 0.1956 verir.
  2. PARCA BOOTSTRAP guven araligi (%95). Parca ornekler, cunku mikro F1 parca
     basina TP/FP/FN toplamidir.
  3. MARKA BAZINDA kirilim + makro + en kotu marka. Tek bir markanin cok GT'si
     mikro'yu tasiyabilir (D6'da NIT GT'nin %46'si); marka tablosu bunu gorunur
     kilar.
  4. TEMIZ ALT KUME duyarliligi: `results/bolme_denetimi.json` icindeki, hicbir
     egitim parcasiyla kaba iz (kutu 0.5mm + GT sayisi) paylasmayan D7 parcalari.
     Manset ile bu alt kume arasindaki fark buyukse sayi supheli demektir.
  5. TABAN verilirse FARK ve markalarin kaci artida.

Makbuz `parca_kirilim` alanini icermelidir (pid -> {mfg, rob:[tp,fp,fn],
tes:[tp,fp,fn]}). `sonda_dagitim_dogrula.py` ve `sonda_d6_urun.py` yazar.
"""
import json
import os
import sys

import numpy as np


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def topla(kir, pidler=None, alan="rob"):
    tp = fp = fn = 0
    for p, v in kir.items():
        if pidler is not None and p not in pidler:
            continue
        a = v[alan]
        tp += a[0]; fp += a[1]; fn += a[2]
    return tp, fp, fn


def bootstrap(kir, pidler=None, alan="rob", n=4000, tohum=0):
    """PARCA bootstrap. Grup bootstrap'a gerek yok: bolme denetimi D7 icinde
    kaba-iz ikiz oranini %22.3 olcmustu ve ikizler AYNI markadadir; parca
    ornekleme burada guven araligini daraltmaz, cunku birim zaten parcadir."""
    ps = sorted(kir) if pidler is None else sorted(pidler)
    A = np.asarray([kir[p][alan] for p in ps], float)
    rng = np.random.default_rng(tohum)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(A), len(A))
        s = A[i].sum(0)
        v.append(f1(s[0], s[1], s[2]))
    v = np.asarray(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def marka_tablo(kir, pidler=None, alan="rob"):
    d = {}
    for p, v in kir.items():
        if pidler is not None and p not in pidler:
            continue
        a = d.setdefault(v["mfg"], [0, 0, 0, 0])
        a[0] += v[alan][0]; a[1] += v[alan][1]; a[2] += v[alan][2]; a[3] += 1
    return {m: {"f1": f1(*a[:3]), "n": a[3], "TP": a[0], "FP": a[1],
                "FN": a[2]} for m, a in d.items()}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    yol = sys.argv[1]
    m = json.load(open(yol))
    kir = m["sonuc"].get("parca_kirilim") or m["sonuc"].get("parca_tp_fp_fn")
    if not kir:
        sys.exit("makbuzda `parca_kirilim` yok -- olcumu yeniden kos.")
    taban = None
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        t = json.load(open(sys.argv[2]))
        taban = t["sonuc"].get("parca_kirilim") or \
            t["sonuc"].get("parca_tp_fp_fn")

    print(f"MAKBUZ  {yol}")
    print(f"kol     P6 {m['sonuc'].get('p6_acik')} | genis "
          f"{m['sonuc'].get('genis_acik')} | poz kafasi "
          f"{m['sonuc'].get('poz_kafasi')}")
    print(f"parca   {len(kir)}")
    sy = m["sonuc"].get("p6_sayac") or {}
    if sy.get("cagri"):
        p6 = sy.get("p6", 0)
        print(f"P6 kolu {p6}/{sy['cagri']} parcada CALISTI "
              f"({100 * p6 / sy['cagri']:.1f}%) | rejim disi "
              f"{sy.get('rejim_disi', 0)} | tablo yok {sy.get('tablo_yok', 0)}")
        if p6 == 0:
            print("!! P6 HIC CALISMAMIS -- bu sayi TABAN sayisidir.")
    print()

    for alan, ad in (("rob", "ROBOT (yanal<=2mm, isaretli aci<=10)"),
                     ("tes", "TESPIT (konum, yon serbest)")):
        tp, fp, fn = topla(kir, alan=alan)
        a, b = bootstrap(kir, alan=alan)
        print(f"{ad}")
        print(f"  MIKRO F1 {f1(tp, fp, fn):.4f}   %95 GA [{a:.4f}, {b:.4f}]"
              f"   TP {tp}  FP {fp}  FN {fn}"
              f"   recall {tp / max(tp + fn, 1):.4f}"
              f"   kesinlik {tp / max(tp + fp, 1):.4f}")

    mt = marka_tablo(kir)
    print(f"\n{'marka':<8}{'n':>5}{'robot F1':>10}{'TP':>7}{'FP':>7}{'FN':>7}"
          + ("{:>10}".format("taban") if taban else ""))
    tb = marka_tablo(taban) if taban else {}
    art = 0
    for k, v in sorted(mt.items(), key=lambda x: -x[1]["FN"] - x[1]["TP"]):
        satir = (f"{k:<8}{v['n']:>5}{v['f1']:>10.4f}{v['TP']:>7}{v['FP']:>7}"
                 f"{v['FN']:>7}")
        if taban and k in tb:
            satir += f"{tb[k]['f1']:>10.4f}"
            art += int(v["f1"] > tb[k]["f1"])
        print(satir)
    print(f"{'MAKRO':<8}{'':>5}{np.mean([v['f1'] for v in mt.values()]):>10.4f}")
    print(f"{'EN KOTU':<8}{'':>5}{min(v['f1'] for v in mt.values()):>10.4f}")
    if taban:
        t_tp, t_fp, t_fn = topla(taban)
        n_tp, n_fp, n_fn = topla(kir)
        print(f"\nTABAN {sys.argv[2]}")
        print(f"  robot {f1(t_tp, t_fp, t_fn):.4f} -> {f1(n_tp, n_fp, n_fn):.4f}"
              f"   ({f1(n_tp, n_fp, n_fn) - f1(t_tp, t_fp, t_fn):+.4f})"
              f"   {art}/{len(mt)} markada ARTI")

    # --- TEMIZ ALT KUME duyarliligi ---------------------------------------
    bd = "results/bolme_denetimi.json"
    if os.path.exists(bd):
        temiz = set(json.load(open(bd))["sonuc"].get("d7_temiz", []))
        ort = temiz & set(kir)
        if len(ort) > 20:
            tp, fp, fn = topla(kir, ort)
            a, b = bootstrap(kir, ort)
            tam = f1(*topla(kir))
            print(f"\nTEMIZ ALT KUME ({len(ort)} parca -- hicbir egitim "
                  f"parcasiyla kaba iz paylasmayan)")
            print(f"  robot {f1(tp, fp, fn):.4f}   %95 GA [{a:.4f}, {b:.4f}]"
                  f"   manset farki {f1(tp, fp, fn) - tam:+.4f}")
            print("  (fark buyukse manset geometri benzerliginden besleniyor "
                  "demektir)")


if __name__ == "__main__":
    main()
