# -*- coding: utf-8 -*-
"""R1: ACIDAKI ~90 DERECE POPULASYONU -- kim, nerede, neden?

R0 bulgusu: aci medyani 0.00 deg (yani cogunlukla MUKEMMEL) ama %90'lik dilim 89.73 ve maksimum
90.00. Acidan kaybedilen 162 noktanin %66'si 45-90 arasinda ve 90'da YIGILIYOR.

Bu bir kestirim gurultusu DEGIL: 90 derece, eksenin DIK secilmesi demektir. Halka bicimli bir
agizda normal-kovaryansin en KUCUK ozvektoru eksendir; en BUYUGU alinirsa tam 90 derece cikar.
Ama kod dogru ozvektoru aliyor -- demek ki ariza belirli bir ALT KUMEDE.

SORULAR (hepsi olculur, hicbiri varsayilmaz):
  1. 90-derecelikler belirli PARCALARDA mi toplaniyor, yoksa her yere mi dagilmis?
  2. Ayni parcada hem 0 hem 90 derece var mi? (varsa parca genelinde bir cerceve sorunu DEGIL)
  3. GT ekseni (Gd) parca eksenlerine gore nasil duruyor? Tahmin (Pd) nasil?
  4. 90-derecelikler dar/derin kanallarda mi, sig agizlarda mi? (aday ozellikleri ile bak)
  5. Iki eksen DIK ise, TAHMINI 90 dondurmek "duzeltir" mi -- yani sistematik bir eksen
     karisikligi mi (a<->b), yoksa rastgele mi?
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    with open("results/r0_kayit.pkl", "rb") as f:
        K = pickle.load(f)
    ac = np.array([e["aci"] for e in K])
    kotu = ac > 10.0
    dik = ac > 45.0
    print(f"{len(K)} eslesme | aci>10: {int(kotu.sum())} | aci>45: {int(dik.sum())} "
          f"({dik.sum()/max(kotu.sum(),1):.1%} kotulerin)")

    print("\n1) 90-DERECELIKLER PARCALARDA TOPLANIYOR MU?")
    pp = collections.defaultdict(lambda: [0, 0])
    for e, b in zip(K, dik):
        pp[e["pid"]][0] += 1; pp[e["pid"]][1] += int(b)
    etkilenen = {p: v for p, v in pp.items() if v[1]}
    print(f"  {len(etkilenen)}/{len(pp)} parcada en az bir dik eksen var")
    tam = [p for p, v in etkilenen.items() if v[1] == v[0]]
    kismi = [p for p, v in etkilenen.items() if 0 < v[1] < v[0]]
    print(f"  TAMAMI dik olan parca : {len(tam)}  (parca genelinde eksen sorunu)")
    print(f"  KISMEN dik olan parca : {len(kismi)}  (parca ici karisik -> cerceve sorunu DEGIL)")
    print(f"  ornek tam-dik parcalar: {tam[:8]}")

    print("\n2) ACI DAGILIMI (kovalar)")
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 30), (30, 60), (60, 85), (85, 90.1)):
        i = (ac >= lo) & (ac < hi)
        print(f"  [{lo:>3}, {hi:>5}) : {int(i.sum()):>4} ({i.mean():>6.1%})")

    print("\n3) DIK OLANLAR nerede? (rejim / uretici / kume)")
    for alan in ("rejim", "mfg", "kume"):
        c = collections.Counter(e[alan] for e, b in zip(K, dik) if b)
        t = collections.Counter(e[alan] for e in K)
        print("  " + alan + ": " + " | ".join(
            f"{k} {c.get(k,0)}/{t[k]} ({c.get(k,0)/t[k]:.1%})" for k in sorted(t)))

    print("\n4) DIK OLANLARIN YANAL MESAFESI farkli mi?")
    ya = np.array([e["yanal"] for e in K])
    print(f"  dik olanlar   : medyan {np.median(ya[dik]):.2f}mm (n={int(dik.sum())})")
    print(f"  digerleri     : medyan {np.median(ya[~dik]):.2f}mm (n={int((~dik).sum())})")
    print("  -> yanali da kotuyse ayni kok neden; iyiyse SADECE eksen sorunu")

    print("\n5) GT EKSENLERI: dik-eslesmelerde GT ekseni ozel mi?")
    with open("results/_u4_der.pkl", "rb") as f:
        DER = {r["pid"]: r for r in pickle.load(f)}
    eks_dik, eks_iyi = [], []
    for e, b in zip(K, dik):
        r = DER.get(e["pid"])
        if r is None or not len(r["Gd"]):
            continue
        # GT eksenlerinin parca icindeki cesitliligi: hepsi ayni yone mi bakiyor?
        c = np.abs(r["Gd"] @ r["Gd"].T)
        (eks_dik if b else eks_iyi).append(float(np.median(c)))
    if eks_dik and eks_iyi:
        print(f"  dik-eslesmelerin parcasinda GT eksen benzerligi (medyan |cos|): "
              f"{np.median(eks_dik):.3f}")
        print(f"  iyi eslesmelerin parcasinda                                  : "
              f"{np.median(eks_iyi):.3f}")
        print("  -> 1.0'a yakin = parcadaki tum CP'ler ayni yone bakiyor")

    print("\n6) TEK PARCA ORNEGI (en cok dik eksenli parca)")
    if etkilenen:
        pid = max(etkilenen, key=lambda p: etkilenen[p][1])
        alt = [e for e in K if e["pid"] == pid]
        r = DER.get(pid)
        print(f"  {pid} | {etkilenen[pid][1]}/{etkilenen[pid][0]} dik | "
              f"rejim {alt[0]['rejim']} | diag {alt[0]['diag']:.0f}mm")
        print(f"  acilar: {sorted(round(e['aci'], 1) for e in alt)}")
        if r is not None and len(r["Gd"]):
            print(f"  GT eksenleri (ilk 4): {np.round(r['Gd'][:4], 3).tolist()}")
            print(f"  tahmin eksenleri (ilk 4): {np.round(r['Pd'][:4], 3).tolist()}")

    with open("results/r1_aci_teshis.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(K), "aci_10_ustu": int(kotu.sum()), "aci_45_ustu": int(dik.sum()),
                   "etkilenen_parca": len(etkilenen), "tam_dik_parca": len(tam),
                   "kismi_dik_parca": len(kismi),
                   "dik_yanal_medyan": float(np.median(ya[dik])) if dik.any() else None,
                   "iyi_yanal_medyan": float(np.median(ya[~dik]))}, f, indent=1)
    print("\nmakbuz -> results/r1_aci_teshis.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
