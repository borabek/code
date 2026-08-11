# -*- coding: utf-8 -*-
"""R2: "BIR PARCA = BIR EKSEN" varsayimini SINA (medyanla degil, DAGILIMLA).

R1'de parca ici GT eksen benzerliginin MEDYANI 1.000 cikti. Medyan bir varsayimi TASIYAMAZ:
|cos| kullandigimiz icin BIRBIRINE TERS bakan iki sira da 1.000 verir, ve medyan 1.000 iken
parcanin ucte biri dik olabilir.

Planlanan duzeltme (parca ici eksen uzlasisi: susan CP'lere guvenilir eksenden yon tasi) TAM
OLARAK bu varsayima dayaniyor. Yanlissa duzeltme dogru eksenleri de BOZAR -- yani bu olcum
kill kriterinin kendisi.

OLCULEN:
  1. Her parcada GT eksenlerinin BASKIN yone gore acilari -> parcalarin yuzde kaci "tek eksenli"?
  2. Tek-eksenli olmayan parcalar hangileri, kac CP'li?
  3. Uzlasi uygulansaydi GT'ye gore TAVAN ne olurdu (her CP'ye parcanin baskin GT ekseni
     verilseydi kac CP 10 derece icinde kalirdi)?
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def baskin_eksen(D):
    """Isaretten bagimsiz baskin yon: sum(d d^T) matrisinin en buyuk ozvektoru."""
    C = (D[:, :, None] * D[:, None, :]).sum(0)
    ev, evec = np.linalg.eigh(C)
    return evec[:, -1]


def main():
    with open("results/_u4_der.pkl", "rb") as f:
        DER = pickle.load(f)
    with open("results/r0_kayit.pkl", "rb") as f:
        K = pickle.load(f)

    print("1) PARCA ICI GT EKSEN TUTARLILIGI")
    tek, karisik, sapma_hep = 0, [], []
    for r in DER:
        G = np.asarray(r["Gd"], float)
        if len(G) < 2:
            continue
        b = baskin_eksen(G)
        ac = np.degrees(np.arccos(np.clip(np.abs(G @ b), 0, 1)))
        sapma_hep.extend(ac.tolist())
        if ac.max() <= 10.0:
            tek += 1
        else:
            karisik.append((r["pid"], len(G), float(ac.max()),
                            int((ac > 10).sum())))
    n = tek + len(karisik)
    print(f"  {n} parca (>=2 CP) | TEK EKSENLI (hepsi 10 derece icinde): {tek} ({tek/n:.1%})")
    print(f"  KARISIK: {len(karisik)} ({len(karisik)/n:.1%})")
    s = np.array(sapma_hep)
    print(f"  tum CP'lerin baskin eksene sapmasi: medyan {np.median(s):.2f} | "
          f"%90 {np.percentile(s,90):.2f} | %99 {np.percentile(s,99):.2f} | maks {s.max():.2f}")
    print(f"  10 derece icinde kalan CP orani: {(s <= 10).mean():.1%}")

    print("\n2) KARISIK PARCALAR (en cok sapan 10)")
    for pid, ncp, mx, kac in sorted(karisik, key=lambda x: -x[2])[:10]:
        print(f"    {pid:<14} {ncp:>3} CP | maks sapma {mx:>6.1f} deg | {kac} CP disarida")

    print("\n3) UZLASI TAVANI: her CP'ye parcanin BASKIN GT ekseni verilseydi")
    print(f"  10 derece icinde kalirdi: {(s <= 10).mean():.1%} (su anki aci gecisi %81.4)")
    print("  NOT: bu bir TAVAN -- gercekte baskin ekseni GT'den degil TAHMINLERDEN kestirecegiz.")

    print("\n4) TAHMINLERDEN kestirilen baskin eksen GT baskin eksenine ne kadar yakin?")
    fark = []
    for r in DER:
        G = np.asarray(r["Gd"], float); P = np.asarray(r["Pd"], float)
        if len(G) < 1 or len(P) < 2:
            continue
        bg, bp = baskin_eksen(G), baskin_eksen(P)
        fark.append(float(np.degrees(np.arccos(np.clip(abs(float(bg @ bp)), 0, 1)))))
    f = np.array(fark)
    print(f"  {len(f)} parca | medyan {np.median(f):.2f} deg | %75 {np.percentile(f,75):.2f} | "
          f"%90 {np.percentile(f,90):.2f}")
    print(f"  10 derece icinde: {(f <= 10).mean():.1%}  <- uzlasinin GERCEKCI tavani")

    print("\n5) UZLASI KIMI DUZELTIR, KIMI BOZAR? (tespitte eslesen noktalar uzerinde)")
    # Her eslesme icin: mevcut aci vs parcanin TAHMIN-baskin ekseni kullanilsaydi olacak aci
    per = collections.defaultdict(list)
    for e in K:
        per[e["pid"]].append(e)
    DERD = {r["pid"]: r for r in DER}
    duzelen = bozulan = ayni = 0
    yeni_aci = []
    for pid, es in per.items():
        r = DERD.get(pid)
        if r is None or len(r["Pd"]) < 2:
            continue
        bp = baskin_eksen(np.asarray(r["Pd"], float))
        G = np.asarray(r["Gd"], float)
        if not len(G):
            continue
        # uzlasi ekseninin GT eksenlerine acisi (parca basina tek deger, GT'ler zaten paralel)
        a_yeni = float(np.degrees(np.arccos(np.clip(np.abs(G @ bp), 0, 1)).min())) if len(G) else 90.0
        for e in es:
            yeni_aci.append(a_yeni)
            eski_ok = e["aci"] <= 10.0
            yeni_ok = a_yeni <= 10.0
            duzelen += int(yeni_ok and not eski_ok)
            bozulan += int(eski_ok and not yeni_ok)
            ayni += int(eski_ok == yeni_ok)
    tot = duzelen + bozulan + ayni
    print(f"  {tot} eslesme | DUZELEN {duzelen} ({duzelen/max(tot,1):.1%}) | "
          f"BOZULAN {bozulan} ({bozulan/max(tot,1):.1%}) | degismeyen {ayni}")
    print(f"  NET: {duzelen - bozulan:+d} nokta")
    print("  -> BOZULAN sayisi buyukse uzlasi HERKESE degil, yalniz GUVENSIZ eksenlere "
          "uygulanmali (secici uzlasi).")

    with open("results/r2_eksen_varsayimi.json", "w", encoding="utf-8") as f_:
        json.dump({"tek_eksenli_parca_orani": tek / max(n, 1),
                   "cp_10deg_icinde": float((s <= 10).mean()),
                   "tahmin_baskin_10deg": float((f <= 10).mean()),
                   "uzlasi_duzelen": duzelen, "uzlasi_bozulan": bozulan,
                   "uzlasi_net": duzelen - bozulan}, f_, indent=1)
    print("\nmakbuz -> results/r2_eksen_varsayimi.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
