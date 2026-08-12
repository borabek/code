# -*- coding: utf-8 -*-
"""GT YON SOZLESMESI TUTARLI MI? (marka icinde ve markalar arasinda)

BULGU (2026-08-12, III. KOL): analitik eksen kullanildiginda SUPU/MOR'da
"merkezden agza" (DISARI) isaret dogru sonuc veriyor, UPUN'da ise "agizdan
merkeze" (ICERI). Yani GT'nin yon isareti markaya gore DEGISIYOR gorunuyor.

BU CIDDI. Robot metrigi ISARETLI aci kullaniyor; sozlesme tutarsizsa dogru
tahminler cezalandiriliyor olabilir ve bu D7 dahil BUTUN yon olcumlerini
etkiler.

BU SONDA dogrudan GT'ye bakar: her CP icin yon, parcanin agirlik merkezinden
DISARI mi ICERI mi bakiyor? Olcu: (CP - merkez) . yon isareti.

  disari_oran : parcadaki CP'lerin kaci DISARI bakiyor
Marka icinde 0'a ya da 1'e yakinsa sozlesme TUTARLI (yonu ne olursa olsun).
0.5 civarindaysa parca icinde bile KARISIK demektir -- o zaman sorun
sozlesme degil, GT ya da geometri.

D7'ye BAKILMAZ (yalniz d6 + tam kayitlari).
"""
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K   # noqa: E402
import d6_kayit          # noqa: E402


def main():
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, r in kay.items():
        G = np.asarray(r.get("G", []), float)
        Gd = np.asarray(r.get("Gd", []), float)
        if len(G) < 2 or len(Gd) != len(G):
            continue
        n = np.linalg.norm(Gd, axis=1, keepdims=True)
        if (n < 1e-9).any():
            continue
        Gn = Gd / n
        merkez = G.mean(0)
        disa = ((G - merkez) * Gn).sum(1) > 0
        a = ist[r.get("mfg", "?")]
        a["parca"].append(1)
        a["gt"].append(len(G))
        a["disari"].append(float(disa.mean()))
        # PARCA ICI tutarlilik: cogunluk ne kadar baskin
        a["baskinlik"].append(float(max(disa.mean(), 1 - disa.mean())))

    print(f"{'marka':<8}{'parca':>7}{'GT':>7}{'DISARI orani':>14}"
          f"{'parca ici baskinlik':>21}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        if len(a["parca"]) < 5:
            continue
        d = float(np.mean(a["disari"]))
        b = float(np.mean(a["baskinlik"]))
        out[m_] = {"parca": len(a["parca"]), "gt": sum(a["gt"]),
                   "disari_orani": d, "parca_ici_baskinlik": b}
        print(f"{m_:<8}{len(a['parca']):>7}{sum(a['gt']):>7}{d:>14.3f}"
              f"{b:>21.3f}")
    json.dump({"marka": out,
               "not": "disari_orani = GT yonunun parca merkezinden DISARI "
                      "bakma orani. parca_ici_baskinlik = her parcada "
                      "cogunluk yonun payi (1.0 = parca icinde tam tutarli). "
                      "D7'ye BAKILMADI."},
              open("results/gt_yon_sozlesmesi.json", "w"), indent=1)
    print("\nmakbuz -> results/gt_yon_sozlesmesi.json")
    print("OKUMA:")
    print("  baskinlik ~1.0 -> parca ICINDE tutarli (sozlesme var)")
    print("  disari_orani markalar arasinda dagilmissa -> SOZLESME MARKAYA")
    print("     GORE DEGISIYOR; isaret bir PARCA OZELLIGI olarak ogrenilebilir")
    print("  baskinlik ~0.5 -> parca icinde bile karisik; sorun daha derin")


if __name__ == "__main__":
    main()
