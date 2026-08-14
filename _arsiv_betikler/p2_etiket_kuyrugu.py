# -*- coding: utf-8 -*-
"""P2 ILK ADIM: INSAN ETIKETI KUYRUGU -- rastgele DEGIL, olculmus olcutle.

GEREKCE ([[yanal-hata-segmentasyon-kalitesiyle-aciklanir]]): agiz kalitesi dusuk
ceyrekte yanal <=2mm basari %25.8, yuksek ceyrekte %75.9 (50 puan). Etiket butcesi
DUSUK KALITELI agizlara harcanmali; bugune kadar rastgele seciliyordu.

PLAN (kullanici): 150-250 parca, 50'lik partiler, oncelik CWT/WEG/KLM/EFX tipi
yayli-non-round-yuksek-CP, %20-30 KOLAY/NEGATIF kontrol (unutmayi onle).

SIRALAMA OLCUTU (dusuk = oncelikli):
  * `kon_cevre` ve `conf` -- yanal hatayla en yuksek korelasyonlu iki sutun
  * yuksek-CP agirligi (cok kutuplu parcalar)
Kolay kontrol ornekleri AYRI isaretlenir.
"""
import collections, io, json, os, pickle, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import wire_gate

N_ZOR, N_KOLAY = 180, 60           # ~%25 kolay/negatif kontrol
ONCELIK = {"CWT", "WEG", "KLM", "EFX", "NIT", "MOR"}

R = pickle.load(open("results/_der_yeni_g10_n22.pkl", "rb"))
FN = list(wire_gate.FEAT_NAMES)
i_cev = FN.index("kon_cevre"); i_conf = FN.index("conf")
kayit = []
for r in R:
    X = r.get("X"); G = r.get("G")
    if X is None or G is None or not len(G):
        continue
    X = np.asarray(X, float)
    if not len(X):
        continue
    kayit.append({"pid": r["pid"], "mfg": r["mfg"], "n_cp": len(G),
                  "cevre": float(np.median(X[:, i_cev])),
                  "conf": float(np.median(X[:, i_conf]))})
print(f"aday parca {len(kayit)}")
cv = np.asarray([k["cevre"] for k in kayit]); cf = np.asarray([k["conf"] for k in kayit])


def z(a):
    s = a.std() or 1.0
    return (a - a.mean()) / s


skor = z(cv) + z(cf)                       # DUSUK skor = zayif agiz
for i, k in enumerate(kayit):
    k["skor"] = float(skor[i])
    k["oncelik"] = (1.0 if k["mfg"] in ONCELIK else 0.0) + \
                   (0.5 if k["n_cp"] >= 8 else 0.0)
zor = sorted(kayit, key=lambda k: (-k["oncelik"], k["skor"]))[:N_ZOR]
kolay = sorted(kayit, key=lambda k: -k["skor"])[:N_KOLAY]
kuy = [dict(k, tur="ZOR") for k in zor] + [dict(k, tur="KOLAY_KONTROL") for k in kolay]
for i, k in enumerate(kuy):
    k["parti"] = i // 50 + 1
d = collections.Counter(k["mfg"] for k in zor)
print(f"ZOR {len(zor)} | KOLAY_KONTROL {len(kolay)} | parti {max(k['parti'] for k in kuy)}")
print(f"ZOR marka dagilimi: {dict(d)}")
print(f"ZOR medyan cevre {np.median([k['cevre'] for k in zor]):.2f} vs "
      f"TUM {np.median(cv):.2f} | ZOR medyan conf {np.median([k['conf'] for k in zor]):.3f} "
      f"vs TUM {np.median(cf):.3f}")
json.dump({"kuyruk": kuy, "n_zor": len(zor), "n_kolay": len(kolay),
           "olcut": "z(kon_cevre)+z(conf) DUSUK = zayif agiz; oncelik markalari ve "
                    "yuksek-CP one alinir; %25 kolay/negatif kontrol",
           "gerekce": "agiz kalitesi dusuk ceyrekte yanal basari %25.8 / yuksek %75.9"},
          io.open("results/p2_etiket_kuyrugu.json", "w", encoding="utf-8"), indent=1)
print("makbuz -> results/p2_etiket_kuyrugu.json")
