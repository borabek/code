# -*- coding: utf-8 -*-
"""0.45 ICIN BELIRLEYICI SONDA: yanal hata SEGMENTASYON KALITESIYLE aciklaniyor mu?

ARITMETIK: robot = tespit x donusum. Donusumun tavani = yanal<=2mm gecme orani
(%54.6). 0.45 icin YANAL duzelmek ZORUNDA. `v_o` (agiz-ortasi) tez sabiti ve
degismeyecek -- AMA `v_o` agiz cevresindeki tepelerin ORTALAMASI. Segmentasyon
agzi daha iyi secerse centroid daha DOGRU yere duser. Bu tez-sadik bir yol.

SORU: yanal hata, o adayin segmentasyon kalitesiyle korele mi?
  * EVET ise -> veri kollari (B9/D20/sentetik) yanal'a da etki eder, 0.45 MENZILDE
  * HAYIR ise -> yanal `v_o`'nun ICSEL siniri, teze sadik 0.45 YOK

VEKIL OLCUTLER (aday basina, turetmeden gelir):
  n_verts   : agzi olusturan tepe sayisi (az tepe -> centroid oynak)
  confidence: o bolgedeki ortalama baglanti olasiligi (dusuk -> seg emin degil)
  area      : agiz alani
Bunlar X22'nin icinde. Yanal hata ile ILISKI olcu + KOSULLU DAGILIM.
"""
import pickle, sys, os
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import wire_gate

R = pickle.load(open("results/_der_yeni_g10_n22.pkl", "rb"))
FN = list(wire_gate.FEAT_NAMES)
print(f"X22 sutunlari: {FN}\n")
lat, feats = [], []
for r in R:
    P, D, G = r.get("P"), r.get("Pd"), r.get("G")
    X = r.get("X")
    if P is None or G is None or X is None or not len(P) or not len(G): continue
    if len(X) != len(P): continue
    P = np.asarray(P, float); D = np.asarray(D, float); G = np.asarray(G, float)
    X = np.asarray(X, float)
    tol = max(3.0, 0.06 * r["diag"])
    for i in range(len(P)):
        dd = np.linalg.norm(G - P[i], axis=1); j = int(np.argmin(dd))
        if dd[j] > tol: continue
        u = D[i] / (np.linalg.norm(D[i]) + 1e-9)
        v = G[j] - P[i]
        lat.append(float(np.linalg.norm(v - (v @ u) * u)))
        feats.append(X[i])
lat = np.asarray(lat); Fm = np.asarray(feats)
print(f"eslesen aday {len(lat)} | yanal medyan {np.median(lat):.3f}mm | "
      f"<=2mm %{100*(lat<=2).mean():.1f}\n")

print("YANAL HATA ile KORELASYON (Spearman, |rho| buyuk = aciklayici):")
from scipy.stats import spearmanr
sk = []
for c in range(Fm.shape[1]):
    if np.std(Fm[:, c]) < 1e-9: continue
    rho, _ = spearmanr(Fm[:, c], lat)
    sk.append((abs(rho), rho, FN[c] if c < len(FN) else f"c{c}"))
sk.sort(reverse=True)
for a, rho, ad in sk[:8]:
    print(f"  {ad:<22} rho {rho:+.3f}")

print("\nKOSULLU: en aciklayici sutunun ust/alt ceyreginde yanal <=2mm orani")
_a, _r, ad = sk[0]; c = FN.index(ad) if ad in FN else 0
q1, q3 = np.percentile(Fm[:, c], [25, 75])
alt = lat[Fm[:, c] <= q1]; ust = lat[Fm[:, c] >= q3]
print(f"  {ad} DUSUK ceyrek: <=2mm %{100*(alt<=2).mean():.1f} (medyan {np.median(alt):.2f}mm)")
print(f"  {ad} YUKSEK ceyrek: <=2mm %{100*(ust<=2).mean():.1f} (medyan {np.median(ust):.2f}mm)")
print(f"\n  FARK: {abs(100*(ust<=2).mean()-100*(alt<=2).mean()):.1f} puan")
