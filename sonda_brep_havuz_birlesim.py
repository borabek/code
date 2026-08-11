# -*- coding: utf-8 -*-
"""BAGLAYICI KISIT HAVUZ: B-rep fiziksel onerileri havuza EKLEMEK recall'u acar mi?

Olculdu (`results/havuz_recall_d7.json`): D7 marka-disi havuz recall G7'de
**0.6654**. Donusum ~%48 oldugu icin robot tavani ~0.32 -- yani MEVCUT HAVUZLA
0.50 IMKANSIZ. Gate/secici/adet uzerine kurulan her kol bu tavanin altinda.

Bu sonda tek soruyu sorar: segmentasyon havuzuna B-rep'ten gelen FIZIKSEL agiz
onerilerini (silindir agizlari + duzlemsel ic halkalar) EKLERSEK recall nereye
cikar? Kaydin `mouth_a`/`mouth_b`/aciklik merkezleri ZATEN diskte.

DURUSTLUK: bu bir TEZ SONUCU DEGIL. Tezin `v_o` turetmesi ve 5 sinif DEGISMIYOR;
B-rep onerileri EK bir aday KAYNAGI. Kullanicinin kendi plani bunu "tez-omurgali
geometrik post-process genisletmesi" olarak ayirmisti ve ayri raporlanir.

Ayrica kesinlik bedeli ACIKCA sayilir: recall bedava degil, aday sayisi artar.
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
from sina_kume import esle_macar

d7 = set(map(str, json.load(open("results/d7_sinav_kumesi.json"))["pidler"]))
R = [r for r in pickle.load(open("results/_der_yeni_G7BIRLESIK.pkl", "rb"))
     if str(r["pid"]) in d7]
cy = pickle.load(open("results/_d7_silindirler.pkl", "rb"))
ac = pickle.load(open("results/_d7_acikliklar.pkl", "rb"))


def brep_oneriler(pid, r_max=None):
    """Silindir agizlari (iki uc) + aciklik merkezleri. Yon = eksen/normal."""
    P, D = [], []
    for c in cy.get(pid) or []:
        if r_max is not None and float(c.get("radius", 0)) > r_max:
            continue
        ax = np.asarray(c["axis"], float)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        for u, s in (("mouth_a", 1.0), ("mouth_b", -1.0)):
            if c.get(u) is not None:
                P.append(np.asarray(c[u], float)); D.append(s * ax)
    for o in ac.get(pid) or []:
        if isinstance(o, dict) and o.get("center") is not None:
            n = np.asarray(o.get("normal", [0, 0, 1]), float)
            n = n / (np.linalg.norm(n) + 1e-12)
            P.append(np.asarray(o["center"], float)); D.append(n)
    return (np.asarray(P, float).reshape(-1, 3),
            np.asarray(D, float).reshape(-1, 3))


KOLLAR = {
    "SEG (kanonik)": lambda pid, P, D: (P, D),
    "SEG + B-rep (tum)": None,
    "SEG + B-rep (r<=3mm)": None,
    "B-rep TEK BASINA": None,
}
sonuc = {}
for ad in KOLLAR:
    TP = FN = nA = 0; per = collections.defaultdict(lambda: [0, 0])
    for r in R:
        pid = str(r["pid"]); G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        P = np.asarray(r["P"], float); D = np.asarray(r["Pd"], float)
        if ad == "SEG + B-rep (tum)":
            bP, bD = brep_oneriler(pid)
            P = np.vstack([P, bP]) if len(bP) else P
            D = np.vstack([D, bD]) if len(bD) else D
        elif ad == "SEG + B-rep (r<=3mm)":
            bP, bD = brep_oneriler(pid, r_max=3.0)
            P = np.vstack([P, bP]) if len(bP) else P
            D = np.vstack([D, bD]) if len(bD) else D
        elif ad == "B-rep TEK BASINA":
            P, D = brep_oneriler(pid)
        tp, fp, fn = esle_macar(P, D, G, np.asarray(r["Gd"], float), r["diag"],
                                max(3.0, 0.06 * r["diag"]), 180.0, True)[:3]
        TP += tp; FN += fn; nA += len(P)
        a = per[r["mfg"]]; a[0] += tp; a[1] += fn
    rc = TP / max(TP + FN, 1)
    sonuc[ad] = {"recall": rc, "aday_per_parca": nA / max(len(R), 1),
                 "marka": {m: a[0] / max(a[0] + a[1], 1) for m, a in per.items()}}
    print(f"{ad:<22} recall {rc:.4f} | aday/parca {sonuc[ad]['aday_per_parca']:>6.1f}",
          flush=True)

t = sonuc["SEG (kanonik)"]
print(f"\n{'marka':<8} {'SEG':>8} {'+B-rep':>8} {'fark':>8}")
u = sonuc["SEG + B-rep (tum)"]
for m in sorted(t["marka"], key=lambda k: t["marka"][k]):
    print(f"  {m:<7} {t['marka'][m]:>7.4f} {u['marka'].get(m,0):>8.4f} "
          f"{u['marka'].get(m,0)-t['marka'][m]:>+8.4f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": sonuc,
           "not": "HAVUZ RECALL. B-rep onerileri TEZ TURETMESI DEGIL, ek aday "
                  "kaynagi -- ayri raporlanir. D7 marka-disi."},
          open("results/brep_havuz_birlesim.json", "w"), indent=1)
