# -*- coding: utf-8 -*-
"""Y6: SIMETRI KANITI -- insan gozu GEREKTIRMEYEN, MODELDEN BAGIMSIZ GT-tamlik testi.

NEDEN BU: gorsel denetim iki kez konvansiyon sorununa carpti (uretici CP'si kanalin icinde
ve yonu iceri bakiyor; robotunki agizda ve disari) ve kullanici hakli olarak "bakarak
anlayamiyorum" dedi. Uzmanlik isteyen bir yargiyi insana yuklemek yerine, NESNEL bir olcut
kuruyorum.

FIKIR: klemens TEKRARLI bir yapidir -- kutuplar sabit bir ADIMLA dizilir. Ureticinin
LISTELEDIGI CP'lerden bu adimi ve yonu olcebiliriz. Eger robotun buldugu "yanlis" nokta,
listelenmis bir girisin TAM BIR KUTUP ADIMI otesindeyse, o nokta ureticinin kendi
oruntusunun bir dugumundedir: yani orada AYNI cinsten bir giris olmasi gerekir ve uretici
onu listelememistir.

KRITIK: bu test AGIN CIKTISINI HIC KULLANMAZ. Girdi yalniz (a) ureticinin CP listesi ve
(b) robotun buldugu noktanin KONUMU. Ne segmentasyon olasiligi, ne gate skoru, ne ozniteligi.
Yani "kendi FP'lerimi kendi modelimle dogru ilan etme" dongusu YOK.

ADIM TAHMINI: parcanin GT noktalari arasindaki tum fark vektorleri; en sik gorulen kisa
vektor = kutup adimi. En az 3 GT gerekir; azsa parca testten CIKAR (ve orana katilmaz).

KILL/OKUMA: bu bir OLCUM duzeltmesidir. Cikan oran, "313 FP'nin en az su kadari gercek
listelenmemis giristir" seklinde ALT SINIR olarak raporlanir -- oruntude olmayan bir FP
de gercek olabilir, bu test onu YAKALAMAZ.
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOL = 1.5          # mm, orgu dugumune uzaklik toleransi
MIN_GT = 3


def adim_tahmini(G):
    """GT noktalarindan KUTUP ADIMI vektorunu tahmin et (en sik kisa fark vektoru)."""
    n = len(G)
    if n < MIN_GT:
        return None
    D = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = G[j] - G[i]
            L = np.linalg.norm(v)
            if 1.0 < L < 60.0:
                D.append(v)
    if len(D) < 2:
        return None
    D = np.array(D)
    # en kisa fark uzunlugu civarindaki vektorler = komsu kutup adimi
    L = np.linalg.norm(D, axis=1)
    taban = float(np.percentile(L, 10))
    m = np.abs(L - taban) <= max(0.6, 0.10 * taban)
    if m.sum() < 2:
        return None
    A = D[m]
    # isaret birlestir (v ve -v ayni adim)
    ref = A[0] / (np.linalg.norm(A[0]) + 1e-9)
    A = np.array([v if float(v @ ref) >= 0 else -v for v in A])
    return A.mean(0)


def main():
    import olcum_kumesi

    with io.open("results/fp_denetim.json", encoding="utf-8") as f:
        FD = json.load(f)
    FP = FD["hepsi"]
    DER, _ = olcum_kumesi.kume("results/_der_tam.pkl")
    GT = {r["pid"]: np.asarray(r["G"], float) for r in DER}

    byp = {}
    for i, f_ in enumerate(FP):
        byp.setdefault(f_["pid"], []).append(i)

    SONUC = []
    atlanan_parca = 0
    for pid, idxs in sorted(byp.items()):
        G = GT.get(pid, np.zeros((0, 3)))
        s = adim_tahmini(G)
        if s is None:
            atlanan_parca += 1
            for i in idxs:
                SONUC.append({"idx": i, "pid": pid, "mfg": FP[i]["mfg"],
                              "rejim": FP[i]["rejim"], "test": "yok", "k": None,
                              "sapma": None})
            continue
        adim = float(np.linalg.norm(s))
        for i in idxs:
            p = np.array(FP[i]["nokta"], float)
            en_iyi, en_k = None, None
            for g in G:
                v = p - g
                t = float(v @ s) / (adim ** 2 + 1e-12)     # kac adim otede
                k = int(round(t))
                if k == 0:
                    continue
                sap = float(np.linalg.norm(v - k * s))     # orgu dugumune uzaklik
                if en_iyi is None or sap < en_iyi:
                    en_iyi, en_k = sap, k
            SONUC.append({"idx": i, "pid": pid, "mfg": FP[i]["mfg"],
                          "rejim": FP[i]["rejim"],
                          "test": "orgude" if (en_iyi is not None and en_iyi <= TOL) else "degil",
                          "k": en_k, "sapma": en_iyi, "adim_mm": adim})

    test_edilen = [x for x in SONUC if x["test"] != "yok"]
    orgude = [x for x in test_edilen if x["test"] == "orgude"]
    print(f"\n{len(SONUC)} FP | test edilebilen {len(test_edilen)} "
          f"({atlanan_parca} parca GT<{MIN_GT} -> disarida)")
    if test_edilen:
        p_ = len(orgude) / len(test_edilen)
        z = 1.96; nn = len(test_edilen)
        d_ = 1 + z * z / nn
        m_ = (p_ + z * z / (2 * nn)) / d_
        s_ = z * np.sqrt(p_ * (1 - p_) / nn + z * z / (4 * nn * nn)) / d_
        lo, hi = max(0, m_ - s_), min(1, m_ + s_)
        print(f"\nORGU DUGUMUNDE: {len(orgude)}/{len(test_edilen)} = {p_:.1%} "
              f"(Wilson %95 GA {lo:.1%}-{hi:.1%})")
        print(f"  tolerans {TOL} mm | medyan sapma "
              f"{np.median([x['sapma'] for x in orgude]) if orgude else float('nan'):.2f} mm")
        for m in sorted({x["mfg"] for x in test_edilen}):
            a = [x for x in test_edilen if x["mfg"] == m]
            b = [x for x in a if x["test"] == "orgude"]
            print(f"  {m}: {len(b)}/{len(a)} = {len(b)/len(a):.1%}")
        for rj in ("dusuk", "cok"):
            a = [x for x in test_edilen if x["rejim"] == rj]
            b = [x for x in a if x["test"] == "orgude"]
            if a:
                print(f"  {rj}-CP: {len(b)}/{len(a)} = {len(b)/len(a):.1%}")

        # --- DUZELTILMIS KESINLIK (ALT SINIR)
        tp, fp = FD["tp"], FD["fp"]
        pay_lo, pay, pay_hi = lo, p_, hi
        for ad, q in (("alt sinir", pay_lo), ("nokta", pay), ("ust", pay_hi)):
            k0 = tp / (tp + fp); k1 = (tp + fp * q) / (tp + fp)
            print(f"  {ad:<10} oran {q:.1%} -> kesinlik {k0:.4f} -> {k1:.4f}")
    with io.open("results/y6_simetri.json", "w", encoding="utf-8") as f:
        json.dump({"tol_mm": TOL, "min_gt": MIN_GT, "sonuc": SONUC,
                   "test_edilen": len(test_edilen), "orgude": len(orgude),
                   "not": ("Bu test AGIN CIKTISINI KULLANMAZ: girdi yalniz uretici CP listesi "
                           "+ robot noktasinin KONUMU. Sonuc bir ALT SINIRDIR -- oruntude "
                           "olmayan bir FP de gercek olabilir, bu test onu yakalamaz.")},
                  f, indent=1)
    print("makbuz -> results/y6_simetri.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
