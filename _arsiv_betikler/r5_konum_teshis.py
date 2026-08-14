# -*- coding: utf-8 -*-
"""R5: KONUM SECICISININ TESHISI -- kod yazmadan ONCE olc.

NEDEN BU SIRA: yon kolunda ilk surum SIFIR birakmisti ve sebebi etiketin ISARETSIZ
olmasiydi (sozluk +- ciftleri iceriyordu, +a ile -a AYNI etiketi aliyordu). Kolu
"olu" ilan etmek yerine ETIKETIN kendisine bakmak +0.0363 kazandirdi. Konum kolu da
daha once "kendi kahininin %0'ini yakaliyor" diye kapanmisti. Ayni hatayi tersinden
yapmamak icin once SORUNUN CINSINI olcuyorum.

DORT SORU:
  1. Yon secicisi dagitildiktan SONRA yanal tarafta ne kadar oda kaldi?
     (aci gecen ama yanal gecmeyen ciftler = konumun gercek av sahasi)
  2. KURTARMA vs HASAR dengesi: sozlukte, mevcut>2mm iken <=2mm'ye ceken giris kac;
     mevcut<=2mm iken >2mm'ye iten giris kac?
  3. Etiket OGRENILEBILIR mi: pozitif orani nedir? Yon kolunda isaretsiz etiket %31.4
     ile doymustu; konumda benzer bir DOYMA varsa (her giris pozitif) argmax rastgele
     secer ve kol yine sifir birakir.
  4. Konum TESPITI de oynatir (tespit olcutu yanal max(3mm, %6 diag)). Kahin tespitte
     ne kadar kazandirir/kaybettirir?

Bu betik HICBIR SEY DAGITMAZ. Sadece results/r5_konum_teshis.json uretir.
"""
import io, json, os, pickle, sys
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import wire_gate
from sina_kume import esle, f1w
from d1_yon_dagit import satirla

ONB = "results/_r4_sozluk.pkl"
MARJ = 0.05


def yon_uygula(d_, YS):
    """Dagitilan ISARETLI yon secicisini uygula -- taban ARTIK bu."""
    Pd = d_["Pd"].copy()
    if YS is None:
        return Pd
    ax, _, RJ = satirla(d_["X"], d_["SY"], d_["UZ"], d_["Pd"], None)
    if not ax:
        return Pd
    pr = YS["clf"].predict_proba(np.array(ax, float))[:, 1]
    SK_ = {}
    for n_, (i, gi) in enumerate(RJ):
        SK_.setdefault(i, {})[gi] = pr[n_]
    for i, sc in SK_.items():
        g = max(sc, key=sc.get)
        if g != 0 and sc[g] - sc.get(0, 0.0) >= MARJ:
            Pd[i] = d_["SY"][i][g][1]
    return Pd


def main():
    DER, gate, ek = T2.yukle()
    assert os.path.exists(ONB), f"{ONB} yok -- once r4_birlesik_secici kosmali"
    PARCA = pickle.load(open(ONB, "rb"))
    YS = wire_gate._load("results/yon_secici.pkl")
    print(f"sozluk {len(PARCA)} parca | yon secici {'VAR' if YS else 'YOK'}", flush=True)

    # ---- sayaclar
    n_cift = 0            # tespit toleransinda eslesen (aday, GT) cifti
    n_aci_gecen = 0       # acisi <=10 derece (ISARETLI) olan
    n_av = 0              # aci gecti AMA yanal>2mm  -> konumun av sahasi
    n_kurtarma = 0        # av sahasinda sozlukte <=2mm bir giris VAR
    n_hasar_riski = 0     # mevcut<=2mm iken sozlukte >2mm'ye iten giris VAR
    n_zaten = 0           # mevcut zaten <=2mm
    kazanan = {}          # kurtaran girisin adi
    doyma = []            # her aday icin: sozlukteki <=2mm giris orani
    yanal_mevcut = []
    KOL = {k: [] for k in ("taban_rob", "kahin_rob", "taban_det", "kahin_det")}
    gg = []

    for r in DER:
        d_ = PARCA.get(r["pid"])
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        if d_ is None:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); SK = []
        else:
            P = d_["P"].copy(); Pd = yon_uygula(d_, YS); SK = d_["SK"]
        Pk = P.copy()
        if len(P) and len(G):
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            tol_d = max(3.0, 0.06 * float(r["diag"]))
            for i in range(len(P)):
                if not np.isfinite(pe[i]).any():
                    continue
                b = int(np.argmin(pe[i]))
                if pe[i, b] > tol_d:
                    continue                      # tespitte eslesmiyor: konumun isi degil
                n_cift += 1
                aci = np.degrees(np.arccos(np.clip(float(Pd[i] @ Gd[b]), -1, 1)))
                yan0 = float(pe[i, b])
                yanal_mevcut.append(yan0)
                if aci <= 10.0:
                    n_aci_gecen += 1
                # sozlugu tara
                en, ep, ea = yan0, None, "mevcut"
                iyi = 0; top = 0; kotu = False
                for a, q in (SK[i] if i < len(SK) else {}).items():
                    w = np.asarray(q, float) - G[b]
                    yy = float(np.linalg.norm(w - float(w @ Gd[b]) * Gd[b]))
                    top += 1
                    if yy <= 2.0:
                        iyi += 1
                    else:
                        kotu = True
                    if yy < en:
                        en, ep, ea = yy, np.asarray(q, float), a
                if top:
                    doyma.append(iyi / top)
                if yan0 <= 2.0:
                    n_zaten += 1
                    if kotu:
                        n_hasar_riski += 1
                elif aci <= 10.0:
                    n_av += 1
                    if en <= 2.0:
                        n_kurtarma += 1
                        kazanan[ea] = kazanan.get(ea, 0) + 1
                if ep is not None:
                    Pk[i] = ep
        KOL["taban_rob"].append((rj,) + esle(P, Pd, G, Gd, r["diag"], 2.0, 10.0, False, isaretli=True))
        KOL["kahin_rob"].append((rj,) + esle(Pk, Pd, G, Gd, r["diag"], 2.0, 10.0, False, isaretli=True))
        KOL["taban_det"].append((rj,) + esle(P, Pd, G, Gd, r["diag"], 0.0, 180.0, True))
        KOL["kahin_det"].append((rj,) + esle(Pk, Pd, G, Gd, r["diag"], 0.0, 180.0, True))
        gg.append(r["geo"])

    Y = np.array(yanal_mevcut); D_ = np.array(doyma)
    print(f"\n--- 1. AV SAHASI (yon secicisi UYGULANMIS taban) ---")
    print(f"tespitte eslesen cift        {n_cift}")
    print(f"  acisi <=10 (isaretli)      {n_aci_gecen}  ({n_aci_gecen/max(n_cift,1):.1%})")
    print(f"  yanal zaten <=2mm          {n_zaten}  ({n_zaten/max(n_cift,1):.1%})")
    print(f"  AV: aci gecti + yanal>2mm  {n_av}  ({n_av/max(n_cift,1):.1%} tum ciftin)")
    print(f"\n--- 2. KURTARMA vs HASAR ---")
    print(f"av sahasinda sozluk kurtarabiliyor  {n_kurtarma}/{n_av} "
          f"({n_kurtarma/max(n_av,1):.1%})")
    print(f"gecen noktada bozan giris VAR       {n_hasar_riski}/{n_zaten} "
          f"({n_hasar_riski/max(n_zaten,1):.1%})")
    print(f"kurtaran giris adlari: {dict(sorted(kazanan.items(), key=lambda x: -x[1]))}")
    print(f"\n--- 3. ETIKET DOYMASI ---")
    print(f"aday basina <=2mm giris orani: medyan {np.median(D_):.2f} | "
          f"ort {D_.mean():.2f} | tamami-pozitif olan aday {float((D_>=0.999).mean()):.1%} | "
          f"hic-pozitif-olmayan {float((D_<=0.001).mean()):.1%}")
    print(f"mevcut yanal hata: medyan {np.median(Y):.2f}mm | %75 {np.percentile(Y,75):.2f} | "
          f"%90 {np.percentile(Y,90):.2f}")
    print(f"\n--- 4. KAHIN TAVANI ---")
    tr, kr = f1w(KOL["taban_rob"]), f1w(KOL["kahin_rob"])
    td, kd = f1w(KOL["taban_det"]), f1w(KOL["kahin_det"])
    print(f"{'':<22}{'robot(FIZIKSEL)':>17}{'tespit':>10}")
    print(f"{'taban':<22}{tr:>17.4f}{td:>10.4f}")
    print(f"{'konum KAHINI':<22}{kr:>17.4f}{kd:>10.4f}")
    print(f"{'fark':<22}{kr-tr:>+17.4f}{kd-td:>+10.4f}")

    hukum = []
    if n_av / max(n_cift, 1) < 0.05:
        hukum.append("AV SAHASI COK KUCUK -- konum kolu yapisal olarak kucuk")
    if n_kurtarma / max(n_av, 1) < 0.30:
        hukum.append("SOZLUK KURTARAMIYOR -- zenginlestirme gerekir, secici degil")
    if float((D_ >= 0.999).mean()) > 0.5:
        hukum.append("ETIKET DOYMUS -- ayirt edici etiket kurulmali (yon kolundaki isaret hatasinin muadili)")
    if kr - tr < 0.02:
        hukum.append("KAHIN TAVANI DUSUK -- mukemmel secici bile +0.02 vermez, KOL KAPANIR")
    if not hukum:
        hukum.append("KOL ACIK: av sahasi var, sozluk kurtariyor, etiket ayirt edici, tavan yeterli")
    print("\nHUKUM:")
    for h in hukum:
        print("  *", h)

    json.dump({"n_cift": n_cift, "n_aci_gecen": n_aci_gecen, "n_av": n_av,
               "n_zaten": n_zaten, "n_kurtarma": n_kurtarma, "n_hasar_riski": n_hasar_riski,
               "kurtaran_girisler": kazanan,
               "doyma_medyan": float(np.median(D_)) if len(D_) else None,
               "doyma_tamami_pozitif": float((D_ >= 0.999).mean()) if len(D_) else None,
               "yanal_medyan": float(np.median(Y)) if len(Y) else None,
               "taban_robot": tr, "kahin_robot": kr, "d_robot": kr - tr,
               "taban_tespit": td, "kahin_tespit": kd, "d_tespit": kd - td,
               "hukum": hukum},
              io.open("results/r5_konum_teshis.json", "w"), indent=1)
    print("\nmakbuz -> results/r5_konum_teshis.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
