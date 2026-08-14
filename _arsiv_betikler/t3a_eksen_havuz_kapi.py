# -*- coding: utf-8 -*-
"""T3a: EKSEN-FARKINDALIKLI HAVUZLAMA -- PAHALI ISTEN ONCEKI UCUZ KAPI.

NEDEN BU KOL: `votes`, gate'in GENELLESEN TEK ozelligi (uretici-disi AUC dususu 0.018;
digerleri 0.12-0.18 -- [[gate-memorizes-not-learns]]). Ve `r1_oy_parcalanmasi` olctu:
uretici CP'lerinin **%52'sinde** uye noktalari 5mm'den genis yayiliyor, yani ayni acikligi
bulan uyeler AYRI adaylara bolunuyor ve genellesen tek sinyal BOZULUYOR.

MESAFEYLE COZULEMEZ: bu gece olctum, GT'lerin en-yakin-komsu mesafesi medyan 5.15mm ve
**%66'sinin 8mm icinde komsusu var**. Parcalanma yayilimi (medyan 5.27mm) ile gercek CP
ADIMI AYNI OLCEKTE. Kumeleme yaricapini buyutmek "ayni acikligi iki uye gordu" ile "iki
komsu CP" ayrimini imkansiz kilar. Cozum yaricap DEGIL, AYNI ACIKLIGA AIT OLMA olmali:
eksen-farkindalikli havuzlama (dik mesafe <= 3mm VE eksenler <=20 derece hizali, DERINLIK
serbest).

NEDEN YENIDEN OLCULUYOR: bu bayrak daha once denendi ve elendi -- ama 8 UYELI havuza
GECISLE BIRLIKTE (G3). Iki degisiklik ayni kosuda oldugu icin eksen havuzunun KENDI etkisi
hic ayrisdirilmadi. Dagitilan 4 uyeli toplulukla YALNIZ BASINA hic olculmedi.

BU BETIK PAHALI ISI YAPMAZ. Adil bir uctan uca olcum, EGITIM korpusunun da yeni dagilimla
yeniden turetilmesini gerektirir (~2-4 saat GPU; aksi halde gate ile aday dagilimi
uyusmaz -- G3'te tam bu hata yapilmisti). Once ucuz kapi:

    (1) aday havuzu GT'nin ne kadarini iceriyor (recall) -- DUSMEMELI
    (2) GT basina OY sayisi -- YUKSELMELI (kolun tum gerekcesi bu)
    (3) aday sayisi -- asiri dusmemeli

KAPI: oy/GT >= +0.15 VE recall dususu <= 0.005. Gecerse pahali is HAK EDILIR.

cp_config.json GECICI olarak yamalanir ve `finally` ile GERI ALINIR (SHA ile dogrulanir).
"""
import hashlib, io, json, os, shutil, subprocess, sys, time
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CFG = "cp_config.json"
YEDEK = "cp_config.json.t3a_yedek"
CIKTI = "results/_der_eksen.pkl"


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def istat(DER):
    """(aday_recall, oy_per_GT, aday_sayisi) -- GT'ye TESPIT toleransiyla bakar."""
    ul = 0; n = 0; na = 0
    for r in DER:
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        n += len(G)
        P = np.asarray(r["P"], float) if r["P"] is not None else np.zeros((0, 3))
        na += len(P)
        if len(P) and len(G):
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            ul += int((pe.min(0) <= max(3.0, 0.06 * float(r["diag"]))).sum())
    return ul / max(n, 1), na, n


OY_SUTUN = 11      # FEAT_NAMES_13 = [..., "aspect", "flat", "chan_conn", "votes", "conf"]


def oy_istat(DER):
    """GT ile eslesen adaylarin ORTALAMA oy sayisi -- URUNUN KENDI `_votes` degeri.

    DIKKAT (ilk surumde tuzaga dusuyordum): oyu "birlesmis noktanin 5mm kuresi icindeki
    uye noktalarini say" diye hesaplamak bu deneyi YANLI yapar. Eksen-farkindalikli
    havuzlama DERINLIK farkini bilerek serbest birakir; o uyeler 5mm kurenin DISINDA
    kalir ve yeni kol kendi kazandigi oylari KAYBETMIS gorunurdu. `wire_gate.feats_for`
    oyu X'in 11. sutununa yaziyor (`float(c.get("_votes", 1))`) -- iki turetme de
    ayni sutunu tasiyor, karsilastirma boylece elmayla elma olur.
    """
    oy = []
    for r in DER:
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        P = np.asarray(r["P"], float) if r["P"] is not None else np.zeros((0, 3))
        X = r.get("X")
        if not len(P) or not len(G) or X is None or np.asarray(X).shape[1] <= OY_SUTUN:
            continue
        X = np.asarray(X, float)
        for b in range(len(G)):
            d = P - G[b]
            a = d @ Gd[b]
            pe = np.linalg.norm(d - a[:, None] * Gd[b], axis=1)
            pe = np.where(np.abs(a) > 40, np.inf, pe)
            if not np.isfinite(pe).any() or pe.min() > max(3.0, 0.06 * float(r["diag"])):
                continue
            oy.append(float(X[int(np.argmin(pe)), OY_SUTUN]))
    return float(np.mean(oy)) if oy else 0.0, len(oy)


def main():
    import pickle
    import olcum_kumesi
    # KONTROL, 2 GUNLUK ONBELLEK DEGIL, AYNI GECE AYNI ORTAMDA URETILMIS TURETMEDIR.
    # Sebep: `diffusion_net` bu gece yeniden kuruldu ve 8 parcalik dogrulamada konumlarin
    # surec-arasi ~0.4mm'ye kadar oynadigi olculdu ([[robot-nondeterminism]] pymeshlab).
    # Taze bir "eksen havuzu ACIK" turetmesini 2 gun onceki onbellekle karsilastirmak,
    # havuzlama etkisini o oynamayla ve aradaki kod kaymasiyla KARISTIRIRDI.
    KONTROL = "results/_der_kontrol.pkl"
    assert os.path.exists(KONTROL), f"{KONTROL} yok -- once bayrak KAPALI turetme kosmali"
    ESKI, rap = olcum_kumesi.kume(KONTROL)
    olcum_kumesi.rapor_bas(rap)
    try:
        _E2, _ = olcum_kumesi.kume("results/_der_tam.pkl")
        _r2, _a2, _g2 = istat(_E2)
        print(f"[kayma bilgisi] 2 gunluk onbellek: recall {_r2:.4f} / {_a2} aday")
    except Exception:
        pass

    if not os.path.exists(CIKTI):
        s0 = sha(CFG)
        shutil.copy2(CFG, YEDEK)
        try:
            c = json.load(io.open(CFG, encoding="utf-8"))
            print(f"mevcut robot_eksen_havuz = {c.get('robot_eksen_havuz')}")
            c["robot_eksen_havuz"] = True
            c["_t3a_gecici"] = ("EKSEN HAVUZ DENEYI -- bu anahtar dosyada goruyorsan "
                                "t3a yarida kesilmis demektir, cp_config.json.t3a_yedek'ten GERI AL")
            with io.open(CFG, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=1, ensure_ascii=False)
            print("cp_config YAMALANDI (robot_eksen_havuz=True); turetme basliyor...", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "turet.py", "--cikti", CIKTI],
                               capture_output=True, text=True)
            print(r.stdout[-2500:])
            if r.returncode != 0:
                print("TURETME HATASI:", r.stderr[-1500:])
            print(f"turetme {time.time()-t0:.0f}s", flush=True)
        finally:
            shutil.copy2(YEDEK, CFG)
            os.remove(YEDEK)
            s1 = sha(CFG)
            print(f"cp_config GERI ALINDI  sha {s0} -> {s1}  "
                  f"{'AYNI (dogrulandi)' if s0 == s1 else '!!! FARKLI -- ELLE KONTROL ET'}")
            assert s0 == s1, "cp_config geri alinamadi"
    else:
        print(f"{CIKTI} zaten var, turetme atlandi")

    with open(CIKTI, "rb") as f:
        YENI_HAM = pickle.load(f)
    YENI, rap2 = olcum_kumesi.kume(CIKTI)
    print(f"\nolcum kumesi: eski {len(ESKI)} parca | yeni {len(YENI)} parca")
    r0, na0, ng0 = istat(ESKI); r1, na1, ng1 = istat(YENI)
    o0, n0 = oy_istat(ESKI); o1, n1 = oy_istat(YENI)
    print(f"\n{'':<22}{'ESKI (5mm kure)':>18}{'YENI (eksen)':>15}{'fark':>10}")
    print(f"{'aday recall':<22}{r0:>18.4f}{r1:>15.4f}{r1-r0:>+10.4f}")
    print(f"{'aday sayisi':<22}{na0:>18}{na1:>15}{na1-na0:>+10}")
    print(f"{'oy / eslesen GT':<22}{o0:>18.3f}{o1:>15.3f}{o1-o0:>+10.3f}")
    print(f"{'  (n)':<22}{n0:>18}{n1:>15}")
    kapi = (o1 - o0) >= 0.15 and (r0 - r1) <= 0.005
    print(f"\nKAPI: oy/GT >= +0.15 VE recall dususu <= 0.005 -> "
          f"{'GECTI -- pahali gate yeniden-turetmesi HAK EDILDI' if kapi else 'GECMEDI -- kol kapanir, pahali is YAPILMAZ'}")
    with io.open("results/t3a_eksen_havuz_kapi.json", "w", encoding="utf-8") as f:
        json.dump({"eski": {"recall": r0, "aday": na0, "oy": o0, "n_oy": n0, "gt": ng0},
                   "yeni": {"recall": r1, "aday": na1, "oy": o1, "n_oy": n1, "gt": ng1},
                   "d_oy": o1 - o0, "d_recall": r1 - r0, "kapi": bool(kapi)}, f, indent=1)
    print("makbuz -> results/t3a_eksen_havuz_kapi.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
