# -*- coding: utf-8 -*-
"""PROTOKOL: egitimde NE KULLANILAMAZ -- tek kaynak, zorunlu bekci (F0-1).

NEDEN AYRI MODUL: `olcum_kumesi.sinav_egitim_maskesi()` zaten vardi ama ISTEGE BAGLIYDI --
cagiran unutursa hicbir sey uyarmiyordu. 2026-08-03'te tam bu oldu: dagitilan egitim
korpusu 95 grup-temiz LOCKED parcasinin 77'sini icermis, ve bu ancak elle arandiginda
bulundu. Olculdu (2026-08-04, bu modul yazilirken):

    dagitilan `zengin_parite_w2.npz` : 1709 parca
      LOCKED grubuna ait            :  128 parca  <- IHLAL
      OLCUM kumesi grubuna ait      :  391 parca

IKI AYRI YASAK VAR, KARISTIRILMAMALI:
  * LOCKED (tek-atislik sinav)  : HER egitimde yasak. Ihlal edilirse sinav hakki YANAR.
  * OLCUM kumesi (194 parca)    : o kume uzerinde OLCULECEK gate'te yasak. F0-1'e gore
    194'un tamami "development"tir -- yani AYAR/HUKUM bolmelerinde kullanilabilir, ama
    uzerinde puanlanacak modelin egitimine giremez.

YASAK PARCA DEGIL **GEOMETRI GRUBU** duzeyindedir: parcalarin %80'inin korpusta ikizi var
([[geometry-twin-leakage]]). Parca kimligiyle diskalamak ikizi disarida birakmaz.
"""
import io
import json

import numpy as np

import olcum_kumesi as OK

MAKBUZ = "results/protokol_dogrulama.json"


def yasak_gruplar(olcum_da=False, der_yolu="results/_der_tam.pkl"):
    """Egitimde kullanilamayacak GEOMETRI GRUPLARI."""
    g = set(OK.locked_gruplari())
    if olcum_da:
        D, _ = OK.kume(der_yolu)
        g |= {r["geo"] for r in D}
    return g


def egitim_maskesi(pidler, olcum_da=False, der_yolu="results/_der_tam.pkl"):
    """Egitimde KULLANILABILIR olanlar True. Grup duzeyinde calisir."""
    gk = OK.geo_anahtarlari()
    yg = yasak_gruplar(olcum_da, der_yolu)
    g = np.array([gk.get(str(p), "yok:" + str(p)) for p in pidler])
    return ~np.isin(g, list(yg))


def dogrula(pidler, ad="egitim", olcum_da=False, sert=True, der_yolu="results/_der_tam.pkl"):
    """Protokol bekcisi. Ihlal varsa `sert=True` ise HATA FIRLATIR.

    Sessiz gecis YOKTUR: bu fonksiyon her cagrida makbuz yazar, boylece "bekci kosuldu mu"
    sorusu sonradan yanitlanabilir.
    """
    pidler = [str(p) for p in pidler]
    m = egitim_maskesi(pidler, olcum_da, der_yolu)
    ihlal = sorted(set(np.array(pidler)[~m].tolist()))
    kayit = {"ad": ad, "n_parca": len(set(pidler)), "n_ihlal_parca": len(ihlal),
             "olcum_da_yasak": bool(olcum_da), "ilk_ihlaller": ihlal[:20]}
    try:
        with io.open(MAKBUZ, encoding="utf-8") as f:
            hepsi = json.load(f)
    except Exception:
        hepsi = []
    hepsi = [h for h in hepsi if h.get("ad") != ad][-49:] + [kayit]
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump(hepsi, f, indent=1, ensure_ascii=False)
    if ihlal and sert:
        raise AssertionError(
            f"PROTOKOL IHLALI [{ad}]: {len(ihlal)} parca yasak geometri grubunda "
            f"(LOCKED{' + OLCUM' if olcum_da else ''}). Ilk 10: {ihlal[:10]}. "
            f"`protokol.egitim_maskesi()` ile filtreleyin.")
    return m


# --------------------------------------------------------------------------------------
# F0-3 DAIMI OLCUM KURALLARI -- her kol icin gecerli, istisnasiz.
KURALLAR = """
1. KILL ONCEDEN YAZILIR. Kol kosmadan once "neyi gecerse yasar" yazili olmali; sonucu
   gorup esik belirlemek yasak. (2026-08-04: r9c'de esik tutmadi ve KAYDIRILMADI.)
2. AYAR = DEV + ATANMAMIS.  HUKUM = VAL.  Ayar kumesinde secilen TEK ayar VAL'de raporlanir.
3. HAVUZLANMIS SATIR SECIM ICIN KULLANILMAZ -- ayar kumesini de icerdigi icin yaniltir.
   Olculdu (T1): DEV'de +0.0277 gosteren kural VAL'de -0.0162 cikti; havuzlanmis +0.0011
   diyordu. Havuzlanmis yalniz BILGI satiridir.
4. GRUP BOOTSTRAP: parca degil GEOMETRI GRUBU. Parcalarin %80'inin ikizi var; parca
   bootstrap'i guven araligini SAHTE DARALTIR.
5. TEK REJIMLI ALT KUMEDE `f1w` CAGIRMA -- `f1_rejim` kullan (agirliklar var olan rejimler
   uzerinden normalize edilir).
6. DAGITIM OLURSA: duman_testi.py + pytest tests/ + manset YENIDEN URETILIR ve config'e
   provenance yazilir.
7. ADAY-URETIMI kararlari (cluster_mm / min_vertices / dedupe / promote / havuzlama)
   ONBELLEKLE OLCULEMEZ -- yeniden turetme sart.
8. TEZ DEGISMEZLERI (asagidaki TEZ sozlugu) her kolun basinda dogrulanir.
9. BULUNAN HATA GOZ ARDI EDILMEZ. Bir madde biterken cikan hata/uyari/tutarsizlik
   "sonra bakariz" diye gecilmez: LISTEYE H-maddesi olarak eklenir, ETKISI OLCULUR ve
   duzeltilir. Duzeltme olcum kumesini oynatabiliyorsa once ETKI raporlanir, sonra
   uygulanir. (Bu proje hatalari bulup ertelediginde her seferinde bedelini odedi:
   diffusion_net dususu, gate bayatlamasi, isaretsiz kahin, kimlik ayristirmasi.)
"""


def bolmeler(der_yolu="results/_der_tam.pkl"):
    """(ayar_pidleri, hukum_pidleri) -- kural 2'nin TEK kaynagi."""
    s3 = json.load(io.open(OK.SPLIT, encoding="utf-8"))
    val = {str(p) for p in s3["val"]["parts"]}
    D, _ = OK.kume(der_yolu)
    hepsi = {r["pid"] for r in D}
    return (hepsi - val), (hepsi & val)


# --------------------------------------------------------------------------------------
# TEZ DEGISMEZLERI -- 34 maddelik listenin TAMAMI boyunca dokunulmayacak uc sey.
# Soz olarak degil TEST olarak tutulur: her kol kosarken bu dogrulanir.
#   1. AG      : Scheffler DiffusionNet, 5 sinif (Housing/Contact/SnapPoint/CableEntry/
#                LabelSurface) -- dagitilan 4 checkpoint'lik topluluk
#   2. ORGU    : UNIFORM IZOTROPIK remesh, hedef ~6000 tepe (tezin domain-gap cozumu)
#   3. CP TANIMI: v_o = acikligin AGIZ sinir noktalarindan turetilen merkez (Abb. 44)
# Yardimci basliklar, gate, secici ve fiziksel bayraklar SON ISLEMDIR; bunlari degistirmek
# tezden sapma DEGILDIR. Yukaridaki ucunu degistirmek SAPMADIR.
TEZ = {"remesh_hedef": 6000, "n_sinif": 5, "n_checkpoint": 4,
       "cp_def": "cp-v3-thesis-connection-classes"}


def tez_dogrula(sert=True):
    """Tez degismezleri yerinde mi? Liste boyunca her kolun basinda cagrilir."""
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    sapma = []
    cks = cfg["current_product"].get("checkpoints") or cfg.get("robot_vote2_checkpoints") or []
    if len(cks) != TEZ["n_checkpoint"]:
        sapma.append(f"topluluk {len(cks)} uye (tez surumu {TEZ['n_checkpoint']})")
    if cfg.get("cp_def_version") != TEZ["cp_def"]:
        sapma.append(f"cp_def_version={cfg.get('cp_def_version')} (beklenen {TEZ['cp_def']})")
    try:
        import inspect

        import thesis_remesh
        src = inspect.getsource(thesis_remesh.remesh_uniform)
        if "target" not in src:
            sapma.append("thesis_remesh.remesh_uniform imzasi degismis")
    except Exception as e:
        sapma.append(f"thesis_remesh okunamadi: {type(e).__name__}")
    if sapma and sert:
        raise AssertionError("TEZDEN SAPMA: " + " | ".join(sapma))
    return sapma


def _selftest():
    """Bekcinin GERCEKTEN yakaladigini dogrula -- 'sessiz gecis' testi."""
    lg = sorted(yasak_gruplar())
    assert lg, "LOCKED grubu bos -- protokol kurulamaz"
    gk = OK.geo_anahtarlari()
    kirli = [p for p, g in gk.items() if g == lg[0]]
    assert kirli, "LOCKED grubuna ait parca bulunamadi"
    m = egitim_maskesi(kirli)
    assert not m.any(), "bekci LOCKED parcasini yakalamadi"
    try:
        dogrula(kirli, ad="_selftest", sert=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("dogrula() ihlalde HATA FIRLATMADI -- sessiz gecis!")
    temiz = [p for p, g in gk.items() if g not in yasak_gruplar()][:5]
    assert egitim_maskesi(temiz).all(), "bekci temiz parcayi yanlislikla eledi"
    return True


if __name__ == "__main__":
    import sys
    print(f"LOCKED yasak grubu: {len(yasak_gruplar())}")
    print(f"LOCKED+OLCUM yasak grubu: {len(yasak_gruplar(olcum_da=True))}")
    d = np.load("results/zengin_parite_w2.npz", allow_pickle=True)
    pid = [str(x) for x in d["pids"]]
    for ad, oc in (("dagitilan_gate_LOCKED", False), ("dagitilan_gate_LOCKED+OLCUM", True)):
        m = egitim_maskesi(pid, olcum_da=oc)
        u = set(pid); ui = set(np.array(pid)[~m].tolist())
        print(f"{ad:<30} {len(u)} parca -> ihlal {len(ui)} | aday {len(pid)} -> temiz {int(m.sum())}")
        dogrula(pid, ad=ad, olcum_da=oc, sert=False)
    print(f"\n_selftest: {'GECTI' if _selftest() else 'KALDI'}")
    print(f"makbuz -> {MAKBUZ}")
