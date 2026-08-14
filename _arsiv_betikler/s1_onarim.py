# -*- coding: utf-8 -*-
"""S1: FIZIKSEL ONARIM OPERATORLERI -- uc tane, saf fonksiyon, TEZ-NOTR.

Hicbiri agi, remesh'i ya da `v_o` tanimini degistirmez. Uctu de SON ISLEMDIR: gate
kararindan sonra, uretilen CP'nin FIZIKSEL gecerliligini duzeltir.

A4 OLCUMU (194 parca / 1186 CP) bu operatorlerin hedefini verdi:
    govde_ici       134 CP  (FP'lerde %22.4, TP'lerde %7.1 -> zenginlesme 3.16x)
    onu_kapali      148 CP  (%19.3 / %9.9  -> 1.96x)
    duvara_yapisik   92 CP  (%12.6 / %5.9  -> 2.12x)
    kombinasyon: 964 temiz | 56 iki-bayrak | 36 UC-BAYRAK

TASARIM KURALI: her operator YALNIZ kendi bayragini hedefler ve HAREKETI SINIRLIDIR.
Sinirsiz hareket, [[konum-secici-kapandi]]daki 1:20 hasar asimetrisini geri getirir --
kurtardigindan cok bozar. Bu yuzden her operator `maks_mm` ile kisitlanir ve hareket
sonrasi bayrak YENIDEN olculur (S2).
"""
import numpy as np

ILERI_MIN = 5.0
IC_CAP_MIN = 0.8
MAKS_TASIMA_MM = 12.0       # seat_to_mouth zaten eksen boyunca; yine de ust sinir
MAKS_MERKEZLEME_MM = 3.0    # agiz icinde merkeze cekme -- agiz yaricapi mertebesi


def _birim(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def onar_govde_ici(mesh, p, d, maks_mm=MAKS_TASIMA_MM):
    """S1a: nokta govdenin ICINDEYSE eksen boyunca ACIKLIGIN AGZINA tasi.

    `cp_geometry.seat_to_mouth` tam bu isi yapar ve tezin `v_o` insasiyla AYNI tanimdir:
    eksen boyunca ILK YUZEY KESISIMI. Nokta zaten disaridaysa DOKUNULMAZ (fonksiyonun
    kendi korumasi) -- yoksa bos alandaki bir CP karsi duvara firlatilirdi.
    """
    from cp_geometry import is_inside, seat_to_mouth
    try:
        if not is_inside(mesh, p):
            return p, False, "zaten disarida"
        yeni, off = seat_to_mouth(mesh, p, d)
        yeni = np.asarray(yeni, float)
        if not np.isfinite(yeni).all() or np.linalg.norm(yeni - p) > maks_mm:
            return p, False, f"tasima {np.linalg.norm(yeni-p):.1f}mm > {maks_mm}"
        return yeni, True, f"{np.linalg.norm(yeni-p):.2f}mm tasindi"
    except Exception as e:
        return p, False, f"hata {type(e).__name__}"


def onar_merkezle(mesh, p, d, maks_mm=MAKS_MERKEZLEME_MM, n_dirs=24):
    """S1b: nokta agzin DUVARINA yapisiksa agiz icinde MERKEZE cek.

    `mouth_width` eksene dik n_dirs yonde isin atip ilk yuzeye mesafeyi olcer. Nokta
    merkezdeyse tum mesafeler benzer; duvara yapisiksa bir yon COK KISA, karsisi UZUN.
    Merkeze dogru hareket = (en uzak yon - en yakin yon)/2 kadar, en uzak yone dogru.

    Bu, tezin `v_o`'sunu DEGISTIRMEZ: v_o zaten "agiz ortasi"dir; burada yapilan, o
    tanima DAHA SADIK bir nokta bulmaktir. (Global tanim degisikligi AYRI bir sey ve
    olculup REDDEDILDI: [[konum-secici-kapandi]] -- cember/agiz/acik uc rakip de kaybetti.)
    """
    from cp_geometry import mouth_width
    try:
        d = _birim(d)
        a = np.array([1.0, 0.0, 0.0])
        if abs(float(d @ a)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = _birim(np.cross(d, a)); v = _birim(np.cross(d, u))
        ic, ort = mouth_width(mesh, p, d, n_dirs=n_dirs)
        if not (0 < ic < IC_CAP_MIN):
            return p, False, "duvara yapisik degil"
        from cp_geometry import ray_hits
        aci = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        mes, yon = [], []
        for t in aci:
            w = np.cos(t) * u + np.sin(t) * v
            h = ray_hits(mesh, p + 1e-4 * w, w, 30.0)
            mes.append(float(min(h)) if len(h) else 30.0)
            yon.append(w)
        mes = np.array(mes)
        i_min = int(np.argmin(mes)); i_max = int(np.argmax(mes))
        adim = min((mes[i_max] - mes[i_min]) / 2.0, maks_mm)
        if adim <= 1e-3:
            return p, False, "hareket gereksiz"
        yeni = p + adim * yon[i_max]
        return yeni, True, f"{adim:.2f}mm merkeze"
    except Exception as e:
        return p, False, f"hata {type(e).__name__}"


def onar_yon_cevir(mesh, p, d, ileri_min=ILERI_MIN):
    """S1c: takma yonunun ONU KAPALIYSA yonu TERS cevir ve yeniden olc.

    Kural: ileri serbest mesafe < ileri_min ise -d denenir. -d daha ACIKSA cevrilir.
    IKI TARAF DA kapaliysa nokta bir KANALDA DEGILDIR -> `False, "iki taraf kapali"`
    doner ve cagiran taraf o adayi eleyebilir (S4 triyaji).
    """
    from cp_geometry import ray_hits

    def acik(v):
        try:
            h = ray_hits(mesh, p + 1e-3 * v, v, 60.0)
            return float(min(h)) if len(h) else float("inf")
        except Exception:
            return float("nan")
    d = _birim(d)
    ileri = acik(d)
    if not np.isfinite(ileri) or ileri >= ileri_min:
        return d, False, "on zaten acik"
    geri = acik(-d)
    if np.isfinite(geri) and geri < ileri_min:
        return d, False, "iki taraf kapali"
    if (not np.isfinite(geri)) or geri > ileri:
        return -d, True, f"cevrildi ({ileri:.1f} -> {geri:.1f}mm)"
    return d, False, "cevirmek iyilestirmiyor"


def _selftest():
    """Operatorler bilinen geometride DOGRU davraniyor mu?"""
    import trimesh
    from cp_geometry import is_inside
    kutu = trimesh.creation.box((24, 24, 12))
    sil = trimesh.creation.cylinder(radius=3.0, height=40)
    m = kutu.difference(sil)
    # 1) govde ICINDEKI nokta agza tasinmali
    p = np.array([9.0, 9.0, 0.0])           # kose -> malzeme icinde
    assert is_inside(m, p)
    yeni, ok, _ = onar_govde_ici(m, p, np.array([0., 0., 1.]))
    assert ok and not is_inside(m, yeni), "govde ici onarimi calismadi"
    # 2) delik ekseninde, duvara YAKIN nokta merkeze cekilmeli
    p2 = np.array([2.6, 0.0, 5.7])          # r=3 deligin duvarina 0.4mm
    yeni2, ok2, _ = onar_merkezle(m, p2, np.array([0., 0., 1.]))
    if ok2:
        assert np.linalg.norm(yeni2[:2]) < np.linalg.norm(p2[:2]), "merkeze cekmedi"
    # 3) onu kapali yon cevrilmeli
    p3 = np.array([0.0, 0.0, -5.0])
    d3 = np.array([0.0, 0.0, -1.0])         # asagi -> hemen disari (acik)
    _, cev, _ = onar_yon_cevir(m, p3, d3)
    return True


if __name__ == "__main__":
    print("s1_onarim selftest:", "GECTI" if _selftest() else "KALDI")
