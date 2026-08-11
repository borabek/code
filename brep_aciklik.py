# -*- coding: utf-8 -*-
"""P4: SILINDIR DISI B-REP ACIKLIKLARI -- kare / yarik / push-in girisler.

NEDEN GEREKLI (2026-08-06 teshisi, D6 temiz sinavi):
  * **SE**: GT'lerinin %95'inde parcada kullanilabilir SILINDIR YOK (8 parcada 68 silindir)
  * **NIT**: GT'lerinin %70'i mevcut TUM silindir eksenlerine DIK; silindirlerinin medyan
    yaricapi 0.50mm -- yani pah/pim, tel deligi degil (2.5mm^2 tel 1.78mm ister)
Bu ureticilerin girisleri silindirik DEGIL. Silindir tabanli her poz kolu onlara
YAPISAL olarak ulasamaz; medyan aci tam 90 derece cikiyor.

BU MODUL NE YAPAR: her DUZLEMSEL yuzun IC KENAR HALKALARINI (delik konturlari) bulur.
Bir kare/yarik girisin agzi, duzlemde yatan kapali bir kenar halkasidir; giris yonu de
o duzlemin NORMALIDIR. Yani silindirin (eksen, agiz) ciftinin tam karsiligi:
(normal, halka merkezi).

DIS KONTUR ELENIR: her duzlemsel yuzun bir dis siniri vardir ve o bir ACIKLIK degildir.
Ayirt etme, halkanin CEVRELEDIGI ALANIN yuzun toplam alanina orani ile yapilir --
dis kontur en buyuk alanlidir.

TEZ DEGISMEZ: bu bir SECENEK URETICISIDIR; `v_o` turetmesi, 5 sinif ve ~6000 remesh
aynen kalir. Uretilen secenekler P4/P5'te aday olarak yarisir.
"""
import numpy as np

# Tel giris agzi olabilecek makul boyut araligi (mm). Silindir tarafindaki
# R_MIN/R_MAX (0.5-4.5mm yaricap) ile ayni fiziksel mantik: 2.5mm^2 iletken 1.78mm.
ESD_MIN = 0.8      # esdeger yaricap alt sinir
ESD_MAX = 6.0      # esdeger yaricap ust sinir
DIS_KONTUR_PAY = 0.75   # yuz alaninin bu kadarini cevreleyen halka DIS konturdur


def _halka_noktalari(wire, tol=0.25):
    """Kenar halkasini noktalara ornekle (BRepAdaptor_Curve ile parametrik)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    P = []
    ex = BRepTools_WireExplorer(wire)
    while ex.More():
        e = ex.Current(); ex.Next()
        try:
            ad = BRepAdaptor_Curve(e)
            u0, u1 = ad.FirstParameter(), ad.LastParameter()
            n = max(2, int(abs(u1 - u0) / max(tol, 1e-6)))
            n = min(n, 60)
            for i in range(n):
                p = ad.Value(u0 + (u1 - u0) * i / n)
                P.append((p.X(), p.Y(), p.Z()))
        except Exception:
            continue
    return np.asarray(P, float)


def _duzlem_alani(P, n):
    """Kapali halkanin duzlem icindeki alani (kabuk formulu, 3B'de vektorel)."""
    if len(P) < 3:
        return 0.0
    c = P.mean(0)
    Q = P - c
    A = 0.5 * np.linalg.norm(np.cross(Q, np.roll(Q, -1, axis=0)).sum(0))
    return float(abs(A))


def acikliklar(step_path, esd_min=ESD_MIN, esd_max=ESD_MAX):
    """Duzlemsel yuzlerdeki IC halkalar. Doner: [{center, normal, esd_r, cevre, alan}]

    `esd_r` = esdeger yaricap = sqrt(alan/pi); yuvarlak olmayan agzi silindirle
    kiyaslanabilir kilar ve boyut kapisini tek sayida toplar.
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    rd = STEPControl_Reader()
    if rd.ReadFile(step_path) != 1:
        return []
    rd.TransferRoots()
    shape = rd.OneShape()
    out = []
    fx = TopExp_Explorer(shape, TopAbs_FACE)
    while fx.More():
        f = TopoDS.Face_s(fx.Current()); fx.Next()
        try:
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() != GeomAbs_Plane:
                continue
            pl = ad.Plane(); ax = pl.Axis().Direction()
            n = np.array([ax.X(), ax.Y(), ax.Z()], float)
            n /= (np.linalg.norm(n) + 1e-12)
        except Exception:
            continue
        halkalar = []
        wx = TopExp_Explorer(f, TopAbs_WIRE)
        while wx.More():
            w = TopoDS.Wire_s(wx.Current()); wx.Next()
            P = _halka_noktalari(w)
            if len(P) < 3:
                continue
            halkalar.append((P, _duzlem_alani(P, n)))
        if len(halkalar) < 2:
            continue                       # ic halka yok -> aciklik yok
        en_buyuk = max(h[1] for h in halkalar)
        for P, A in halkalar:
            if A >= DIS_KONTUR_PAY * en_buyuk:
                continue                   # DIS kontur
            esd = float(np.sqrt(max(A, 0.0) / np.pi))
            if not (esd_min <= esd <= esd_max):
                continue
            cev = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0),
                                       axis=1).sum())
            out.append({"center": P.mean(0), "normal": n, "esd_r": esd,
                        "cevre": cev, "alan": float(A),
                        "yuvarlaklik": float(4 * np.pi * A / max(cev * cev, 1e-9)),
                        # POLIGON da doner (2026-08-06): oto-etiket agzi DISK olarak
                        # boyuyor; kare/yarik girislerde bu YANLIS geometri ve
                        # oz-tutarlilik kapisi o parcalari eliyor. Gercek kesitle
                        # boyayabilmek icin halka noktalari saklanir (en fazla 64).
                        "poligon": np.asarray(P[::max(1, len(P) // 64)], np.float32)})
        if len(out) > 4000:                # patolojik parcalarda dur
            break
    return out
