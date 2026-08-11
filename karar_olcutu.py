# -*- coding: utf-8 -*-
"""KARAR OLCUTU -- bir kolun dagitilip dagitilmayacagina karar veren TEK kural.

NEDEN VAR (2026-07-31/08-01 kosusunun en pahali dersi): ayni gece UC FARKLI cubuk kullandim ve
ucu de savunulabilir gorunuyordu:

  * t15/TOPO       : "gorulmemis uretici EN KOTU durumu ARTMALI"        -> topoloji GECTI
  * ExtraTrees kolu: "HER IKI uretici-disi bolmede de artmali"          -> ExtraTrees OLDU
  * t16/rejim-oran : "artmali" (BUYUKLUK YOK)                           -> +0.0018 ile GECTI (gurultu)
  * u2/R taramasi  : yine buyukluksuz                                    -> R=8 lafzen gecti, bootstrap bitirdi
  * u8/yonlendirme : "her eksende en az X kadar iyi"                     -> -0.0020 ve GA ust ucu TAM 0.0000'da dustu

Cubugu her deneyde yeniden yazmak, farkinda olmadan SONUCA GORE cubuk secmeye acik kapi birakir.
Bu modul kurali TEK yerde tanimlar; deney betikleri yalniz sayilari verir.

KURAL (gecerli surum):
  Bir kol dagitilabilir ancak ve ancak
    (1) TANIDIK veride kayip <= `tanidik_tolerans` (varsayilan 0.01), VE
    (2) gorulmemis-uretici bolmelerinde ORTALAMA en az `min_kazanc` kadar artmali
        (varsayilan 0.01 -- ISARET DEGIL BUYUKLUK), VE
    (3) EN KOTU bolme kotulesmemeli (>= mevcut en kotu - `en_kotu_tolerans`), VE
    (4) hicbir bolmede `maks_bolme_kaybi`'ndan (varsayilan 0.05) fazla kayip olmamali.

(2) ORTALAMA uzerinden yazildi cunku "en kotu artsin" tek basina RISK TASIMAYA izin veriyor
(ExtraTrees: en kotu +0.053 ama diger bolme -0.039, ortalama +0.007 = hicbir sey). (4) ise
ortalamanin tek bir bolmedeki cokusu gizlemesini engeller.

GA SINIRI: bir fark, %95 GA'si sifiri iceriyorsa GURULTUDUR. `ga` verilirse kural bunu da arar.
"""
import numpy as np


class Karar:
    def __init__(self, gecti, gerekce, ayrinti):
        self.gecti = bool(gecti); self.gerekce = gerekce; self.ayrinti = ayrinti

    def __bool__(self):
        return self.gecti

    def __str__(self):
        satir = "\n".join(f"    {k:<28}{v}" for k, v in self.ayrinti.items())
        return f"{'GECTI' if self.gecti else 'GECMEDI'} -- {self.gerekce}\n{satir}"


def degerlendir(taban, aday, tanidik_anahtar="tanidik", *, ga=None,
                tanidik_tolerans=0.01, min_kazanc=0.01, en_kotu_tolerans=0.0,
                maks_bolme_kaybi=0.05, kanit_gerekli=True):
    """taban/aday: {bolme_adi: F1}. `tanidik_anahtar` disindakiler gorulmemis-uretici sayilir.

    ga: {bolme_adi: (alt, ust)} verilirse, tanidik bolmedeki kaybin GURULTU olup olmadigi ve
    ortalama kazancin gercekligi bu araliklarla degerlendirilir.
    """
    bolmeler = [k for k in taban if k != tanidik_anahtar]
    assert bolmeler, "en az bir gorulmemis-uretici bolmesi gerekir"
    d = {k: aday[k] - taban[k] for k in taban}
    t = d.get(tanidik_anahtar, 0.0)
    ort = float(np.mean([d[k] for k in bolmeler]))
    ek_t = min(taban[k] for k in bolmeler)
    ek_a = min(aday[k] for k in bolmeler)
    en_kotu_bolme = min(d[k] for k in bolmeler)

    k1 = t >= -tanidik_tolerans
    k2 = ort >= min_kazanc
    k3 = ek_a >= ek_t - en_kotu_tolerans
    k4 = en_kotu_bolme >= -maks_bolme_kaybi
    ayr = {
        "(1) tanidik fark": f"{t:+.4f}  (>= {-tanidik_tolerans:+.4f})  {'OK' if k1 else 'X'}",
        "(2) bolme ORTALAMASI": f"{ort:+.4f}  (>= {min_kazanc:+.4f})  {'OK' if k2 else 'X'}",
        "(3) en kotu bolme": f"{ek_t:.4f} -> {ek_a:.4f}  {'OK' if k3 else 'X'}",
        "(4) en buyuk bolme kaybi": f"{en_kotu_bolme:+.4f}  (>= {-maks_bolme_kaybi:+.4f})  "
                                    f"{'OK' if k4 else 'X'}",
    }
    if ga:
        for k, (lo, hi) in ga.items():
            ayr[f"    GA {k}"] = (f"[{lo:+.4f}, {hi:+.4f}] "
                                  f"{'GURULTU (sifiri iceriyor)' if lo <= 0 <= hi else 'GERCEK'}")
    # (5) KANIT SARTI -- 2026-08-01 denetiminde acilan delik: GA HESAPLANIP YAZDIRILIYOR ama
    # karara KATILMIYORDU (ok = all([k1..k4])). Yani %95 GA'si sifiri iceren, yani gurultuden
    # ayirt edilemeyen bir kazanc "GECTI" alabiliyordu. Artik:
    #   * ga verilmisse: kazanci TASIYAN bolmelerden EN AZ BIRI sifiri DISLAMALI (lo > 0), ve
    #     hicbir bolmede KANITLANMIS buyuk kayip olmamali (hi < -maks_bolme_kaybi).
    #   * ga verilmemisse ve kanit_gerekli ise: karar KANITSIZ sayilir ve GECMEZ.
    #     (Tarihsel kollari yeniden uretirken kanit_gerekli=False ile cagrilir -- o kollarin
    #      GA kapisi olmadan verilmis olmasi bulgunun KENDISIDIR.)
    k5 = True
    if ga:
        kanitli_kazanc = any(lo > 0 for k, (lo, hi) in ga.items()
                             if k in bolmeler and d.get(k, 0.0) > 0)
        kanitli_kayip = any(hi < -maks_bolme_kaybi for k, (lo, hi) in ga.items() if k in bolmeler)
        k5 = kanitli_kazanc and not kanitli_kayip
        ayr["(5) KANIT (GA)"] = (f"kanitli kazanc {'VAR' if kanitli_kazanc else 'YOK'}"
                                 f" | kanitli buyuk kayip "
                                 f"{'VAR' if kanitli_kayip else 'yok'}  {'OK' if k5 else 'X'}")
    elif kanit_gerekli:
        k5 = False
        ayr["(5) KANIT (GA)"] = "GA VERILMEDI -> karar KANITSIZ  X"
    else:
        ayr["(5) KANIT (GA)"] = "kanit araniyor DEGIL (kanit_gerekli=False)"

    ok = all([k1, k2, k3, k4, k5])
    neden = ("bes sart da saglandi" if ok else
             "; ".join(x for x, c in (("tanidik veride kayip fazla", not k1),
                                      ("bolme ortalamasi yeterince artmadi", not k2),
                                      ("en kotu bolme kotulesti", not k3),
                                      ("bir bolmede buyuk kayip", not k4),
                                      ("kanit yok (GA sifiri iceriyor ya da verilmedi)",
                                       not k5)) if c))
    return Karar(ok, neden, ayr)


def _kendini_sina():
    """Bu gecenin gercek kollarini kurala sok -- kural, verdigim kararlari yeniden uretiyor mu?"""
    olay = [
        ("ExtraTrees (KILL bekleniyor)",
         {"tanidik": 0.7355, "WEI": 0.4736, "PXC": 0.7048},
         {"tanidik": 0.7300, "WEI": 0.5266, "PXC": 0.6658}, False),
        ("topoloji (GECMESI bekleniyor)",
         {"tanidik": 0.7287, "WEI": 0.4736, "PXC": 0.7048},
         {"tanidik": 0.7410, "WEI": 0.4832, "PXC": 0.7203}, True),
        ("R=8 yaricap (KILL bekleniyor: kazanc gurultu)",
         {"tanidik": 0.7410, "WEI": 0.4832, "PXC": 0.7203},
         {"tanidik": 0.7422, "WEI": 0.5013, "PXC": 0.6961}, False),
        ("her parcaya z-skor (SINIRDA)",
         {"tanidik": 0.7410, "WEI": 0.4832, "PXC": 0.7203},
         {"tanidik": 0.7390, "WEI": 0.5702, "PXC": 0.6828}, True),
        ("cokus yonlendirme (GECMESI bekleniyor)",
         {"tanidik": 0.7410, "WEI": 0.4832, "PXC": 0.7203},
         {"tanidik": 0.7439, "WEI": 0.5682, "PXC": 0.7029}, True),
    ]
    print("KURAL KENDINI SINIYOR (gecenin gercek kollari):\n")
    hepsi = True
    for ad, t, a, bek in olay:
        # TARIHSEL kollar GA KAPISI OLMADAN karara baglanmisti -- bulgunun kendisi bu.
        # Burada o gunku kurali yeniden uretmek icin kanit_gerekli=False veriliyor.
        k = degerlendir(t, a, kanit_gerekli=False)
        uy = (bool(k) == bek)
        hepsi &= uy
        print(f"{ad}\n  {k}\n  beklenen {'GECSIN' if bek else 'KALSIN'} -> "
              f"{'TUTARLI' if uy else 'TUTARSIZ <<<'}\n")
    print("KURAL, gecenin kararlarini yeniden uretiyor" if hepsi else
          "KURAL gecenin kararlariyla CELISIYOR -- kural ya da karar yanlisti")
    return hepsi


if __name__ == "__main__":
    _kendini_sina()
