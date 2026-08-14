# -*- coding: utf-8 -*-
"""DECISION OLCUTU -- a kolun dagitilip dagitilmayacagina karar veren TEK rule.

WHY IT EXISTS (2026-07-31/08-01 kosusunun most pahali dersi): same gece UC FARKLI cubuk kullandim and
ucu de savunulabilir gorunuyordu:

  * t15/TOPO       : "gorulmemis manufacturer EN KOTU durumu ARTMALI"        -> topoloji GECTI
  * ExtraTrees kolu: "HER IKI manufacturer-disi bolmede de artmali"          -> ExtraTrees OLDU
  * t16/regime-ratio : "artmali" (BUYUKLUK YOK)                           -> +0.0018 with GECTI (noise)
  * u2/R taramasi  : yine buyukluksuz                                    -> R=8 lafzen gecti, bootstrap bitirdi
  * u8/yonlendirme : "each eksende at least X up to iyi"                     -> -0.0020 and GA upper ucu TAM 0.0000'da dustu

Cubugu each deneyde yeniden yazmak, farkinda olmadan SONUCA GORE cubuk secmeye open gate birakir.
Bu modul kurali TEK places tanimlar; deney betikleri only sayilari gives.

RULE (gecerli version):
  Bir arm dagitilabilir however and however
    (1) TANIDIK veride loss <= `tanidik_tolerans` (varsayilan 0.01), VE
    (2) gorulmemis-manufacturer bolmelerinde ORTALAMA at least `min_kazanc` up to artmali
        (varsayilan 0.01 -- ISARET DEGIL BUYUKLUK), VE
    (3) EN KOTU split kotulesmemeli (>= mevcut most kotu - `en_kotu_tolerans`), VE
    (4) no bolmede `maks_bolme_kaybi`'ndan (varsayilan 0.05) extra loss olmamali.

(2) ORTALAMA uzerinden yazildi because "most kotu artsin" single basina RISK TASIMAYA izin veriyor
(ExtraTrees: most kotu +0.053 but diger split -0.039, mean +0.007 = no sey). (4) whereas
ortalamanin single a bolmedeki cokusu gizlemesini engeller.

GA SINIRI: a difference, %95 GA'si sifiri iceriyorsa GURULTUDUR. `ga` verilirse rule bunu da arar.
"""
import numpy as np 


class Karar :
    def __init__ (self ,gecti ,rationale ,ayrinti ):
        self .gecti =bool (gecti );self .rationale =rationale ;self .ayrinti =ayrinti 

    def __bool__ (self ):
        return self .gecti 

    def __str__ (self ):
        line_ ="\n".join (f"    {k :<28}{v }"for k ,v in self .ayrinti .items ())
        return f"{'GECTI'if self .gecti else 'GECMEDI'} -- {self .rationale }\n{line_ }"


def degerlendir (baseline ,candidate ,tanidik_anahtar ="tanidik",*,ga =None ,
tanidik_tolerans =0.01 ,min_kazanc =0.01 ,en_kotu_tolerans =0.0 ,
maks_bolme_kaybi =0.05 ,kanit_gerekli =True ):
    """baseline/candidate: {bolme_adi: F1}. `tanidik_anahtar` disindakiler gorulmemis-manufacturer sayilir.

    ga: {bolme_adi: (lower, upper)} verilirse, tanidik bolmedeki kaybin GURULTU olup olmadigi and
    mean kazancin gercekligi this araliklarla degerlendirilir.
    """
    bolmeler =[k for k in baseline if k !=tanidik_anahtar ]
    assert bolmeler ,"en az bir gorulmemis-manufacturer bolmesi gerekir"
    d ={k :candidate [k ]-baseline [k ]for k in baseline }
    t =d .get (tanidik_anahtar ,0.0 )
    ort =float (np .mean ([d [k ]for k in bolmeler ]))
    ek_t =min (baseline [k ]for k in bolmeler )
    ek_a =min (candidate [k ]for k in bolmeler )
    en_kotu_bolme =min (d [k ]for k in bolmeler )

    k1 =t >=-tanidik_tolerans 
    k2 =ort >=min_kazanc 
    k3 =ek_a >=ek_t -en_kotu_tolerans 
    k4 =en_kotu_bolme >=-maks_bolme_kaybi 
    ayr ={
    "(1) tanidik fark":f"{t :+.4f}  (>= {-tanidik_tolerans :+.4f})  {'OK'if k1 else 'X'}",
    "(2) split ORTALAMASI":f"{ort :+.4f}  (>= {min_kazanc :+.4f})  {'OK'if k2 else 'X'}",
    "(3) en kotu split":f"{ek_t :.4f} -> {ek_a :.4f}  {'OK'if k3 else 'X'}",
    "(4) en buyuk split kaybi":f"{en_kotu_bolme :+.4f}  (>= {-maks_bolme_kaybi :+.4f})  "
    f"{'OK'if k4 else 'X'}",
    }
    if ga :
        for k ,(lo ,hi )in ga .items ():
            ayr [f"    GA {k }"]=(f"[{lo :+.4f}, {hi :+.4f}] "
            f"{'GURULTU (sifiri iceriyor)'if lo <=0 <=hi else 'GERCEK'}")
            # (5) KANIT SARTI -- 2026-08-01 denetiminde acilan hole: GA HESAPLANIP YAZDIRILIYOR but
            # karara KATILMIYORDU (ok = all([k1..k4])). Yani %95 GA'si sifiri iceren, i.e. gurultuden
            # ayirt edilemeyen a kazanc "GECTI" alabiliyordu. Artik:
            #   * ga verilmisse: kazanci TASIYAN bolmelerden EN AZ BIRI sifiri DISLAMALI (lo > 0), and
            #     no bolmede KANITLANMIS large loss olmamali (hi < -maks_bolme_kaybi).
            #   * ga verilmemisse and kanit_gerekli whereas: karar KANITSIZ sayilir and GECMEZ.
            #     (Tarihsel kollari yeniden uretirken kanit_gerekli=False with cagrilir -- that kollarin
            #      GA kapisi olmadan verilmis olmasi bulgunun KENDISIDIR.)
    k5 =True 
    if ga :
        kanitli_kazanc =any (lo >0 for k ,(lo ,hi )in ga .items ()
        if k in bolmeler and d .get (k ,0.0 )>0 )
        kanitli_kayip =any (hi <-maks_bolme_kaybi for k ,(lo ,hi )in ga .items ()if k in bolmeler )
        k5 =kanitli_kazanc and not kanitli_kayip 
        ayr ["(5) KANIT (GA)"]=(f"kanitli kazanc {'VAR'if kanitli_kazanc else 'YOK'}"
        f" | kanitli buyuk loss "
        f"{'VAR'if kanitli_kayip else 'yok'}  {'OK'if k5 else 'X'}")
    elif kanit_gerekli :
        k5 =False 
        ayr ["(5) KANIT (GA)"]="GA VERILMEDI -> karar KANITSIZ  X"
    else :
        ayr ["(5) KANIT (GA)"]="evidence araniyor DEGIL (kanit_gerekli=False)"

    ok =all ([k1 ,k2 ,k3 ,k4 ,k5 ])
    neden =("bes sart da saglandi"if ok else 
    "; ".join (x for x ,c in (("tanidik veride loss fazla",not k1 ),
    ("split ortalamasi yeterince artmadi",not k2 ),
    ("en kotu split kotulesti",not k3 ),
    ("bir bolmede buyuk loss",not k4 ),
    ("evidence yok (GA sifiri iceriyor ya da verilmedi)",
    not k5 ))if c ))
    return Karar (ok ,neden ,ayr )


def _kendini_sina ():
    """Bu gecenin real kollarini kurala sok -- rule, verdigim kararlari yeniden uretiyor mu?"""
    olay =[
    ("ExtraTrees (KILL bekleniyor)",
    {"tanidik":0.7355 ,"WEI":0.4736 ,"PXC":0.7048 },
    {"tanidik":0.7300 ,"WEI":0.5266 ,"PXC":0.6658 },False ),
    ("topoloji (GECMESI bekleniyor)",
    {"tanidik":0.7287 ,"WEI":0.4736 ,"PXC":0.7048 },
    {"tanidik":0.7410 ,"WEI":0.4832 ,"PXC":0.7203 },True ),
    ("R=8 yaricap (KILL bekleniyor: kazanc noise)",
    {"tanidik":0.7410 ,"WEI":0.4832 ,"PXC":0.7203 },
    {"tanidik":0.7422 ,"WEI":0.5013 ,"PXC":0.6961 },False ),
    ("her parcaya z-skor (SINIRDA)",
    {"tanidik":0.7410 ,"WEI":0.4832 ,"PXC":0.7203 },
    {"tanidik":0.7390 ,"WEI":0.5702 ,"PXC":0.6828 },True ),
    ("cokus yonlendirme (GECMESI bekleniyor)",
    {"tanidik":0.7410 ,"WEI":0.4832 ,"PXC":0.7203 },
    {"tanidik":0.7439 ,"WEI":0.5682 ,"PXC":0.7029 },True ),
    ]
    print ("KURAL KENDINI SINIYOR (gecenin gercek kollari):\n")
    hepsi =True 
    for ad ,t ,a ,bek in olay :
    # TARIHSEL kollar GA KAPISI OLMADAN karara baglanmisti -- bulgunun kendisi this.
    # Burada that gunku kurali yeniden uretmek for kanit_gerekli=False veriliyor.
        k =degerlendir (t ,a ,kanit_gerekli =False )
        uy =(bool (k )==bek )
        hepsi &=uy 
        print (f"{ad }\n  {k }\n  beklenen {'GECSIN'if bek else 'KALSIN'} -> "
        f"{'TUTARLI'if uy else 'TUTARSIZ <<<'}\n")
    print ("KURAL, gecenin kararlarini yeniden uretiyor"if hepsi else 
    "KURAL gecenin kararlariyla CELISIYOR -- kural ya da karar yanlisti")
    return hepsi 


if __name__ =="__main__":
    _kendini_sina ()
