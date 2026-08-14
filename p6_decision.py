# -*- coding: utf-8 -*-
"""P6 DECISION KODU -- training de urun de BU fonksiyonlari cagirir.

Bu projede measurement yolu urunden IKI KEZ ayristi and two gece kaybettirdi
([[measurement-zaafiyetleri-kapatildi]], [[measurement-yolu-and-secim-kusurlari]]). Sebep each
seferinde ayniydi: karar kurali two places YAZILMISTI. Burada single places duruyor.

`donustur` : feature donusumu (A+B part-ici z-skor, C+D ham)
`sec`      : acgozlu skor order + konum NMS -> (P, D)
"""
import numpy as np 

AB =67 # A(58 pool) + B(9 mouth, adayin own yonuyle)
NMS_MM =5.0 # dagitilan `wire_gate.crowd_mask` with same scale


def donustur (X ,zskor ="ab"):
    """A+B blogu part-ici z-skor, C+D ham.

    WHY SEPARATE: A+B mutlak buyukluklerdir (olasilik, mm, sayim) and markadan
    markaya olcegi kayar -- part-ici z-skor gate tarihindeki most large single
    kazancti. C+D ZATEN gorelidir (angle, ratio, destek yuzdesi); onlari a more
    normalize etmek p5-v2'de uctan uca 0.1649 -> 0.1465 DUSURMUSTU.
    """
    import wire_gate 
    X =np .asarray (X ,float )
    if zskor =="hepsi":
        return wire_gate .within_part (X ,"zskor")
    if zskor =="yok":
        return X 
    if zskor =="sira":
    # PARCA-ICI SIRA (yuzdelik). Z-skor parcanin ORTALAMA and SAPMASINA
    # baglidir; candidate count and karisimi degisince (training ~232 candidate/part,
    # NIT 407 and cogunlugu mesh) same fiziksel candidate different a z-skora
    # dusuyor. Sira this kaymadan ETKILENMEZ: "this parcadaki most derin 3. hole"
    # ifadesi candidate sayisindan bagimsizdir.
        return np .hstack ([wire_gate .within_part (X [:,:AB ],"sira"),X [:,AB :]])
    if zskor =="ikisi":
    # Hem z-skor hem order: model hangisine nerede guvenecegine karar versin.
        return np .hstack ([wire_gate .within_part (X [:,:AB ],"zskor"),
        wire_gate .within_part (X [:,:AB ],"sira")[:,AB :],
        X [:,AB :]])
    return np .hstack ([wire_gate .within_part (X [:,:AB ],"zskor"),X [:,AB :]])


KAYNAK_AD =["kay_seg","kay_brep","kay_mesh"]


def kaynak_blok (src_ ):
    """Aday kaynagi -> 3 sutunluk gosterge (0 seg / 1 B-rep / 2 mesh tepesi).

    WHY GEREKLI: mesh tepeleri havuzun cogunlugunu olusturur (part basina
    ~250 candidate) and large cogunlugu yanlistir; B-rep agizlari very more few but very
    more zengindir; segmentasyonun `v_o` adayi most azdir. Bu ON OLASILIK farkini
    modele soylememek, ona same isi ogrenmeyi features uzerinden zorlamak
    demek. Onbellek `source` alanini already tasiyor -- yeniden inference gerekmez.
    """
    k =np .asarray (src_ ,int ).reshape (-1 )
    return np .stack ([(k ==0 ),(k ==1 ),(k ==2 )],axis =1 ).astype (float )


def kabul_maskesi (s ,rule_ ):
    """Skorlardan KABUL maskesi. `rule` = ("mutlak", e) ya da ("goreli", ratio, baseline).

    GORELI rule urunun own kuralidir (`p1c_threshold.maske`): candidate, KENDI
    PARCASINDAKI most high skorun `ratio` katini gecmeli VE `baseline`i asmali.
    Parcalar arasi skor olcegi kaydigi for mutlak threshold some parcalarda no
    seyi, bazilarinda each seyi geciriyor.
    """
    s =np .asarray (s ,float )
    if not len (s ):
        return np .zeros (0 ,bool )
    if rule_ [0 ]=="mutlak":
        return s >=rule_ [1 ]
    return (s >=rule_ [1 ]*float (np .max (s )))&(s >=rule_ [2 ])


def sec_ayrintili (P ,idx ,YD ,s ,threshold ,nms_mm =NMS_MM ):
    """`sec` with AYNI karar, but secilen ADAY indekslerini de returns.

    Teshis for: "konumu correct sectik but yonu mu kacirdik?" sorusu however
    secilen adaylarin kimligi bilinerek sorulabilir.
    Doner: (P_sec, D_sec, aday_idx)
    """
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    idx =np .asarray (idx ,int )
    YD =np .asarray (YD ,float ).reshape (-1 ,3 )
    s =np .asarray (s ,float )
    rule_ =("mutlak",float (threshold ))if np .isscalar (threshold )else tuple (threshold )
    k =np .where (kabul_maskesi (s ,rule_ ))[0 ]
    if not len (k ):
    # DORT value: erken cikis dali da normal dalla AYNI imzayi dondurmeli.
    # Skor alani eklendiginde burasi 3'te kalmisti and D7 okumasinin P6 kolu
    # "expected 4, got 3" with CoKTU. Cikis dallari imza degisikliginde
    # gozden kaciyor -- test bunu yakalar.
        return (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),np .zeros (0 ,int ),
        np .zeros (0 ,float ))
    rank_ =k [np .argsort (-s [k ])]
    ap ,ad ,ai ,asc ,kapali =[],[],[],[],set ()
    for j in rank_ :
        i =int (idx [j ])
        if i in kapali :
            continue 
        p =P [i ]
        if ap and float (np .min (np .linalg .norm (
        np .asarray (ap )-p ,axis =1 )))<nms_mm :
            kapali .add (i )
            continue 
        ap .append (p )
        ad .append (YD [j ])
        ai .append (i )
        asc .append (float (s [j ]))# GUVEN KAPISI for: secilen secenegin skoru
        kapali .add (i )
    return (np .asarray (ap ,float ).reshape (-1 ,3 ),
    np .asarray (ad ,float ).reshape (-1 ,3 ),np .asarray (ai ,int ),
    np .asarray (asc ,float ))


def sec (P ,idx ,YD ,s ,threshold ,nms_mm =NMS_MM ):
    """Acgozlu secim: most high skordan basla, konum NMS uygula.

    Bir konum kabul edilince (ya da NMS'e takilinca) that konumun DIGER direction
    secenekleri kapanir -- i.e. a konuma EN IYI direction secilir and a konum
    most extra a prediction produces.

    Doner: (P_sec, D_sec)

    GOVDE YOK -- `sec_ayrintili`ye devreder. Onceden same acgozlu loop IKI KEZ
    yazilmisti; this projede karar kodunun two places durmasi full two times silent
    ayrisma uretti. Tek body, two imza.
    """
    return sec_ayrintili (P ,idx ,YD ,s ,threshold ,nms_mm )[:2 ]
