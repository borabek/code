# -*- coding: utf-8 -*-
"""HALKA NORMALI ILE ISARET DUZELTMESI — last-islem

Her CP'nin direction ISARETINI, mouth cevresindeki yuzun normaliyle uyumlu hale
getirir. Konumu and ekseni DEGISTIRMEZ, only isareti (yonun +/- olmasi).

MEASURED (2026-08-13/14, VAL 100 part, esli part bootstrap):

| populasyon | path | difference | %95 GA |
|---|---|---|---|
| **TANIDIK brand** | olculen | **+0.0195** | **[+0.0030, +0.0378]** KESIN |
| TANIDIK brand | saha | +0.0329 | [−0.0155, +0.0819] |
| hard/gorulmemis (150) | olculen | +0.0023 | notr |
| hard/gorulmemis (150) | saha | −0.0016 | small negatif |

**Kazanc TANIDIK brand populasyonuna OZGUDUR**; gorulmemis markada notr.
Kural, baseline sistemin ZATEN CALISTIGI places yardim ediyor.

WHY HALKA, EN YAKIN TEPE DEGIL. Ilk denemede referans as CP'ye EN
YAKIN tepenin normali kullanildi and arm **−0.0266** verdi. Sebep measured:
that normal deligin **DUVAR** normalidir and kablo girisi eksenine **DIKTIR**
(prediction yonuyle arasindaki angle median **88.9 derece**, 617 CP). Eksene
dik a referansla sign atamak rastgeleye yakindir. Halka (merkeze
`IC_R`..`DIS_R` uzaklikta) hole duvarinin DISINDA, duz yuzeydedir.

DAVRANIS (VAL 617 CP): 95 CP'nin isaretini cevirir (%15.4) --
48 DUZELTIR, 27 BOZAR, 20 notr. Net +21 CP; kararli cevirmelerin %64'u
correct. Kural correct yonde but gurultuludur.

ESIK EKLEMEK YARDIM ETMIYOR: "only guclu celiskide cevir" varyanti
95/105/115/125/140/160 derecede +0.0345/+0.0298/+0.0298/+0.0063/
−0.0063/−0.0047 verdi -- monotonik not, i.e. gurultuye uydurma.
Sade rule is used.
"""
import numpy as np 

IC_R =float (__import__ ("os").environ .get ("HI_IC","3.0"))
DIS_R =float (__import__ ("os").environ .get ("HI_DIS","8.0"))


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def ring_normals (V ,F ,P ,ic_r =IC_R ,dis_r =DIS_R ):
    """each point for AGIZ CEVRESI face normali. (n,3); bulunamazsa sifir."""
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    if not len (P ):
        return np .zeros ((0 ,3 ))
    try :
        import trimesh 
        from scipy .spatial import cKDTree 
        ag =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
        N =np .asarray (ag .vertex_normals ,float )
    except Exception :# noqa: BLE001
        return np .zeros ((len (P ),3 ))
    agac =cKDTree (np .asarray (V ,float ))
    out =np .zeros ((len (P ),3 ))
    for i ,p in enumerate (P ):
        kom =agac .query_ball_point (p ,dis_r )
        if not kom :
            continue 
        d =np .linalg .norm (np .asarray (V ,float )[kom ]-p [None ,:],axis =1 )
        halka =np .asarray (kom )[d >=ic_r ]
        if not len (halka ):
            halka =np .asarray (kom )
        v =N [halka ].mean (0 )
        n =float (np .linalg .norm (v ))
        if n >1e-9 :
            out [i ]=v /n 
    return out 


def duzelt (cps ,V ,F ):
    """CP listesindeki `direction` ISARETLERINI yerinde fixes.

    Konum and axis DEGISMEZ. Halka normali bulunamayan CP'ye DOKUNULMAZ
    (sessizce bozmaktansa dokunmamak yeglenir).
    Doner: (cps, cevrilen_sayisi)
    """
    if not cps :
        return cps ,0 
    P =np .asarray ([c ["point"]for c in cps ],float ).reshape (-1 ,3 )
    D =_birim (np .asarray ([c ["direction"]for c in cps ],
    float ).reshape (-1 ,3 ))
    H =ring_normals (V ,F ,P )
    gecerli =np .linalg .norm (H ,axis =1 )>1e-9 
    cevir =gecerli &(np .sum (D *H ,axis =1 )<0 )
    for i in np .where (cevir )[0 ]:
        cps [i ]["direction"]=(-D [i ]).tolist ()
    return cps ,int (cevir .sum ())


def self_check ():
    """SENTETIK: yonu bilerek TERS cevrilmis a delikte rule fixes mi."""
    import trimesh 
    kutu =trimesh .creation .box (extents =(30 ,30 ,30 ))
    sil =trimesh .creation .cylinder (radius =2.0 ,height =60.0 )
    part =kutu .difference (sil )
    V =np .asarray (part .vertices ,float )
    F =np .asarray (part .faces ,int )
    mouth =np .array ([0.0 ,0.0 ,15.0 ])# +z yuzundeki hole agzi
    dogru =np .array ([0.0 ,0.0 ,1.0 ])# DISARI
    cps =[{"point":mouth .tolist (),"direction":(-dogru ).tolist ()}]
    cps ,n =duzelt (cps ,V ,F )
    yeni =np .asarray (cps [0 ]["direction"],float )
    ok =float (yeni @dogru )>0.9 
    print (f"  ters verilen direction duzeltildi mi: {'EVET'if ok else 'HAYIR'} "
    f"(cevrilen {n })")
    # correct verilen direction BOZULMAMALI
    cps2 =[{"point":mouth .tolist (),"direction":dogru .tolist ()}]
    cps2 ,n2 =duzelt (cps2 ,V ,F )
    ok2 =float (np .asarray (cps2 [0 ]["direction"],float )@dogru )>0.9 
    print (f"  dogru verilen direction korundu mu : {'EVET'if ok2 else 'HAYIR'} "
    f"(cevrilen {n2 })")
    print ("SENTETIK YETENEK:","GECTI"if (ok and ok2 )else "KALDI")
    return ok and ok2 


if __name__ =="__main__":
    self_check ()
