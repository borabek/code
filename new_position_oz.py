# -*- coding: utf-8 -*-
"""L3 + L4 + L5 — KONUM duzeyi YENI BILGI

DIAGNOSIS. Baglayan sey part-ici KONUM siralamasi (NIT konum AUC 0.7053,
gereken 0.944). Hedef fonksiyonuna nisan alan three arm (yerel negatif +0.0034,
lambda +0.0024, array uyeligi +0.0016) kapiyi GECEMEDI. Cephe correct but
araclar zayifti. Geriye YENI BILGI kaliyor.

Uc feature ailesi, ucu de "hangi opening kablo girisi" sorusunun FIZIKSEL
karsiligindan turetildi:

L3 AYNA ESI (5 column). Bir klemenste giris and cikis KARSIT yuzlerdedir --
bugun measured: dense markada GT yonleri two signed cluster, TEK axis
(kip+- = 2.0, kipEks = 1.0). Oyleyse real a girisin AYNI EKSEN
DOGRUSU on a esi olmalidir. Esi olmayan opening muhtemelen giris
degildir (montaj deligi, ray yuvasi, test noktasi).

L4 VIDA CIFTI (4 column). Kablo girisi islevsel as a sikma vidasiyla
eslesir; vida ekseni girise approximately DIKTIR and yakinindadir. S4 cluster
puanlamasi was tried (dustu) but OPEN double geometrisi never denenmedi.

L5 ISIN-TEMAS (4 column). Giristen ICERI atilan isin a KONTAGA varir;
montaj deligi varmaz. STEP rengine gerek YOK -- segmentasyon ciktisinda
CONTACT sinifi already present. (`c_metal` single basina OLU, isin ekseni single
basina OLU; this ikisinin BILESIMI ayri a sorudur.)

Hepsi GT KULLANMAZ.
"""
import numpy as np 
from scipy .spatial import cKDTree 

AYNA_MIN ,AYNA_MAKS =4.0 ,80.0 # es aralik (mm)
AYNA_TOL =2.5 # dogruya dik deviation (mm)
VIDA_R =14.0 # vida arama yaricapi (mm)
VIDA_DIK =25.0 # 90 dereceden izin verilen deviation
ISIN_N =5 # konum basina isin
ISIN_ACI =15.0 
ISIN_CAP =30.0 


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def konum_yonu (idx ,YD ,tek ):
    """each benzersiz konum for TEMSILI direction (secenek yonlerinin ana ekseni).

    ISARETSIZ ana axis: +v and -v same sayilir, because measured -- dense
    parcada yonler two signed kumede but TEK eksende.
    """
    out =np .zeros ((len (tek ),3 ))
    sira =np .argsort (idx ,kind ="stable")
    idx_s =idx [sira ]
    sinir =np .flatnonzero (np .diff (idx_s ))+1 
    for t_ ,p in enumerate (np .split (sira ,sinir )):
        Y =_birim (YD [p ])
        if len (Y )==1 :
            out [t_ ]=Y [0 ]
            continue 
            # unsigned mean: most large ozvektor (direction tensoru)
        T =Y .T @Y 
        w ,V_ =np .linalg .eigh (T )
        out [t_ ]=V_ [:,-1 ]
    return out 


def ayna_esi (Pk ,Dk ):
    """L3: same axis dogrusu on es present mi. (n,5)"""
    n =len (Pk )
    out =np .zeros ((n ,5 ),np .float32 )
    if n <2 :
        return out 
    for i in range (n ):
        v =Pk -Pk [i ][None ,:]
        uz =np .linalg .norm (v ,axis =1 )
        al =v @Dk [i ]
        dik =np .sqrt (np .maximum (uz **2 -al **2 ,0.0 ))
        uy =(dik <=AYNA_TOL )&(np .abs (al )>=AYNA_MIN )&(np .abs (al )<=AYNA_MAKS )
        uy [i ]=False 
        if not uy .any ():
            out [i ]=(0.0 ,0.0 ,0.0 ,0.0 ,0.0 )
            continue 
        j =np .where (uy )[0 ]
        en =j [int (np .argmin (np .abs (al [j ])))]
        # esin yonu bizimkiyle same eksende mi (unsigned)
        hiz =abs (float (Dk [en ]@Dk [i ]))
        out [i ]=(1.0 ,float (abs (al [en ])),float (dik [en ]),
        float (len (j )),hiz )
    return out 


def vida_cifti (Pk ,Dk ):
    """L4: yakinda DIK yonlu a opening present mi. (n,4)"""
    n =len (Pk )
    out =np .zeros ((n ,4 ),np .float32 )
    if n <2 :
        return out 
    agac =cKDTree (Pk )
    for i in range (n ):
        kom =np .asarray (agac .query_ball_point (Pk [i ],VIDA_R ),int )
        kom =kom [kom !=i ]
        if not len (kom ):
            continue 
        cos =np .abs (np .clip (Dk [kom ]@Dk [i ],-1 ,1 ))
        deviation =np .abs (90.0 -np .degrees (np .arccos (cos )))
        dik =deviation <=VIDA_DIK 
        if not dik .any ():
            out [i ]=(0.0 ,0.0 ,float (len (kom )),float (deviation .min ()))
            continue 
        dk =kom [dik ]
        uz =np .linalg .norm (Pk [dk ]-Pk [i ],axis =1 )
        out [i ]=(1.0 ,float (uz .min ()),float (len (kom )),
        float (deviation [dik ].min ()))
    return out 


def isin_temas (Pk ,Dk ,isinci ,V ,p_temas ):
    """L5: iceri atilan isin KONTAGA variyor mu. (n,4)"""
    n =len (Pk )
    out =np .zeros ((n ,4 ),np .float32 )
    if not n :
        return out 
    agac =cKDTree (V )
    for i in range (n ):
        d0 =Dk [i ]
        a =np .array ([1.0 ,0.0 ,0.0 ])
        if abs (d0 @a )>0.9 :
            a =np .array ([0.0 ,1.0 ,0.0 ])
        e1 =_birim (np .cross (d0 ,a ).reshape (1 ,3 ))[0 ]
        e2 =np .cross (d0 ,e1 )
        t =np .radians (ISIN_ACI )
        fi =np .linspace (0 ,2 *np .pi ,ISIN_N -1 ,endpoint =False )
        direction =np .vstack ([d0 ,_birim (
        np .cos (t )*d0 [None ,:]+
        np .sin (t )*(np .cos (fi )[:,None ]*e1 [None ,:]+
        np .sin (fi )[:,None ]*e2 [None ,:]))])
        # IKI YONE de bak: axis isareti bilinmiyor, ici which is taraf hangisiyse
        direction =np .vstack ([direction ,-direction ])
        kok =np .tile (Pk [i ],(len (direction ),1 ))+0.15 *direction 
        try :
            yer ,ind_r =isinci .intersects_location (
            kok ,direction ,multiple_hits =False )[:2 ]
        except Exception :# noqa: BLE001
            continue 
        if not len (yer ):
            continue 
        uz =np .linalg .norm (yer -kok [ind_r ],axis =1 )
        ic =uz <=ISIN_CAP 
        if not ic .any ():
            continue 
        pt =p_temas [agac .query (yer [ic ])[1 ]]
        out [i ]=(float (pt .max ()),float (pt .mean ()),
        float (uz [ic ][int (np .argmax (pt ))]),float (ic .mean ()))
    return out 
