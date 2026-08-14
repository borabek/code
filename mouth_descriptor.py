# -*- coding: utf-8 -*-
"""AGIZ TANIMLAYICILARI: "buraya gercekten tel girer mi?"

WHY: B-rep havuzu robot tavanini 0.3412 -> 0.5748 acti but no selector
kullanamadi (brep-pool-ceiling-acildi-selector-acamadi). Sebep, denenen 58
ozniteligin a agzin TEL GIRISI mi VIDA DELIGI mi ALET YUVASI mi oldugunu
soylememesi. Bu modul that bilgiyi URETIR -- hazir feature aramak instead of.

Hepsi FIZIKSEL and etiketsiz hesaplanabilir:
  radius        silindir yaricapi
  depth       hole uzunlugu (two mouth arasi)
  narinlik       depth / radius (vida deligi sig, tel kanali deep)
  girme          agizdan ICERI serbest path (eksende first carpisma)
  girme_kenar    same but yaricapin %70'i up to YANA kaydirilmis 4 isin -> telin
                 govdesi gercekten giriyor mu (eksende empty, kenarda full = fake)
  erisim         agizdan DISARI serbest path -> robot this agza ulasabiliyor mu
  es_eksen       parcada same yone bakan and yaricapi yakin kardes count
                 (real CP'ler SIRA olusturur; tekil hole cogunlukla vida)
  aralik_duzeni  kardeslerin axis boyu araliklarinin duzenliligi (1 = kusursuz)
  yaricap_yuzde  yaricapin PARCA ICINDEKI yuzdeligi (brand olceginden bagimsiz)

TEZE SADIK: segmentasyon, remesh, `v_o` turetmesi DEGISMEZ. Bunlar only
ADAY PUANLAMA for ek olculerdir.
"""
import numpy as np 

AD =["yaricap","depth","narinlik","girme","girme_kenar","erisim",
"es_eksen","aralik_duzeni","yaricap_yuzde"]


TOPAK =1500 # single cagrida atilacak most extra isin


def _ilk_mesafe (mesh ,O ,Dv ,uzak ):
    """Isin basina first carpisma mesafesi. trimesh imzasindan BAGIMSIZ.

    `intersects_location` (konum, isin_idx, ucgen_idx) returns; isin indeksini
    ACIKCA kullaniriz. `intersects_id`'nin donus order surumler arasi degisiyor
    and sessizce wrong sutunu okumak this projede more before yasandi.

    TOPAKLI (2026-08-11): `multiple_hits=True` each isin for TUM kesisimleri
    returns. Mesh tepesi havuzu acilinca part basina option ~200'den ~5000'e
    output and single cagri 14 MILYON kesisim uretip belleği patlatti (8 payin 4'u
    MemoryError with became). Topaklama sonucu DEGISTIRMEZ -- each isin for most
    small distance alindigi for bolerek hesaplamak same sayiyi gives.
    """
    out =np .full (len (O ),float (uzak ))
    if not len (O ):
        return out 
    for b in range (0 ,len (O ),TOPAK ):
        s =slice (b ,b +TOPAK )
        loc ,ir ,_tri =mesh .ray .intersects_location (
        O [s ],Dv [s ],multiple_hits =True )
        if not len (loc ):
            continue 
        Ob =O [s ]
        d =np .linalg .norm (loc -Ob [ir ],axis =1 )
        alt =out [s ].copy ()
        np .minimum .at (alt ,ir ,d )# loop instead of vektorel
        out [s ]=alt 
    return out 


def _dik_eksenler (ax ):
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (ax @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (ax ,a )
    n =np .linalg .norm (u )
    if n <1e-12 :
        return np .array ([0.0 ,1.0 ,0.0 ]),np .array ([0.0 ,0.0 ,1.0 ])
    u =u /n 
    return u ,np .cross (ax ,u )


def tanimla (P ,D ,met ,mesh ,diag ):
    """(n,9) tanimlayici matrisi. `met` = brep_adaylari(meta=True) ucuncu ogesi."""
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =np .asarray (D ,float ).reshape (-1 ,3 )
    n =len (P )
    X =np .zeros ((n ,len (AD )))
    if n ==0 :
        return X 
    uzak =float (diag )
    rad =np .array ([float (m .get ("radius",m .get ("esd_r",0.0 ))or 0.0 )
    for m in met ])
    der =np .zeros (n )
    for i ,m in enumerate (met ):
        a ,b =m .get ("mouth_a"),m .get ("mouth_b")
        if a is not None and b is not None :
            der [i ]=float (np .linalg .norm (np .asarray (a ,float )-
            np .asarray (b ,float )))
    X [:,0 ]=rad 
    X [:,1 ]=der 
    X [:,2 ]=der /np .maximum (rad ,1e-6 )

    eps =max (1e-3 ,1e-4 *diag )
    X [:,3 ]=_ilk_mesafe (mesh ,P -eps *D ,-D ,uzak )# ICERI
    X [:,5 ]=_ilk_mesafe (mesh ,P +eps *D ,D ,uzak )# DISARI

    ken =np .full (n ,uzak )
    O ,Dv ,sahip =[],[],[]
    for i in range (n ):
        if rad [i ]<=1e-6 :
            continue 
        u ,v =_dik_eksenler (D [i ])
        for off in (u ,-u ,v ,-v ):
            O .append (P [i ]+0.7 *rad [i ]*off -eps *D [i ])
            Dv .append (-D [i ])
            sahip .append (i )
    if O :
        d =_ilk_mesafe (mesh ,np .asarray (O ),np .asarray (Dv ),uzak )
        for j ,i in enumerate (sahip ):
            if d [j ]<ken [i ]:
                ken [i ]=d [j ]
    X [:,4 ]=ken 

    K =np .zeros (n )
    DUZ =np .zeros (n )
    if n >1 :
        C =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-12 )
        cos =np .abs (C @C .T )
        for i in range (n ):
            k =(cos [i ]>0.99 )&(np .abs (rad -rad [i ])<0.3 )
            k [i ]=False 
            K [i ]=float (k .sum ())
            if K [i ]>=2 :
            # SIRA YONU VERIDEN CIKARILIR, secilmez. Ilk surumde eksene dik
            # RASTGELE a `u`ya yansitiyordum; kardesler that eksende cakisinca
            # araliklar sifir cikiyor and DUZENSIZ a array "kusursuz" (1.0)
            # gorunuyordu. Testi this yakaladi.
                off =np .vstack ([P [k ]-P [i ],np .zeros (3 )])
                off =off -off .mean (0 )
                yay =np .linalg .norm (off ,axis =1 ).max ()
                if yay <1e-6 :
                    continue # all of them cakisik -> duzen YOK
                u =np .linalg .svd (off ,full_matrices =False )[2 ][0 ]
                t =np .sort (off @u )
                diff =np .diff (t )
                ort =abs (float (diff .mean ()))
                if len (diff )and ort >1e-6 :
                    DUZ [i ]=1.0 /(1.0 +float (diff .std ())/ort )
        X [:,8 ]=(rad [:,None ]>rad [None ,:]).mean (1 )
    else :
        X [:,8 ]=0.5 
    X [:,6 ]=K 
    X [:,7 ]=DUZ 
    return X 
