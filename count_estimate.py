# -*- coding: utf-8 -*-
"""CP ADEDINI GEOMETRIDEN TAHMIN ET — metadata YOK.

WHY. `robot_cp.extract_highcp` dense parcada baseline yoldan very more iyi
(OOF F1 0.807) but **`cp_count` ISTER** -- ureticinin CP count, i.e.
METADATA. Kod okundu: `cp_count` only IKI places is used:
  1. `highcp_selector.rank_feats` inside `nratio = N/n` (single feature)
  2. last `[:N]` kirpmasi
Havuz uretimi and lattice/gate oznitelikleri **N'den bagimsiz**. Yani
metadata bagimliligini kaldirmak for single gereken a **N tahmini**.

FIKIR. Klemensin CP'leri DUZENLI BIR IZGARADA durur. Izgara adimini
(pitch) and yayilimi olcersen site sayisini sayabilirsin -- this tamamen
GEOMETRIKTIR, no metadata bilgisi gerektirmez. Gereken makine already
`highcp_selector.lattice_feats` inside present (SVD with surface eksenleri +
`gridfit` pitch tahmini); here count tahminine uyarlaniyor.

ILK KAPI (this dosyanin `__main__`'i). Tahminci GT NOKTALARINDAN adedi geri
bulabiliyor mu? Bulamiyorsa gurultulu adaydan never bulamaz. Bu kontrol
CIKARIM GEREKTIRMEZ, saniyeler surer and kolu bosuna kosmaktan kurtarir.
"""
import os 
import sys 

import numpy as np 


def _eksenler (P ):
    """noktalarin yayildigi two ana axis (SVD) + izdusumler."""
    Q =P -P .mean (0 )
    _ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False )
    u ,v =Vt [0 ],Vt [1 ]
    return P @u ,P @v 


def _pitch (pp ,span ,min_oran =0.02 ):
    """a eksende IZGARA ADIMI: pozitif bosluklarin ortancasi."""
    pos =np .sort (pp )
    g =np .diff (pos )
    g =g [g >min_oran *span ]
    if not len (g ):
        return None 
    return float (np .median (g ))


def estimate (P ,ws =None ,min_n =1 ):
    """(n,3) candidate/point kumesinden CP ADEDINI prediction et. metadata YOK.

    Yontem: two ana eksende izgara adimi bulunur, spread/step with each
    eksendeki site count cikarilir, ikisinin CARPIMI site sayisidir.
    Tek sirali parcalarda ikinci eksende step bulunamaz -> 1 alinir.
    Doner: (N_tahmin, teshis_sozlugu)
    """
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    n =len (P )
    if n <2 :
        return max (n ,min_n ),{"reason":"very az nokta"}
    pu ,pv =_eksenler (P )
    span_u =float (pu .max ()-pu .min ())
    span_v =float (pv .max ()-pv .min ())
    span =max (span_u ,span_v ,1.0 )
    hu =_pitch (pu ,span )
    hv =_pitch (pv ,span )
    # each eksende site count = spread / step + 1
    nu =int (round (span_u /hu ))+1 if hu else 1 
    nv =int (round (span_v /hv ))+1 if hv else 1 
    # DEJENERE DURUM: a eksende spread adimdan kucukse single order demektir
    if hu and span_u <0.5 *hu :
        nu =1 
    if hv and span_v <0.5 *hv :
        nv =1 
    N =max (int (nu *nv ),min_n )
    return N ,{"nu":nu ,"nv":nv ,"pitch_u":hu ,"pitch_v":hv ,
    "span_u":span_u ,"span_v":span_v ,"n_nokta":n }


def _kapi ():
    """ILK KAPI: GT noktalarindan adedi geri bulabiliyor muyuz?"""
    os .environ .setdefault ("BA_ALLOW_SEEN","1")
    sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
    import canonical_d7 as K 
    import d6_record 
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ger ,tah ,yogun_ger ,yogun_tah =[],[],[],[]
    for pid ,r in kay .items ():
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        if len (G )<1 :
            continue 
        N ,_ =estimate (G )
        ger .append (len (G ))
        tah .append (N )
        if len (G )>=11 :# dense regime (hicp_threshold)
            yogun_ger .append (len (G ))
            yogun_tah .append (N )
    ger =np .array (ger ,float )
    tah =np .array (tah ,float )
    yg =np .array (yogun_ger ,float )
    yt =np .array (yogun_tah ,float )
    print (f"GT'den adet tahmini -- {len (ger )} part")
    print (f"  TAM isabet        : {100 *(tah ==ger ).mean ():5.1f}%")
    print (f"  +/-1 icinde       : {100 *(np .abs (tah -ger )<=1 ).mean ():5.1f}%")
    print (f"  +/-2 icinde       : {100 *(np .abs (tah -ger )<=2 ).mean ():5.1f}%")
    print (f"  ortanca mutlak error: {np .median (np .abs (tah -ger )):.1f} CP")
    if len (yg ):
        print (f"\nYOGUN parts (GT>=11) -- {len (yg )} part  <- ASIL HEDEF")
        print (f"  TAM isabet        : {100 *(yt ==yg ).mean ():5.1f}%")
        print (f"  +/-1 icinde       : {100 *(np .abs (yt -yg )<=1 ).mean ():5.1f}%")
        print (f"  +/-2 icinde       : {100 *(np .abs (yt -yg )<=2 ).mean ():5.1f}%")
        print (f"  ortanca mutlak error: {np .median (np .abs (yt -yg )):.1f} CP")
        print (f"  ortanca goreli error: "
        f"{100 *np .median (np .abs (yt -yg )/yg ):.1f}%")


if __name__ =="__main__":
    _kapi ()
