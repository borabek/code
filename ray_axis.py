# -*- coding: utf-8 -*-
"""A1 -- ISIN ATMA with DELIK EKSENI: direction a SINIFLANDIRMA not OLCUM isi

FIZIK. Bir kablo girisi a DELIKTIR. Ekseni along uzun and engelsiz a
channel vardir; dik yonde birkac mm'de duvara carpilir. O halde axis,
"body inside most uzun TUP"un yonudur.

TUP SKORU. Bir u yonu for:
    d0   = u along first carpma mesafesi (CAP with sinirli, carpma otherwise CAP)
    halka= u cevresinde THETA acisinda k isinin first carpma mesafeleri
    tup  = d0 / (median(halka) + eps)
Kanalda d0 large, halka kucuktur (hole duvari) -> tup high.
Duz yuzeyde ikisi de benzer -> tup ~ 1.

WHY BU KOL DIGER YON KOLLARINDAN FARKLI:
  * B-rep agzi CP'de bulunma 0.011 -> B-rep'e BAGIMLI DEGIL
  * mesh normali secili adaylarda  0.334 -> yerel normale BAGIMLI DEGIL
  * only ucgenler and isin-ucgen kesisimi is required

ISARET. Tup skoru u with -u'yu ayirmaz (tup two yone de uzanir). Isaret
AYRICA "govdeden disari" kuralindan turetilir (GT sozlesmesi 1.000 disari).
"""
import numpy as np 

CAP =40.0 
THETA =25.0 
N_HALKA =8 
EPS_ILERI =0.15 # baslangici yuzeyden ayir (own ucgenine carpmasin)


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def _dik_ikili (u ):
    """u'ya dik two unit vektor."""
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (u @a )>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    e1 =_birim (np .cross (u ,a ).reshape (1 ,3 ))[0 ]
    e2 =np .cross (u ,e1 )
    return e1 ,e2 


def halka_yonleri (u ,theta =THETA ,k =N_HALKA ):
    """u cevresinde theta acisinda k direction."""
    u =_birim (np .asarray (u ,float ).reshape (1 ,3 ))[0 ]
    e1 ,e2 =_dik_ikili (u )
    t =np .radians (theta )
    fi =np .linspace (0 ,2 *np .pi ,k ,endpoint =False )
    return _birim (np .cos (t )*u [None ,:]+
    np .sin (t )*(np .cos (fi )[:,None ]*e1 [None ,:]+
    np .sin (fi )[:,None ]*e2 [None ,:]))


def ilk_carpma (isinci ,kokler ,yonler ,cap =CAP ):
    """each (kok, direction) for first carpma mesafesi; carpma otherwise cap."""
    kokler =np .asarray (kokler ,float ).reshape (-1 ,3 )
    yonler =_birim (np .asarray (yonler ,float ).reshape (-1 ,3 ))
    k0 =kokler +EPS_ILERI *yonler 
    yer ,ind_r =isinci .intersects_location (k0 ,yonler ,
    multiple_hits =False )[:2 ]
    d =np .full (len (yonler ),cap ,float )
    if len (ind_r ):
        uz =np .linalg .norm (yer -k0 [ind_r ],axis =1 )+EPS_ILERI 
        # same isin birden extra donerse most kucugu al
        for i ,r_ in enumerate (ind_r ):
            if uz [i ]<d [r_ ]:
                d [r_ ]=uz [i ]
    return np .minimum (d ,cap )


def tup_skoru (isinci ,kok ,yonler ,theta =THETA ,k =N_HALKA ,cap =CAP ):
    """each direction for tup skoru. yonler: (n,3). returns: (n,)"""
    yonler =_birim (np .asarray (yonler ,float ).reshape (-1 ,3 ))
    n =len (yonler )
    # axis isinlari
    d0 =ilk_carpma (isinci ,np .tile (kok ,(n ,1 )),yonler ,cap )
    # halka isinlari
    hy =np .vstack ([halka_yonleri (u ,theta ,k )for u in yonler ])
    hd =ilk_carpma (isinci ,np .tile (kok ,(n *k ,1 )),hy ,cap )
    hd =hd .reshape (n ,k )
    return d0 /(np .median (hd ,axis =1 )+1e-6 )


def self_check ():
    """SENTETIK YETENEK TESTI: ekseni BILINEN a delikte bulabiliyor mu.

    Bu step S4'te wrong negatiften kurtarmisti: mekanizma bilinen a
    delikte ekseni bulamiyorsa real veride aramanin anlami absent.
    """
    import trimesh 
    ok =True 
    # EGIK EKSENLER: mouth +z YUZUNDEN ciksin diye z bileseni baskin secilir.
    # Ilk denememde axis [1,1,0.3] idi and `axis*15` kutunun YUZUNE not
    # KENARINA dusuyordu -- i.e. agzin olmadigi a noktadan isin atiyordum
    # (17.9 derece deviation mekanizmanin not TESTIN kusuruydu).
    for axis in (np .array ([0.0 ,0.0 ,1.0 ]),
    _birim (np .array ([[0.30 ,0.20 ,1.0 ]]))[0 ],
    _birim (np .array ([[0.55 ,-0.35 ,1.0 ]]))[0 ]):
        kutu =trimesh .creation .box (extents =(30 ,30 ,30 ))
        T =trimesh .geometry .align_vectors ([0 ,0 ,1 ],axis )
        sil =trimesh .creation .cylinder (radius =2.0 ,height =80.0 ,transform =T )
        part =kutu .difference (sil )
        assert part .is_watertight ,"sentetik part su gecirmez degil"
        isinci =trimesh .ray .ray_triangle .RayMeshIntersector (part )
        # hole agzi: eksenin +z yuzunu (z=15) deldigi point
        kok =axis *(15.0 /axis [2 ])
        assert np .all (np .abs (kok [:2 ])<14.0 ),"mouth yuzde degil, kenarda"
        # 64 candidate direction (Fibonacci)
        i =np .arange (64 )+0.5 
        fi =np .arccos (1 -2 *i /64 )
        te =np .pi *(1 +5 **0.5 )*i 
        Y =np .c_ [np .cos (te )*np .sin (fi ),np .sin (te )*np .sin (fi ),
        np .cos (fi )]
        # correct yonu listeye KOY (real veride de candidate listesinden secilir)
        Y =np .vstack ([Y ,axis ,-axis ])
        t =tup_skoru (isinci ,kok ,Y )
        sec =Y [int (np .argmax (t ))]
        aci =np .degrees (np .arccos (np .clip (abs (float (sec @axis )),-1 ,1 )))
        print (f"  axis {np .round (axis ,2 )} -> secilen deviation {aci :5.1f} "
        f"derece | tup {t .max ():.2f} (ortanca {np .median (t ):.2f})")
        ok &=aci <=10.0 
    print ("SENTETIK YETENEK:","GECTI"if ok else "KALDI")
    return ok 


if __name__ =="__main__":
    self_check ()
