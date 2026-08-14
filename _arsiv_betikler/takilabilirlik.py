# -*- coding: utf-8 -*-
"""K1.1: TAKILABILIRLIK TESTI -- fizik ONERIR, ogrenme ELER.

BUGUNKU MIMARI: ogrenme onerir -> gate eler. Gorulmemis ureticide ONERI asamasi
cokuyor (ADAY_YOK, GT kaybinin %32'si) and direction secimi cokuyor (angle hatasi IKI TEPELI:
%43.9 correct, %51.1 tamamen wrong, ortada rafine edilebilir bant only %5).

BU MODUL TERSINE CEVIRIR: fizik onerir -> ogrenme eler.

PRIMITIF -- SILINDIR YURUTME: a (point, direction) cifti for, yaricapi r which is a silindiri
katiya DEGMEDEN d mm sokabiliyor muyuz? Sokabiliyorsak orasi FIZIKSEL as a kablo
girisidir. Bu, robot metriginin BIREBIR tanimidir (ferrul giriyor mu?).

WHY TRANSFER GARANTISI YAPISAL: no sey EGITILMIYOR. "Gorulmemis manufacturer" diye a
kavram absent -- same geometri each markada same sonucu gives.

OLU KAYITLI "SAF CAD DEDEKTORU"NDEN FARKI: orada geometri DECISION veriyordu and ceiling
0.332'de tikanmisti. Burada geometri only ONERIYOR; "this hole kablo girisi mi, alet
yuvasi mi, montaj deligi mi" ayrimini yine OGRENILMIS model does. Bu ayrim SEMANTIK
and semantik geometriden more iyi transfer eder.

TEZ: `v_o` turetmesi, 5 sinif and ~6000 remesh AYNEN kalir. Bu a ADAY URETECIDIR and
dagitilirsa tez-yolu-single-basina count YAN YANA raporlanir.
"""
import numpy as np 

R_ADAY =(0.9 ,1.4 ,2.0 ,2.8 ,4.0 )# ferrul yaricaplari (mm) -- 1.5-8mm^2 iletken bandi
DERINLIK =6.0 # this up to mm engelsiz girebilmeli
N_ISIN =8 # silindir kesitini orneklemek for cevre isini


def _fibonacci_yon (n ):
    i =np .arange (n )+0.5 
    phi =np .arccos (1 -2 *i /n )
    th =np .pi *(1 +5 **0.5 )*i 
    return np .stack ([np .cos (th )*np .sin (phi ),np .sin (th )*np .sin (phi ),np .cos (phi )],1 )


def _dik_taban (d ):
    a =np .array ([0.0 ,0 ,1.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([1.0 ,0 ,0 ])
    e1 =np .cross (d ,a );e1 /=(np .linalg .norm (e1 )+1e-12 )
    return e1 ,np .cross (d ,e1 )


def silindir_gecer_mi (mesh ,p ,d ,r ,depth =DERINLIK ,n_isin =N_ISIN ):
    """Yaricapi r which is silindir, p'den d yonunde `depth` mm engelsiz gidebiliyor mu?

    Silindirin CEVRESINDEN n_isin count paralel isin atilir; all of them `depth` mm
    engelsiz giderse silindir gecer. Merkez isini de dahil edilir.
    """
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-12 )
    e1 ,e2 =_dik_taban (d )
    ac =np .linspace (0 ,2 *np .pi ,n_isin ,endpoint =False )
    O =[np .asarray (p ,float )]
    for a in ac :
        O .append (np .asarray (p ,float )+r *(np .cos (a )*e1 +np .sin (a )*e2 ))
    O =np .asarray (O ,float )
    D =np .repeat (d [None ,:],len (O ),0 )
    try :
        loc ,idx ,_t =mesh .ray .intersects_location (O ,D ,multiple_hits =False )
    except Exception :
        return False ,0.0 
    mes =np .full (len (O ),np .inf )
    for L ,i in zip (loc ,idx ):
        m =float (np .linalg .norm (L -O [i ]))
        if m <mes [i ]:
            mes [i ]=m 
    en_kisa =float (np .min (mes ))
    return bool (en_kisa >=depth ),en_kisa 


def en_buyuk_gecen_yaricap (mesh ,p ,d ,depth =DERINLIK ):
    """Bu (point, direction) for gecebilen EN BUYUK ferrul yaricapi. Gecmiyorsa 0."""
    en =0.0 
    for r in R_ADAY :
        ok ,_ =silindir_gecer_mi (mesh ,p ,d ,r ,depth )
        if ok :
            en =r 
        else :
            break 
    return en 


def yon_ara (mesh ,p ,yonler =None ,depth =DERINLIK ,n_yon =26 ):
    """Bir NOKTA for most iyi giris yonu: most large ferrulu most derine sokan direction.

    Doner: (direction, radius, depth) -- none of them gecmezse (None, 0, 0).
    """
    Y =_fibonacci_yon (n_yon )if yonler is None else np .asarray (yonler ,float )
    en =(None ,0.0 ,0.0 )
    for d in Y :
        r =en_buyuk_gecen_yaricap (mesh ,p ,d ,depth )
        if r >en [1 ]:
            _ok ,mes =silindir_gecer_mi (mesh ,p ,d ,r ,depth )
            en =(d /(np .linalg .norm (d )+1e-12 ),r ,mes )
    return en 


def kanal_var_mi (mesh ,p ,d ,r ,pre_ =1.0 ,post_ =5.0 ,n_ac =8 ):
    """FIZIKSEL SORU: this noktada, this yonde yaricapi r which is a KANAL present mi?

    Iki wrong kurgudan after correct formulasyon (2026-08-07):
      * v1: GT noktasindan ISIN at -> GT'lerin ~%47'si GOVDE ICINDE oldugu for no
        direction gecmiyordu (ONV 2/47, SE 0/36).
      * v2: parcanin ceyrek kosegeni up to DISARIDAN basla -> telin before uzun a serbest
        koridoru gecmesini sart kosuyor; dense dizide komsu kutuplar kapatiyor (22/184).
      * v3 (this): channel LOKAL a ozelliktir. Eksen along [-before, +after] araliginda
        ornekle; each ornekte eksene DIK r yaricapli diskin BOS olmasini iste.
        Uzaktaki engeller onemsiz -- sorulan sey "ferrul buraya oturur mu".
    """
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-12 )
    e1 ,e2 =_dik_taban (d )
    ac =np .linspace (0 ,2 *np .pi ,n_ac ,endpoint =False )
    cev =np .stack ([np .cos (a )*e1 +np .sin (a )*e2 for a in ac ],0 )
    ts =np .linspace (-pre_ ,post_ ,max (3 ,int ((pre_ +post_ )/0.8 )))
    O ,D =[],[]
    for t in ts :
        m =np .asarray (p ,float )+t *d 
        for c in cev :
            O .append (m );D .append (c )# eksene DIK isin: r inside engel present mi
    O =np .asarray (O ,float );D =np .asarray (D ,float )
    try :
        loc ,idx ,_t =mesh .ray .intersects_location (O ,D ,multiple_hits =False )
    except Exception :
        return False 
    mes =np .full (len (O ),np .inf )
    for L ,i in zip (loc ,idx ):
        v =float (np .linalg .norm (L -O [i ]))
        if v <mes [i ]:
            mes [i ]=v 
    return bool (np .min (mes )>=r )


def kanal_yon_ara (mesh ,p ,n_yon =42 ,r_min =0.9 ):
    """Bu noktada EN BUYUK kanali veren direction. Doner: (direction, radius)."""
    Y =_fibonacci_yon (n_yon )
    en =(None ,0.0 )
    for d in Y :
        for r in R_ADAY :
            if r <r_min :
                continue 
            if kanal_var_mi (mesh ,p ,d ,r ):
                if r >en [1 ]:
                    en =(d ,r )
            else :
                break 
    return en 
