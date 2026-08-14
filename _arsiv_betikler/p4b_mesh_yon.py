# -*- coding: utf-8 -*-
"""P4-b: MESH tabanli ek poz dallari -- B-rep'in ulasamadigi morfolojiler for.

Silindir + duzlemsel opening dallari kahini 0.4002'de tikadi (P4 kapisi 0.82).
Iki dal more, ikisi de MESH'ten (B-rep gerektirmez, therefore SE/NIT like
"silindiri olmayan" parcalarda da works):

  AGIZ-HALKASI DUZLEM NORMALI: adayin cevresindeki surface tepelerine most iyi oturan
    duzlemin normali. Tezin `v_o - v_s`'sinden BAGIMSIZ a kestirim -- that, oturma
    noktasi with mouth merkezini merges; this whereas mouth KONTURUNUN yonelimini kullanir.
  BOSLUK (free-space) YONU: adaydan cikan isinlarin EN UZUN engelsiz gittigi direction.
    Tel giris kanali tanimi geregi most derin bosluktur.

TEZ DEGISMEZ: `v_o` turetmesi, 5 sinif, ~6000 remesh aynen kalir; bunlar SECENEK
produces, secimi P3C/P5 yapar.
"""
import numpy as np 

R_YEREL =4.0 # candidate cevresi yaricapi (mm) -- mouth konturu for
N_ISIN =42 # bosluk taramasi for isin count (Fibonacci kuresi)


def agiz_duzlem_normali (V ,p ,r =R_YEREL ,min_tepe =8 ):
    """Adayin `r` yaricapindaki tepelere most iyi oturan duzlemin normali (PCA)."""
    d =np .linalg .norm (V -p ,axis =1 )
    Q =V [d <=r ]
    if len (Q )<min_tepe :
        return None 
    Q =Q -Q .mean (0 )
    try :
        _u ,_s ,vt =np .linalg .svd (Q ,full_matrices =False )
    except np .linalg .LinAlgError :
        return None 
    return vt [2 ]/(np .linalg .norm (vt [2 ])+1e-12 )# most small varyans yonu


def _fibonacci (n ):
    i =np .arange (n )+0.5 
    phi =np .arccos (1 -2 *i /n )
    th =np .pi *(1 +5 **0.5 )*i 
    return np .stack ([np .cos (th )*np .sin (phi ),np .sin (th )*np .sin (phi ),
    np .cos (phi )],1 )


def bosluk_yonu (mesh ,p ,n =N_ISIN ,max_mm =40.0 ):
    """Adaydan most UZUN engelsiz giden direction. Tel kanali most derin bosluktur."""
    D =_fibonacci (n )
    try :
        loc ,idx ,_tri =mesh .ray .intersects_location (
        np .repeat (p [None ,:],n ,0 ),D ,multiple_hits =False )
    except Exception :
        return None 
    dist_ =np .full (n ,max_mm ,float )
    for L ,i in zip (loc ,idx ):
        m =float (np .linalg .norm (L -p ))
        if m <dist_ [i ]:
            dist_ [i ]=m 
    j =int (np .argmax (dist_ ))
    if dist_ [j ]<2.0 :
        return None 
    return D [j ]/(np .linalg .norm (D [j ])+1e-12 )
