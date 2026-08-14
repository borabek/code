# -*- coding: utf-8 -*-
"""DERINLIK PROFILI IMZASI: "this hole tel girisi mi, vida mi, alet yuvasi mi?"

FIZIK. Uc hole turunun axis along YARICAP PROFILI farklidir:
  tel girisi : agizda huni, after SABIT a bore -> profil duz plato
  vida deligi: konik daralma / dise bagli dalgalanma -> profil DUSER
  alet yuvasi: sig and GENIS -> profil kisa surede biter

Mevcut 9 mouth olcusu (`mouth_descriptor`) bunu however kabaca yakaliyor: radius
TEK a sayidir and depth two mouth arasi mesafedir. Burada axis along
BIRDEN COK derinlikte serbest radius olculur and profil a imza as verilir.

OLCUM. Her depth `t` for, eksene DIK duzlemde 8 yonde isin atilir and first
carpisma mesafesinin MEDYANI that derinlikteki serbest yaricaptir. Isinlar single
cagrida toplu atilir (`mouth_descriptor._ilk_mesafe` topakli).

Sutunlar (12):
  r0..r5      0.5/1/2/4/8/16 mm derinlikte serbest radius
  plato       r2..r5 arasindaki degisim katsayisi (KUCUK = sabit bore = tel)
  daralma     (r0 - r5) / max(r0, eps)   (BUYUK = konik = vida)
  depth    profilin bittigi depth (radius r0'in %30'unun altina dustugu)
  agiz_bagil  r0 / part kosegeni  (scale bagimsiz mouth genisligi)
  duzluk      |r_i - medyan| ortalamasi
  egim        r5 - r0
"""
import numpy as np 

DERINLIKLER =(0.5 ,1.0 ,2.0 ,4.0 ,8.0 ,16.0 )
N_ISIN =8 
OZ_AD =([f"r{i }"for i in range (len (DERINLIKLER ))]+
["plato","daralma","depth","agiz_bagil","duzluk","egim"])


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


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


def profil (P ,D ,mesh ,diag ):
    """(n, len(OZ_AD)) imza matrisi. `D` agizdan DISARI bakan direction."""
    import mouth_descriptor as AT 
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =_birim (D )
    n =len (P )
    X =np .zeros ((n ,len (OZ_AD )))
    if not n or mesh is None :
        return X 
    eps =max (1e-3 ,1e-4 *diag )
    # each candidate x each depth x N_ISIN dik direction
    O ,Dv =[],[]
    for i in range (n ):
        u ,v =_dik_eksenler (D [i ])
        ac =np .linspace (0 ,2 *np .pi ,N_ISIN ,endpoint =False )
        direction =(np .cos (ac )[:,None ]*u [None ]+np .sin (ac )[:,None ]*v [None ])
        for t in DERINLIKLER :
            merkez =P [i ]-t *D [i ]# ICERI correct
            O .append (np .repeat (merkez [None ],N_ISIN ,axis =0 )+eps *direction )
            Dv .append (direction )
    O =np .vstack (O )
    Dv =np .vstack (Dv )
    m =AT ._ilk_mesafe (mesh ,O ,Dv ,diag )
    R =m .reshape (n ,len (DERINLIKLER ),N_ISIN )
    r =np .median (R ,axis =2 )# medyan serbest radius
    X [:,:len (DERINLIKLER )]=r 
    r0 =np .maximum (r [:,0 ],1e-6 )
    plato_bolge =r [:,2 :]
    ort =np .maximum (plato_bolge .mean (1 ),1e-6 )
    X [:,len (DERINLIKLER )+0 ]=plato_bolge .std (1 )/ort 
    X [:,len (DERINLIKLER )+1 ]=(r [:,0 ]-r [:,-1 ])/r0 
    # profilin bittigi depth: radius r0'in %30'unun altina dustugu first t
    dus =r <(0.30 *r0 [:,None ])
    ilk =np .where (dus .any (1 ),np .argmax (dus ,axis =1 ),len (DERINLIKLER )-1 )
    X [:,len (DERINLIKLER )+2 ]=np .asarray (DERINLIKLER ,float )[ilk ]
    X [:,len (DERINLIKLER )+3 ]=r0 /max (diag ,1e-6 )
    X [:,len (DERINLIKLER )+4 ]=np .abs (
    r -np .median (r ,axis =1 ,keepdims =True )).mean (1 )
    X [:,len (DERINLIKLER )+5 ]=r [:,-1 ]-r [:,0 ]
    return X 
