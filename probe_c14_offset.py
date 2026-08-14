# -*- coding: utf-8 -*-
"""C14 ON SONDA: YANAL error oznitelilerden TAHMIN EDILEBILIR mi?

Kolu kurmadan before tavani olc (this haftanin disiplini). Donusumun baglayici kisiti
YANAL konum ([[donusum-lateral-bagli-angle-not]]) and hedeflerin %98'i mesh'te
ULASILABILIR ([[mesh-ceiling-not-lateral-model-hatasi]]) -- i.e. error ogrenilebilir
OLABILIR. Bu probe "may be"i sinar.

KURULUM:
  * each matched candidate for hedef = GT - candidate, YONE DIK duzleme izdusurulmus (2 size)
  * feature = urunun 58 sutunu (X22 + XR)
  * split MARKA-DISI (rastgele CV this problemde HER ZAMAN siser)
  * criterion: lateral hatanin medyani DUSTU mu, and <=2mm passing ratio ARTTI mi
KILL: brand-disi <=2mm orani artmiyorsa arm OLU.
"""
import os ,sys ,pickle ,collections 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
from sklearn .ensemble import RandomForestRegressor 

R =pickle .load (open ("results/_der_yeni_g10_n22.pkl","rb"))
X ,Y ,MF =[],[],[]
for r in R :
    P ,D =r .get ("P"),r .get ("Pd")
    G ,Gd =r .get ("G"),r .get ("Gd")
    if P is None or G is None or not len (P )or not len (G ):
        continue 
    Xr =r .get ("X");XR =r .get ("XR")
    if Xr is None or XR is None or len (Xr )!=len (P ):
        continue 
    P =np .asarray (P ,float );D =np .asarray (D ,float )
    G =np .asarray (G ,float )
    F =np .hstack ([np .asarray (Xr ,float ),np .asarray (XR ,float )])
    tol =max (3.0 ,0.06 *r ["diag"])
    for i in range (len (P )):
        j =int (np .argmin (np .linalg .norm (G -P [i ],axis =1 )))
        v =G [j ]-P [i ]
        if np .linalg .norm (v )>tol :# eslesmemis candidate -> hedef tanimsiz
            continue 
        u =D [i ]/(np .linalg .norm (D [i ])+1e-9 )
        # YONE DIK bilesen: metrik only bunu cezalandirir
        lat =v -(v @u )*u 
        # yerel dik cerceve (e1,e2)
        a =np .array ([1.0 ,0 ,0 ])if abs (u [0 ])<0.9 else np .array ([0 ,1.0 ,0 ])
        e1 =np .cross (u ,a );e1 /=np .linalg .norm (e1 )+1e-9 
        e2 =np .cross (u ,e1 )
        X .append (F [i ]);Y .append ([lat @e1 ,lat @e2 ]);MF .append (r ["mfg"])
X =np .asarray (X ,float );Y =np .asarray (Y ,float );MF =np .asarray (MF )
n0 =np .linalg .norm (Y ,axis =1 )
print (f"matched candidate {len (X )} | manufacturer {len (set (MF ))}")
print (f"TABAN lateral error: medyan {np .median (n0 ):.3f}mm | <=2mm %{100 *(n0 <=2 ).mean ():.1f}\n")

# MARKA-DISI split: at most parcali 3 markayi TEST yap
say =collections .Counter (MF )
test_mf =[m for m ,_ in say .most_common (3 )]
print (f"TEST markalari (brand-disi): {test_mf }\n")
for m in test_mf :
    te =MF ==m ;tr =~te 
    if te .sum ()<50 or tr .sum ()<200 :
        continue 
    reg =RandomForestRegressor (n_estimators =200 ,min_samples_leaf =5 ,
    n_jobs =-1 ,random_state =0 ).fit (X [tr ],Y [tr ])
    pred =reg .predict (X [te ])
    remaining =np .linalg .norm (Y [te ]-pred ,axis =1 )
    ham =np .linalg .norm (Y [te ],axis =1 )
    print (f"  {m :<6} n={te .sum ():<6} medyan {np .median (ham ):.3f} -> "
    f"{np .median (remaining ):.3f}mm | <=2mm %{100 *(ham <=2 ).mean ():.1f} -> "
    f"%{100 *(remaining <=2 ).mean ():.1f}",flush =True )
