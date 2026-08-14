# -*- coding: utf-8 -*-
"""B-kolu: GATE HEDEFI = ROBOT-HAZIR (tespit instead of).

Urunun hedefi ROBOT but gate "this candidate a CP mi" diye egitiliyor. Robot-hazir
olmak lateral<=2mm VE signed angle<=10 gerektirir; gate this ikisini HIC gormuyor.
Hipotez: hedefi degistirmek robot F1'i artirir (precision bedeliyle).

Korpus v4 (g10), MARKA-DISI split. Tek degisken: y.
KILL: brand-disi robot-F1 artmiyorsa arm NULL.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate 
from sklearn .ensemble import RandomForestClassifier 

# ROBOT-HAZIR etiketi turetmeden yeniden uretilir (korpusta absent)
R =pickle .load (open ("results/_der_yeni_g10_n22.pkl","rb"))
X ,y_tes ,y_rob ,mfg ,pid =[],[],[],[],[]
for r in R :
    P ,D =r .get ("P"),r .get ("Pd");G ,Gd =r .get ("G"),r .get ("Gd")
    Xr ,XR =r .get ("X"),r .get ("XR")
    if P is None or G is None or Xr is None or XR is None :continue 
    if not len (P )or not len (G )or len (Xr )!=len (P ):continue 
    P =np .asarray (P ,float );D =np .asarray (D ,float )
    G =np .asarray (G ,float );Gd =np .asarray (Gd ,float )
    F =np .hstack ([np .asarray (Xr ,float ),np .asarray (XR ,float )])
    tol =max (3.0 ,0.06 *r ["diag"])
    for i in range (len (P )):
        dd =np .linalg .norm (G -P [i ],axis =1 )
        j =int (np .argmin (dd ))
        tes =int (dd [j ]<=tol )
        rob =0 
        if tes :
            u =D [i ]/(np .linalg .norm (D [i ])+1e-9 )
            g =Gd [j ]/(np .linalg .norm (Gd [j ])+1e-9 )
            v =G [j ]-P [i ]
            lat =np .linalg .norm (v -(v @u )*u )
            aci =np .degrees (np .arccos (np .clip (float (u @g ),-1 ,1 )))# ISARETLI
            rob =int (lat <=2.0 and aci <=10.0 )
        X .append (F [i ]);y_tes .append (tes );y_rob .append (rob )
        mfg .append (r ["mfg"]);pid .append (r ["pid"])
X =np .asarray (X ,float );y_tes =np .asarray (y_tes );y_rob =np .asarray (y_rob )
mfg =np .asarray (mfg );pid =np .asarray (pid )
M =np .zeros ((len (X ),X .shape [1 ]*2 ))
for u in np .unique (pid ):
    i =np .where (pid ==u )[0 ]
    M [i ]=wire_gate .within_part (X [i ],"zskor")
print (f"candidate {len (M )} | tespit-poz %{100 *y_tes .mean ():.1f} | "
f"ROBOT-poz %{100 *y_rob .mean ():.1f}\n",flush =True )

say =collections .Counter (mfg )
test_mf =[m for m ,c in say .most_common (3 )]
out ={}
for m in test_mf :
    te =mfg ==m ;tr =~te 
    sat ={}
    for ad ,yy in (("hedef=TESPIT",y_tes ),("hedef=ROBOT",y_rob )):
        if yy [tr ].sum ()<50 :continue 
        clf =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =3 ,
        n_jobs =-1 ,random_state =0 ).fit (M [tr ],yy [tr ])
        s =clf .predict_proba (M [te ])[:,1 ]
        p =s >=0.30 
        # ROBOT F1 with degerlendir (urunun hedefi)
        tp =int ((p &(y_rob [te ]==1 )).sum ());fp =int ((p &(y_rob [te ]==0 )).sum ())
        fn =int ((~p &(y_rob [te ]==1 )).sum ())
        sat [ad ]=2 *tp /max (2 *tp +fp +fn ,1 )
    if len (sat )==2 :
        out [m ]=sat 
        print (f"  {m :<6} n={te .sum ():<6} robot-F1: tespit-hedefli {sat ['hedef=TESPIT']:.4f}"
        f" | robot-hedefli {sat ['hedef=ROBOT']:.4f}"
        f" ({sat ['hedef=ROBOT']-sat ['hedef=TESPIT']:+.4f})",flush =True )
if out :
    f =float (np .mean ([v ["hedef=ROBOT"]-v ["hedef=TESPIT"]for v in out .values ()]))
    print (f"\nORTALAMA FARK: {f :+.4f}  ->  {'KOL ACIK'if f >0.01 else 'KOL NULL'}")
    json .dump ({"sonuc":out ,"ortalama_fark":f },open ("results/b_gate_hedefi.json","w"),indent =1 )
    print ("receipt -> results/b_gate_hedefi.json")
