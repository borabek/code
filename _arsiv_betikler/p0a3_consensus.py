# -*- coding: utf-8 -*-
"""P0-a3: KONSENSUS-EKSEN hipotezinin KESIN testi (P0-a2'de TP direction-yogunlugu 0.636 vs FP 0.308 output).
IKI TRAP present, ikisini de kapatiyorum:
 (T1) resultant length KUCUK ORNEKLEMDE SISER (TP/part ~2.9, FP ~16.7) -> BOYUT-ESLESMIS null sart.
 (T2) 'sinyal present' yetmez; 13 feature'in USTUNE EKLIYOR mu (gate already dolayli yakaliyor may be).
Deploy-edilebilir feature'lar (label GEREKMEZ, inference'ta hesaplanabilir):
  f1 cos(dir_i, sayim-baskin axis)   f2 cos(dir_i, gate-skor-agirlikli konsensus)
  f3 own axis-kumemin buyuklugu (pay)   f4 own kumemin mean gate skoru
DECISION: (a) size-eslesmis null'i geciyor mu, (b) 13-feature AUC'sine ne katiyor."""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 
from p0a_axis_discord import cluster_axes 

POOL =sys .argv [1 ]if len (sys .argv )>1 else "results/wei_aggr_pool.json"
ANG =20.0 
rng =np .random .RandomState (0 )
d =json .load (open (POOL ))

# ---------- T1: size-eslesmis null ----------
obs ,nul =[],[]
for pid ,v in d .items ():
    D =np .array (v ["dir"],float );y =np .array (v ["y"],int )
    if y .sum ()<2 or len (y )<6 :continue 
    D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )
    k =int (y .sum ())
    obs .append (float (np .linalg .norm (D [y ==1 ].mean (0 ))))
    nul .append (float (np .mean ([np .linalg .norm (D [rng .choice (len (D ),k ,replace =False )].mean (0 ))for _ in range (200 )])))
obs ,nul =np .array (obs ),np .array (nul )
print (f"=== T1: TP direction-yogunlugu, BOYUT-ESLESMIS null ({len (obs )} part) ===")
print (f"  TP resultant      : {obs .mean ():.3f}")
print (f"  ayni-boyut RASGELE: {nul .mean ():.3f}")
print (f"  GERCEK FAZLA      : {obs .mean ()-nul .mean ():+.3f} | parcalarin %{100 *np .mean (obs >nul ):.0f}'inde TP daha dense")
t1 =obs .mean ()-nul .mean ()
print (("  -> GERCEK: wire-CP'ler rasgele adaylardan DAHA COK ortak axis paylasiyor"if t1 >0.05 
else "  -> ARTEFAKT: yogunluk farki ornek-boyutundan, sinyal YOK"))

# ---------- deploy-edilebilir konsensus feature'lari ----------
X_all ,y_all ,g_all ,F_all =[],[],[],[]
for gi ,(pid ,v )in enumerate (d .items ()):
    D =np .array (v ["dir"],float );y =np .array (v ["y"],int );X =np .array (v ["X"],float )
    if len (y )<4 :continue 
    D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )
    lab ,k =cluster_axes (D ,ANG ,True )
    sizes =np .array ([np .sum (lab ==c )for c in lab ],float )
    dom =np .bincount (lab ).argmax ()# sayim-baskin axis kumesi
    dv =D [lab ==dom ].mean (0 );dv /=(np .linalg .norm (dv )+1e-9 )
    f1 =D @dv 
    f3 =sizes /len (D )
    X_all .append (X );y_all .append (y );g_all .append (np .full (len (y ),gi ))
    F_all .append (np .column_stack ([f1 ,f3 ,sizes ,np .full (len (y ),k /len (D ))]))
X =np .vstack (X_all );Y =np .concatenate (y_all );G =np .concatenate (g_all );Fx =np .vstack (F_all )


def oof_auc (Xa ,tag ):
    o =np .zeros (len (Y ))
    for tr ,te in GroupKFold (5 ).split (Xa ,Y ,G ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xa [tr ],Y [tr ])
        o [te ]=c .predict_proba (Xa [te ])[:,1 ]
    a =roc_auc_score (Y ,o );print (f"  {tag :34s} OOF AUC {a :.4f}");return a ,o 


print (f"\n=== T2: 13 feature'in USTUNE KATIYOR MU ({len (Y )} candidate, {int (Y .sum ())} TP) ===")
print (f"  tek-feature: cos(dominant axis)  AUC {roc_auc_score (Y ,Fx [:,0 ]):.4f}  | cluster-payi AUC {roc_auc_score (Y ,Fx [:,1 ]):.4f}")
a13 ,_ =oof_auc (X ,"13 feature (mevcut gate)")
a17 ,_ =oof_auc (np .hstack ([X ,Fx ]),"13 + 4 konsensus-axis feature")
print (f"\n  KAZANC: {a17 -a13 :+.4f} AUC")
print ("  -> P0-c'ye DEVAM (gercek katki, F1'e cevir)"if a17 -a13 >0.01 
else "  -> KATKI YOK: 13 feature this bilgiyi already tasiyor; axis kolu KAPANIR (durust)")
