# -*- coding: utf-8 -*-
"""K5.2 LOMO + K5.1 GROUP-DRO ten sondasi.

SORU: gate'in esigi/agirligi HAVUZLANMIS kumede secilirse wrong objektif for
secilmis becomes ([[measurement-yolu-and-secim-kusurlari]] ucuncu ders). LOMO (leave-one-
manufacturer-out) secim FARKLI a setting mi seciyor, and EN KOTU manufacturer duzeliyor mu?

KURULUM (corpus: g10 with yeniden turetilmis v4):
  each manufacturer sirayla TEST, rest EGITIM  -> manufacturer basina AUC + F1
  threshold izgarasi taranir; UC secim olcutu KIYASLANIR:
    (a) HAVUZLANMIS mean  (b) LOMO ORTALAMASI  (c) EN KOTU manufacturer (DRO)
KILL: (b) and (c) with (a) AYNI ayari seciyorsa arm NULL.
"""
import collections ,json ,os ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import wire_gate 
from sklearn .ensemble import RandomForestClassifier 

d =np .load ("results/zengin_parite_v4_g10.npz",allow_pickle =True )
X0 =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
y =np .asarray (d ["y"]).astype (int )
pid =np .array ([str (p )for p in d ["pids"]])
mfg =np .array ([str (m )for m in d ["mfg"]])
M =np .zeros ((len (X0 ),X0 .shape [1 ]*2 ))
for u in np .unique (pid ):
    i =np .where (pid ==u )[0 ]
    M [i ]=wire_gate .within_part (X0 [i ],"zskor")
say =collections .Counter (mfg )
markalar =[m for m ,c in say .items ()if c >=300 ]
print (f"candidate {len (M )} | LOMO markalari (n>=300): {markalar }\n",flush =True )

ESIKLER =[0.20 ,0.30 ,0.40 ,0.50 ,0.60 ]
tab ={e :[]for e in ESIKLER }
for m in markalar :
    te =mfg ==m ;tr =~te 
    clf =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [tr ],y [tr ])
    s =clf .predict_proba (M [te ])[:,1 ]
    yt =y [te ]
    for e in ESIKLER :
        p =(s >=e )
        tp =int ((p &(yt ==1 )).sum ());fp =int ((p &(yt ==0 )).sum ())
        fn =int ((~p &(yt ==1 )).sum ())
        f1 =2 *tp /max (2 *tp +fp +fn ,1 )
        tab [e ].append (f1 )
    print (f"  {m :<6} n={te .sum ():<6} "+
    " ".join (f"e{e :.2f}={tab [e ][-1 ]:.3f}"for e in ESIKLER ),flush =True )

print ("\nSECIM OLCUTU KIYASI:")
hav ={e :float (np .mean (v ))for e ,v in tab .items ()}# LOMO ortalamasi
enk ={e :float (np .min (v ))for e ,v in tab .items ()}# EN KOTU manufacturer (DRO)
e_ort =max (hav ,key =hav .get );e_dro =max (enk ,key =enk .get )
for e in ESIKLER :
    print (f"  threshold {e :.2f}: LOMO ort {hav [e ]:.4f} | EN KOTU {enk [e ]:.4f}")
print (f"\n  LOMO ORTALAMASI secer -> threshold {e_ort :.2f} (F1 {hav [e_ort ]:.4f})")
print (f"  EN KOTU (DRO)   secer -> threshold {e_dro :.2f} (en kotu {enk [e_dro ]:.4f})")
print (f"  AYNI MI: {'EVET -> arm NULL'if e_ort ==e_dro else 'HAYIR -> arm ACIK'}")
json .dump ({"markalar":markalar ,"esikler":ESIKLER ,
"lomo_ort":hav ,"en_kotu":enk ,
"secim_ort":e_ort ,"secim_dro":e_dro ,
"same":bool (e_ort ==e_dro )},
open ("results/k52_lomo.json","w"),indent =1 )
print ("receipt -> results/k52_lomo.json")
