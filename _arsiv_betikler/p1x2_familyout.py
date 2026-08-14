# -*- coding: utf-8 -*-
"""P1-x2: KONUM feature kazanci (+0.041 top-N F1) GERCEK mi otherwise AILE-EZBERI mi?
Tuzak: GroupKFold PARCA-bazliydi; kardes parts train'de olunca model 'this ailede tel z=0.3'te'
ezberleyebilir. Receipt already part-out(0.807) vs family-out(0.799) farkini biliyor; KONUM'da this difference
DAHA BUYUK may be (konum aileye ozgudur).
Aile anahtari (etiketsiz, geometry_key ruhunda): candidate-bulutu bbox boyutlari (yuvarlanmis) + log2(candidate count).
DECISION: family-out'ta da kazanc kaliyorsa GERCEK; cokuyorsa aile-ezberi (durustce kapat)."""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

POOL =sys .argv [1 ]if len (sys .argv )>1 else "results/wei_aggr_pool.json"
d =json .load (open (POOL ))
Xs ,Ys ,Gs ,Ps ,Ns ,FAM =[],[],[],[],{},{}
for gi ,(pid ,v )in enumerate (d .items ()):
    y =np .array (v ["y"],int )
    if len (y )<4 :continue 
    P =np .array (v ["P"],float )
    Xs .append (np .array (v ["X"],float ));Ys .append (y );Gs .append (np .full (len (y ),gi ))
    Ps .append (P );Ns [gi ]=v ["N"]
    dims =np .round (P .max (0 )-P .min (0 ),0 )# candidate-bulutu bbox (mm, 1mm yuvarlama)
    FAM [gi ]=f"g:{dims .tolist ()}|c{int (np .log2 (max (len (y ),1 )))}"
X =np .vstack (Xs );Y =np .concatenate (Ys );G =np .concatenate (Gs );P =np .vstack (Ps )
fam_ids =np .array ([FAM [g ]for g in G ])
uf ={f :i for i ,f in enumerate (sorted (set (fam_ids )))}
FG =np .array ([uf [f ]for f in fam_ids ])

A =np .zeros ((len (Y ),3 ))
for gi in np .unique (G ):
    m =np .where (G ==gi )[0 ];pp =P [m ];lo ,hi =pp .min (0 ),pp .max (0 )
    A [m ]=(pp -lo )/np .maximum (hi -lo ,1e-6 )


def run (Xa ,groups ):
    o =np .zeros (len (Y ))
    for tr ,te in GroupKFold (5 ).split (Xa ,Y ,groups ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xa [tr ],Y [tr ])
        o [te ]=c .predict_proba (Xa [te ])[:,1 ]
    return o 


def topn (sc ):
    TP =NK =GT =0 
    for gi in np .unique (G ):
        m =np .where (G ==gi )[0 ];n =Ns [gi ]
        idx =m [np .argsort (-sc [m ])[:n ]]
        TP +=int (Y [idx ].sum ());NK +=len (idx );GT +=n 
    p =TP /max (NK ,1 );r =TP /max (GT ,1 );return 2 *p *r /max (p +r ,1e-9 )


print (f"{len (Y )} candidate / {int (Y .sum ())} TP / {len (Ns )} part / {len (uf )} AILE (ort {len (Ns )/max (len (uf ),1 ):.2f} part/aile)")
for split_name ,groups in (("PARCA-out (onceki)",G ),("AILE-out (KATI)",FG )):
    o0 =run (X ,groups );oA =run (np .hstack ([X ,A ]),groups )
    a0 ,f0 =roc_auc_score (Y ,o0 ),topn (o0 );a1 ,f1 =roc_auc_score (Y ,oA ),topn (oA )
    print (f"\n=== {split_name } ===")
    print (f"  13 feature      AUC {a0 :.4f}  top-N F1 {f0 :.4f}")
    print (f"  13 + konum      AUC {a1 :.4f}  top-N F1 {f1 :.4f}   KAZANC {f1 -f0 :+.4f}")
print ("\nKARAR: AILE-out kazanci >= PARCA-out'un yarisi ise GERCEK sinyal (genellenir).")
print ("       AILE-out'ta cokuyorsa -> aile-ezberi, deploy'da whereas yaramaz (durustce kapat).")
