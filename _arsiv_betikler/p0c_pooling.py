# -*- coding: utf-8 -*-
"""P0-c: SKOR-HAVUZLAMA testi (feature-ekleme DEGIL). FBI argumaninin last kolu: a sirada/kumede
8 opening same tipse, gate skorlarini HAVUZLAMAK gurultuyu sqrt(N) azaltir -> ayrim guclenir.
Eksen-feature'lari eklemek katmadi (+0.0036); but HAVUZLAMA different a mekanizma.
Leave-one-out havuzlama (own skorunu disla = leakage absent), 3 cluster tanimi:
  (a) axis kumesi  (b) benzer-radius+axis  (c) uzamsal komsu (nn)
DECISION: havuzlanmis skor AUC'yi/top-N F1'i artiriyor mu."""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 
from p0a_axis_discord import cluster_axes 

POOL =sys .argv [1 ]if len (sys .argv )>1 else "results/wei_aggr_pool.json"
d =json .load (open (POOL ))
Xs ,Ys ,Gs ,Ps ,Ds ,Ns =[],[],[],[],[],{}
for gi ,(pid ,v )in enumerate (d .items ()):
    y =np .array (v ["y"],int )
    if len (y )<4 :continue 
    Xs .append (np .array (v ["X"],float ));Ys .append (y );Gs .append (np .full (len (y ),gi ))
    Ps .append (np .array (v ["P"],float ));Ds .append (np .array (v ["dir"],float ));Ns [gi ]=v ["N"]
X =np .vstack (Xs );Y =np .concatenate (Ys );G =np .concatenate (Gs );P =np .vstack (Ps );D =np .vstack (Ds )
D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )

# OOF gate skoru (13 feature, sizintisiz)
oof =np .zeros (len (Y ))
for tr ,te in GroupKFold (5 ).split (X ,Y ,G ):
    c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (X [tr ],Y [tr ])
    oof [te ]=c .predict_proba (X [te ])[:,1 ]
base_auc =roc_auc_score (Y ,oof )
print (f"{len (Y )} candidate / {int (Y .sum ())} TP / {len (Ns )} part | BAZ gate OOF AUC {base_auc :.4f}")


def loo_pool (members ,scores ):
    """leave-one-out cluster ortalamasi (own skorunu disla)."""
    out =np .zeros (len (scores ))
    for c in np .unique (members ):
        m =np .where (members ==c )[0 ]
        if len (m )<2 :out [m ]=scores [m ];continue 
        s =scores [m ].sum ()
        out [m ]=(s -scores [m ])/(len (m )-1 )
    return out 


feats ={}
for gi in np .unique (G ):
    m =np .where (G ==gi )[0 ]
    lab ,_ =cluster_axes (D [m ],20.0 ,True )
    feats .setdefault ("axis",np .zeros (len (Y )))[m ]=loo_pool (lab ,oof [m ])
    # uzamsal komsu: most yakin 3 adayin ort skoru
    pp =P [m ];dd =np .linalg .norm (pp [:,None ]-pp [None ],axis =-1 );np .fill_diagonal (dd ,np .inf )
    k =min (3 ,len (m )-1 )if len (m )>1 else 0 
    if k >0 :
        idx =np .argsort (dd ,axis =1 )[:,:k ]
        feats .setdefault ("nn3",np .zeros (len (Y )))[m ]=oof [m ][idx ].mean (1 )
    else :
        feats .setdefault ("nn3",np .zeros (len (Y )))[m ]=oof [m ]
        # part-geneli mean (most kaba pool)
    feats .setdefault ("part",np .zeros (len (Y )))[m ]=loo_pool (np .zeros (len (m ),int ),oof [m ])

print ("\n=== HAVUZLANMIS SKOR tek basina ayirici mi (AUC) ===")
for k ,v in feats .items ():
    print (f"  {k :6s} havuzlanmis-skor AUC {roc_auc_score (Y ,v ):.4f}")

print ("\n=== 13 feature + havuzlanmis skorlar (OOF) ===")
Xp =np .column_stack ([X ]+[feats [k ]for k in feats ])
o2 =np .zeros (len (Y ))
for tr ,te in GroupKFold (5 ).split (Xp ,Y ,G ):
    c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xp [tr ],Y [tr ])
    o2 [te ]=c .predict_proba (Xp [te ])[:,1 ]
a2 =roc_auc_score (Y ,o2 )
print (f"  13+pool AUC {a2 :.4f}   (baz {base_auc :.4f})  KAZANC {a2 -base_auc :+.4f}")


def topn_f1 (sc ):
    TP =NK =GT =0 
    for gi in np .unique (G ):
        m =np .where (G ==gi )[0 ];n =Ns [gi ]
        idx =m [np .argsort (-sc [m ])[:n ]]
        TP +=int (Y [idx ].sum ());NK +=len (idx );GT +=n 
    p =TP /max (NK ,1 );r =TP /max (GT ,1 );return 2 *p *r /max (p +r ,1e-9 )


print (f"\n=== top-N (metadata-assisted) F1 ===")
print (f"  baz gate      : {topn_f1 (oof ):.4f}")
print (f"  13+pool      : {topn_f1 (o2 ):.4f}   KAZANC {topn_f1 (o2 )-topn_f1 (oof ):+.4f}")
print ("\n  -> KAZANC >+0.02 ise havuzlama GERCEK lever; degilse P0 kolu tamamen kapanir (durust).")
