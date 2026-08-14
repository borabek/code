# -*- coding: utf-8 -*-
"""ZENGIN GATE'in DURUST DEPLOY count. analyze_rich'teki 'best-thr' IN-SAMPLE threshold secimiydi (iyimser).
Burada threshold NESTED-CV with secilir (train-fold'da sec, test-fold'da uygula) = deployable.
Ayrica: taper(C)+egrilik(D) katkisiz/negatif output -> 13+A+B (46 dim) varyantini da olc (sade = iyi).
Kiyas: canonical base urun ALL 0.750 (fixed 0.35) / metadata-assisted 0.775."""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

NPZ =sys .argv [1 ]if len (sys .argv )>1 else "results/rich_feats.npz"
d =np .load (NPZ ,allow_pickle =True )
X13 ,XR ,Y ,G ,MF ,POS =d ["X13"],d ["XR"],d ["y"],d ["groups"],d ["mfg"],d ["pos"]
ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
AB =np .hstack ([X13 ,XR [:,0 :33 ]])# 13 + konum(9) + very-radius(24)
ALLF =np .hstack ([X13 ,XR ])
FAM ={}
for g in np .unique (G ):
    m =G ==g ;P =POS [m ]
    FAM [g ]=f"g:{np .round (P .max (0 )-P .min (0 ),0 ).tolist ()}|c{int (np .log2 (max (m .sum (),1 )))}"
uf ={f :i for i ,f in enumerate (sorted (set (FAM .values ())))}
FG =np .array ([uf [FAM [g ]]for g in G ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":np .ones (len (Y ),bool ),"WEI":MF ==1 ,"PXC":MF ==0 }


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )


def gt_of (mask ):return sum (ngt_of [g ]for g in np .unique (G [mask ]))


def score_at (sc ,mask ,thr ):
    k =mask &(sc >=thr );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (mask ))


def oof (Xa ,groups ,seed =0 ):
    o =np .zeros (len (Y ))
    for tr ,te in GroupKFold (5 ).split (Xa ,Y ,groups ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed ).fit (Xa [tr ],Y [tr ])
        o [te ]=c .predict_proba (Xa [te ])[:,1 ]
    return o 


def nested_thr_f1 (sc ,mask ,groups ,K =5 ):
    """threshold train-fold'da secilir, test-fold'da uygulanir -> in-sample iyimserlik YOK."""
    idx =np .where (mask )[0 ];gp =np .unique (groups [idx ])
    rs =np .random .RandomState (0 );gp =gp [rs .permutation (len (gp ))]
    TP =NK =GT =0 ;chosen =[]
    for f in np .array_split (gp ,K ):
        te_g =set (f .tolist ())
        trm =mask &np .array ([g not in te_g for g in groups ])
        tem =mask &np .array ([g in te_g for g in groups ])
        bt ,bf =0.35 ,-1 
        for t in THRS :
            _ ,_ ,ff =score_at (sc ,trm ,t )
            if ff >bf :bf ,bt =ff ,t 
        chosen .append (bt )
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    p ,r ,f1 =prf (TP ,NK ,GT )
    return p ,r ,f1 ,float (np .median (chosen ))


def topn (sc ,mask ):
    TP =NK =GT =0 
    for g in np .unique (G [mask ]):
        m =np .where (mask &(G ==g ))[0 ];n =ngt_of [g ]
        i2 =m [np .argsort (-sc [m ])[:n ]]
        TP +=int (Y [i2 ].sum ());NK +=len (i2 );GT +=n 
    return prf (TP ,NK ,GT )[2 ]


gt_tot =int (sum (ngt_of .values ()))
print (f"{NPZ }: {len (Y )} candidate ({len (Y )/max (gt_tot ,1 ):.2f}x GT) | {len (ngt_of )} part | {len (uf )} aile"
f" | ADAY-TAVAN recall {int (Y .sum ())}/{gt_tot } = {Y .sum ()/max (gt_tot ,1 ):.3f}"
f" (mukemmel-gate F1 tavani {2 *Y .sum ()/max (gt_tot ,1 )/(1 +Y .sum ()/max (gt_tot ,1 )):.3f})\n")
for split_nm ,groups in (("PARCA-out (canonical ile kiyaslanabilir)",G ),("AILE-out (KATI)",FG )):
    print (f"================ {split_nm } ================")
    for nm ,Xa in (("13 BAZ",X13 ),("13+A+B (46d)",AB ),("13+HEPSI (54d)",ALLF )):
        sc =oof (Xa ,groups )
        line =[f"{nm :15s} AUC {roc_auc_score (Y ,sc ):.4f}"]
        for mk in ("ALL","WEI","PXC"):
            p ,r ,f1 ,mt =nested_thr_f1 (sc ,masks [mk ],groups )
            line .append (f"| {mk } nested-thr F1 {f1 :.4f}(thr{mt :.2f})")
        line .append (f"| ALL topN {topn (sc ,masks ['ALL']):.4f}")
        print ("  "+" ".join (line ))
    print ()
print ("KIYAS: canonical base ALL 0.750 (fixed 0.35) | metadata-assisted ALL 0.775 (topN)")
print ("DECISION: 13+A+B nested-thr ALL, 0.750'yi materyal geciyorsa -> DEPLOY ADAYI (receipt + cp_config).")
