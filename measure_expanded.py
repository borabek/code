# -*- coding: utf-8 -*-
"""R1-a: OGRENME EGRISINE ILK GERCEK NOKTA (ekstrapolasyon not, measurement).
Ek data: results/rich_extra.npz = 220 part / 1058 CP (more before kullanilmayan WEI parcalari).
Bunlar segmentasyon egitiminde GORULMUS parts -> SKORLAMADA kullanilmaz, only GATE EGITIMINE girer.
Aile-leakage korumasi: test fold'undaki a aileye ait ek-part egitime ALINMAZ.
Olcum: WORK (541 temiz+kilitsiz part) on family-out, nested-CV threshold -- oncekiyle birebir same protocol.
Kiyas: mevcut 0.7516 (only WORK) / 0.7552 (WORK+153 seen)."""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
x =np .load ("results/rich_extra.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]

# --- ana pool (skorlama + training) ---
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
rp =[str (v )for v in r ["part_ids"]]
ep =[str (v )for v in e ["part_ids"]];emap ={p :i for i ,p in enumerate (ep )}
EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]))
for gi ,pid in enumerate (rp ):
    if pid not in emap :continue 
    ir =np .where (G ==gi )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
    if len (ir )==len (ie ):EM [ir ]=e ["EMB"][ie ]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])
XA =np .hstack ([RICH ,EM ])# ana pool: zengin + embedding
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])

# --- EK pool (only training) : embedding YOK -> same boyuta sifirla ---
xp =[str (v )for v in x ["part_ids"]]
XX =np .hstack ([x ["X13"],x ["XR"][:,0 :33 ]])
XX =np .hstack ([XX ,np .zeros ((len (XX ),EM .shape [1 ]))])
XY =x ["y"];XG =x ["groups"]
xgpid =np .array ([xp [int (g )]for g in XG ])
xfam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in xgpid ])

allf =sorted (set (fam )|set (xfam ));fidx ={v :i for i ,v in enumerate (allf )}
FG =np .array ([fidx [v ]for v in fam ]);XFG =np .array ([fidx [v ]for v in xfam ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":WORK ,"WEI":WORK &(MF ==1 ),"PXC":WORK &(MF ==0 )}
print (f"skorlama (WORK): {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part")
print (f"ek training havuzu: {len (XY )} candidate / {len (set (xgpid ))} part (embedding YOK -> sifir dolgulu)")
print (f"seen (ana havuzdan, egitime): {int (((gseen ==1 )).sum ())} candidate\n")


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );rr =tp /max (gt ,1 );return 2 *p *rr /max (p +rr ,1e-9 )
def gt_of (m ):return int (sum (ngt [int (g )]for g in np .unique (G [m ])))
def sc_at (sc ,m ,t ):
    k =m &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (m ))


def nested (sc ,m ):
    gp =np .unique (FG [np .where (m )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,5 ):
        tg =set (f .tolist ())
        trm =m &np .array ([g not in tg for g in FG ]);tem =m &np .array ([g in tg for g in FG ])
        if not trm .any ()or not tem .any ():continue 
        bt ,bf =0.35 ,-1 
        for t in THRS :
            ff =sc_at (sc ,trm ,t )
            if ff >bf :bf ,bt =ff ,t 
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    return prf (TP ,NK ,GT )


def oof (use_seen ,use_extra ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (XA [idx ],Y [idx ],FG [idx ]):
        te_f =set (FG [idx ][te ].tolist ())
        Xtr ,Ytr =[XA [idx [tr ]]],[Y [idx [tr ]]]
        if use_seen :# ana havuzun seen parcalari (aile-korumali)
            m =(gseen ==1 )&~np .isin (FG ,list (te_f ))
            Xtr .append (XA [m ]);Ytr .append (Y [m ])
        if use_extra :# EK pool (aile-korumali)
            m =~np .isin (XFG ,list (te_f ))
            Xtr .append (XX [m ]);Ytr .append (XY [m ])
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        c .fit (np .vstack (Xtr ),np .concatenate (Ytr ))
        o [idx [te ]]=c .predict_proba (XA [idx [te ]])[:,1 ]
    return o 


print (f"{'training havuzu':34s} {'GT':>6s} {'AUC':>8s} {'ALL':>8s} {'WEI':>8s} {'PXC':>8s}")
res ={}
for nm ,us ,ux in (("WORK (541 part)",False ,False ),
("+ seen (694 part)",True ,False ),
("+ seen + EK 220 (914 part)",True ,True )):
    sc =oof (us ,ux )
    gt =gt_of (WORK )+(int (((gseen ==1 )).sum ()and sum (ngt [int (g )]for g in np .unique (G [gseen ==1 ])))if us else 0 )+(int (x ["ngt"].sum ())if ux else 0 )
    row =(roc_auc_score (Y [WORK ],sc [WORK ]),nested (sc ,masks ["ALL"]),
    nested (sc ,masks ["WEI"]),nested (sc ,masks ["PXC"]))
    res [nm ]=row 
    print (f"{nm :34s} {gt :6d} {row [0 ]:8.4f} {row [1 ]:8.4f} {row [2 ]:8.4f} {row [3 ]:8.4f}",flush =True )

b =res ["WORK (541 part)"][1 ]
for nm ,row in res .items ():
    if nm .startswith ("WORK"):continue 
    print (f"  {nm :32s} ALL kazanci {row [1 ]-b :+.4f}")
json .dump ({k :{"AUC":v [0 ],"ALL":v [1 ],"WEI":v [2 ],"PXC":v [3 ]}for k ,v in res .items ()},
open ("results/expanded_curve_point.json","w"),indent =1 )
print ("\n-> results/expanded_curve_point.json  (EGRIYE ILK GERCEK NOKTA)")
