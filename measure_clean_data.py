# -*- coding: utf-8 -*-
"""TEMIZ VERI-LEVER TESTI: new indirilen 220 part (2327 CP) gate egitimine eklenince ne oluyor?
ONCEKI TESTIN HANDIKAPLARI KALDIRILDI:
  * this parts 'seen' DEGIL -- gercekten new, segmentasyon gormedi
  * embedding absent diye SIFIR DOLGU yapmiyoruz; each two taraf da SADECE zengin feature (46d) kullanir
Olcum: WORK (541 temiz+kilitsiz part), family-out, nested-CV threshold -- onceki protokolle same."""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

r =np .load ("results/rich_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"));LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])# 46d -- embedding YOK
rp =[str (v )for v in r ["part_ids"]]
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])

def load_extra (path ):
    z =np .load (path ,allow_pickle =True )
    zp =[str (v )for v in z ["part_ids"]]
    XX =np .hstack ([z ["X13"],z ["XR"][:,0 :33 ]])
    zg =np .array ([zp [int (g )]for g in z ["groups"]])
    zf =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in zg ])
    return XX ,z ["y"],zf ,len (zp ),int (z ["ngt"].sum ())

EX ={}
for tag ,path in (("seen220","results/rich_extra.npz"),("YENI220","results/rich_new.npz")):
    try :EX [tag ]=load_extra (path )
    except Exception as e :print (f"{tag } yuklenemedi: {e }")

allf =sorted (set (fam )|{f for v in EX .values ()for f in v [2 ]})
fidx ={v :i for i ,v in enumerate (allf )}
FG =np .array ([fidx [v ]for v in fam ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":WORK ,"WEI":WORK &(MF ==1 ),"PXC":WORK &(MF ==0 )}

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

def oof (extras ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (RICH [idx ],Y [idx ],FG [idx ]):
        te_f =set (FG [idx ][te ].tolist ())
        Xtr ,Ytr =[RICH [idx [tr ]]],[Y [idx [tr ]]]
        for tag in extras :
            XX ,YY ,ZF ,_ ,_ =EX [tag ]
            keep =np .array ([fidx .get (f ,-1 )not in te_f for f in ZF ])
            Xtr .append (XX [keep ]);Ytr .append (YY [keep ])
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        c .fit (np .vstack (Xtr ),np .concatenate (Ytr ))
        o [idx [te ]]=c .predict_proba (RICH [idx [te ]])[:,1 ]
    return o 

base_gt =gt_of (WORK )
print (f"skorlama: {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part / GT {base_gt }")
for t ,v in EX .items ():print (f"  ek pool {t }: {v [3 ]} part / GT {v [4 ]}")
print (f"\n{'training havuzu':30s} {'GT':>6s} {'AUC':>8s} {'ALL':>8s} {'WEI':>8s} {'PXC':>8s}")
res ={}
combos =[("TABAN (541 part)",[]),("+ YENI 220",["YENI220"]),
("+ YENI + seen220",["YENI220","seen220"])]
for nm ,ex in combos :
    if any (t not in EX for t in ex ):continue 
    sc =oof (ex )
    gt =base_gt +sum (EX [t ][4 ]for t in ex )
    row =(roc_auc_score (Y [WORK ],sc [WORK ]),nested (sc ,masks ["ALL"]),
    nested (sc ,masks ["WEI"]),nested (sc ,masks ["PXC"]))
    res [nm ]=(gt ,)+row 
    print (f"{nm :30s} {gt :6d} {row [0 ]:8.4f} {row [1 ]:8.4f} {row [2 ]:8.4f} {row [3 ]:8.4f}",flush =True )
b =res ["TABAN (541 part)"]
print ()
for nm ,v in res .items ():
    if nm .startswith ("TABAN"):continue 
    dbl =np .log2 (v [0 ]/b [0 ])
    print (f"  {nm :26s} GT {b [0 ]}->{v [0 ]} ({dbl :.2f} katlama) | ALL {v [2 ]-b [2 ]:+.4f} | katlama basina {(v [2 ]-b [2 ])/max (dbl ,1e-9 ):+.4f}")
json .dump ({k :{"GT":v [0 ],"AUC":v [1 ],"ALL":v [2 ],"WEI":v [3 ],"PXC":v [4 ]}for k ,v in res .items ()},
open ("results/clean_data_test.json","w"),indent =1 )
print ("\n-> results/clean_data_test.json")
