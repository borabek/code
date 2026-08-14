# -*- coding: utf-8 -*-
"""FAZ 2 / P1b -- embedding'i EN IYI MUTLAK sonuca cevirmeye calis (durust okuma sonrasi).
P1 bulgusu: GO kriteri MLP'de gecti AMA most iyi mutlak number 0.7560 -> 0.7596 (+0.004) kaldi;
because RF already zengin feature'lardan bilgiyi cikariyor, 188 embedding kolonu RF'yi seyreltiyor.
Iki mesru rafinasyon:
  (a) EMBEDDING SECIMI: train-fold'da onem/varyansla top-K embedding kolonu (sizintisiz secim)
  (b) ENSEMBLE: rf(rich) + mlp(rich+emb) -- different error yapan modelleri birlestir
Karar split'i: family-out. Hedef: most iyi MUTLAK family-out ALL."""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .neural_network import MLPClassifier 
from sklearn .preprocessing import StandardScaler 
from sklearn .pipeline import make_pipeline 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
rp =[str (x )for x in r ["part_ids"]];ep =[str (x )for x in e ["part_ids"]]
emap ={p :i for i ,p in enumerate (ep )}
EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]));ok =np .zeros (len (Y ),bool )
for gi ,pid in enumerate (rp ):
    if pid not in emap :continue 
    ir =np .where (G ==gi )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
    if len (ir )!=len (ie ):continue 
    EM [ir ]=e ["EMB"][ie ];ok [ir ]=True 
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt_of =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])&ok 
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
FG =np .array ([{v :i for i ,v in enumerate (sorted (set (fam )))}[v ]for v in fam ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":WORK ,"WEI":WORK &(MF ==1 ),"PXC":WORK &(MF ==0 )}
print (f"WORK {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part | rich {RICH .shape [1 ]}d emb {EM .shape [1 ]}d")


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );rr =tp /max (gt ,1 );return 2 *p *rr /max (p +rr ,1e-9 )
def gt_of (m ):return int (sum (ngt_of [int (g )]for g in np .unique (G [m ])))
def sc_at (sc ,m ,t ):
    k =m &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (m ))


def nested (sc ,m ,groups ,K =5 ):
    gp =np .unique (groups [np .where (m )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,K ):
        tg =set (f .tolist ())
        trm =m &np .array ([g not in tg for g in groups ]);tem =m &np .array ([g in tg for g in groups ])
        if not trm .any ()or not tem .any ():continue 
        bt ,bf =0.35 ,-1 
        for t in THRS :
            ff =sc_at (sc ,trm ,t )
            if ff >bf :bf ,bt =ff ,t 
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    return prf (TP ,NK ,GT )


def topn (sc ,m ):
    TP =NK =GT =0 
    for g in np .unique (G [m ]):
        i =np .where (m &(G ==g ))[0 ];n =ngt_of [int (g )]
        j =i [np .argsort (-sc [i ])[:n ]];TP +=int (Y [j ].sum ());NK +=len (j );GT +=n 
    return prf (TP ,NK ,GT )


def rf (seed =0 ):return RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed )
def mlp (seed =0 ):return make_pipeline (StandardScaler (),MLPClassifier ((128 ,48 ),max_iter =400 ,
random_state =seed ,early_stopping =True ,n_iter_no_change =15 ))


def oof (build ,Xa ,groups ,topk =0 ):
    """topk>0: embedding kolonlarini SADECE train-fold'da selects (sizintisiz)."""
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (Xa [idx ],Y [idx ],groups [idx ]):
        itr ,ite =idx [tr ],idx [te ]
        if topk :
            sel_rf =RandomForestClassifier (200 ,min_samples_leaf =5 ,n_jobs =-1 ,random_state =0 )
            sel_rf .fit (EM [itr ],Y [itr ])
            cols =np .argsort (-sel_rf .feature_importances_ )[:topk ]
            Xtr =np .hstack ([RICH [itr ],EM [itr ][:,cols ]]);Xte =np .hstack ([RICH [ite ],EM [ite ][:,cols ]])
        else :
            Xtr ,Xte =Xa [itr ],Xa [ite ]
        o [ite ]=build ().fit (Xtr ,Y [itr ]).predict_proba (Xte )[:,1 ]
    return o 


FULL =np .hstack ([RICH ,EM ])
print (f"\n================ family-out (KARAR) ================")
print (f"{'model':30s} {'AUC':>7s} {'ALL':>7s} {'WEI':>7s} {'PXC':>7s} {'topN':>7s}")
scores ={}
cfgs =[("rf(rich)",lambda :oof (rf ,RICH ,FG )),
("rf(rich+emb_tum)",lambda :oof (rf ,FULL ,FG )),
("rf(rich+emb_top20)",lambda :oof (rf ,FULL ,FG ,topk =20 )),
("rf(rich+emb_top40)",lambda :oof (rf ,FULL ,FG ,topk =40 )),
("mlp(rich+emb)",lambda :oof (mlp ,FULL ,FG ))]
for nm ,fn in cfgs :
    sc =fn ();scores [nm ]=sc 
    print (f"{nm :30s} {roc_auc_score (Y [WORK ],sc [WORK ]):7.4f} {nested (sc ,masks ['ALL'],FG ):7.4f} "
    f"{nested (sc ,masks ['WEI'],FG ):7.4f} {nested (sc ,masks ['PXC'],FG ):7.4f} {topn (sc ,WORK ):7.4f}",flush =True )

print ("\n--- ENSEMBLE (rank-mean, different error yapan models) ---")
def rank01 (s ):
    o =np .zeros (len (s ));i =np .where (WORK )[0 ]
    o [i ]=(np .argsort (np .argsort (s [i ]))/max (len (i )-1 ,1 ));return o 
combos =[("rf(rich) + mlp(+emb)",["rf(rich)","mlp(rich+emb)"]),
("rf(rich) + rf(+emb_top20)",["rf(rich)","rf(rich+emb_top20)"]),
("rf(rich) + rf(+emb_top20) + mlp(+emb)",["rf(rich)","rf(rich+emb_top20)","mlp(rich+emb)"])]
best =("rf(rich)",nested (scores ["rf(rich)"],masks ["ALL"],FG ))
for nm ,ks in combos :
    sc =np .mean ([rank01 (scores [k ])for k in ks ],0 )
    a =nested (sc ,masks ["ALL"],FG )
    print (f"{nm :38s} ALL {a :7.4f} | WEI {nested (sc ,masks ['WEI'],FG ):7.4f} | "
    f"PXC {nested (sc ,masks ['PXC'],FG ):7.4f} | topN {topn (sc ,WORK ):7.4f}",flush =True )
    if a >best [1 ]:best =(nm ,a )
for nm in scores :
    a =nested (scores [nm ],masks ["ALL"],FG )
    if a >best [1 ]:best =(nm ,a )
print (f"\n>>> EN IYI MUTLAK (family-out ALL): {best [0 ]} = {best [1 ]:.4f}  (rich-rf tabani "
f"{nested (scores ['rf(rich)'],masks ['ALL'],FG ):.4f}, fark {best [1 ]-nested (scores ['rf(rich)'],masks ['ALL'],FG ):+.4f})")
