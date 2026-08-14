# -*- coding: utf-8 -*-
"""FAZ 2 / P2b -- URETICI-BASINA MODEL YONLENDIRME (P2'nin olculmus ayrismasindan dogdu).
Bulgu: SetNet WEI'de 0.6813 -> ~0.729 (+0.048) but PXC'de -0.011 -> ALL'da kayboluyor.
Test: WEI adaylari SetNet, PXC adaylari RF/ensemble with puanlansin; esikler manufacturer-basina
nested-CV with secilsin; TP/FP/FN HAVUZLANIP TEK AGREGAT F1 raporlansin (kullanicinin istedigi form).
Skorlar diske kaydedilir (P7'de yeniden training gerekmesin)."""
import json 
import numpy as np ,torch ,torch .nn as nn 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .neural_network import MLPClassifier 
from sklearn .preprocessing import StandardScaler 
from sklearn .pipeline import make_pipeline 
from sklearn .model_selection import GroupKFold 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
rp =[str (x )for x in r ["part_ids"]];ep =[str (x )for x in e ["part_ids"]]
emap ={p :i for i ,p in enumerate (ep )}
EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]))
for gi ,pid in enumerate (rp ):
    if pid not in emap :continue 
    ir =np .where (G ==gi )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
    if len (ir )==len (ie ):EM [ir ]=e ["EMB"][ie ]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]]);FULL =np .hstack ([RICH ,EM ])
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt_of =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
FG =np .array ([{v :i for i ,v in enumerate (sorted (set (fam )))}[v ]for v in fam ])
THRS =np .round (np .arange (0.05 ,0.96 ,0.02 ),3 )
mu ,sd =FULL [WORK ].mean (0 ),FULL [WORK ].std (0 )+1e-6 
XN =(FULL -mu )/sd 
print (f"WORK {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part")


class SetNet (nn .Module ):
    def __init__ (self ,d ,h =128 ,nl =2 ):
        super ().__init__ ()
        self .inp =nn .Sequential (nn .Linear (d ,h ),nn .ReLU (),nn .Linear (h ,h ))
        self .enc =nn .TransformerEncoder (nn .TransformerEncoderLayer (h ,4 ,h *2 ,dropout =0.1 ,
        batch_first =True ,norm_first =True ),nl )
        self .out =nn .Sequential (nn .ReLU (),nn .Linear (h ,1 ))
    def forward (self ,x ):return self .out (self .enc (self .inp (x ))).squeeze (-1 )


def setnet_oof (seeds =(0 ,1 ,2 ),epochs =60 ):
    acc =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (XN [idx ],Y [idx ],FG [idx ]):
        itr ,ite =idx [tr ],idx [te ]
        groups =[np .where ((G ==g )&WORK )[0 ]for g in np .unique (G [itr ])]
        groups =[g for g in groups if len (g )>=2 and Y [g ].sum ()>0 ]
        for seed in seeds :
            torch .manual_seed (seed );np .random .seed (seed )
            net =SetNet (XN .shape [1 ]).to (dev )
            opt =torch .optim .AdamW (net .parameters (),lr =1e-3 ,weight_decay =1e-4 )
            bce =nn .BCEWithLogitsLoss ()
            for _ in range (epochs ):
                np .random .shuffle (groups );net .train ()
                for gi in groups :
                    x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
                    yy =torch .tensor (Y [gi ],dtype =torch .float32 ,device =dev )
                    loss =bce (net (x ).squeeze (0 ),yy )
                    opt .zero_grad ();loss .backward ();opt .step ()
            net .eval ()
            with torch .no_grad ():
                for g in np .unique (G [ite ]):
                    gi =np .where ((G ==g )&WORK )[0 ]
                    if not len (gi ):continue 
                    x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
                    acc [gi ]+=torch .sigmoid (net (x ).squeeze (0 )).cpu ().numpy ()/len (seeds )
    return acc 


def oof (build ,Xa ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (Xa [idx ],Y [idx ],FG [idx ]):
        o [idx [te ]]=build ().fit (Xa [idx ][tr ],Y [idx ][tr ]).predict_proba (Xa [idx ][te ])[:,1 ]
    return o 


print ("egitiliyor: RF(rich), MLP(+emb), SetNet(3 seed)...",flush =True )
s_rf =oof (lambda :RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ),RICH )
s_mlp =oof (lambda :make_pipeline (StandardScaler (),MLPClassifier ((128 ,48 ),max_iter =400 ,random_state =0 ,
early_stopping =True ,n_iter_no_change =15 )),FULL )
s_set =setnet_oof ()
np .savez ("results/oof_scores.npz",rf =s_rf ,mlp =s_mlp ,setnet =s_set ,work =WORK ,y =Y ,groups =G ,
mfg =MF ,fam =FG ,part_ids =np .array (rp ))
print ("-> results/oof_scores.npz",flush =True )


def rank01 (s ,m ):
    o =np .zeros (len (s ));i =np .where (m )[0 ]
    o [i ]=np .argsort (np .argsort (s [i ]))/max (len (i )-1 ,1 );return o 


def gt_of (m ):return int (sum (ngt_of [int (g )]for g in np .unique (G [m ])))
def counts_at (sc ,m ,t ):
    k =m &(sc >=t );return int (Y [k ].sum ()),int (k .sum ())


def nested_counts (sc ,m ,K =5 ):
    """manufacturer-ici nested threshold; HAM SAYILAR returns (havuzlama for)."""
    gp =np .unique (FG [np .where (m )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,K ):
        tg =set (f .tolist ())
        trm =m &np .array ([g not in tg for g in FG ]);tem =m &np .array ([g in tg for g in FG ])
        if not trm .any ()or not tem .any ():continue 
        bt ,bf =0.35 ,-1 
        for t in THRS :
            tp ,nk =counts_at (sc ,trm ,t );gt =gt_of (trm )
            p =tp /max (nk ,1 );rr =tp /max (gt ,1 );ff =2 *p *rr /max (p +rr ,1e-9 )
            if ff >bf :bf ,bt =ff ,t 
        tp ,nk =counts_at (sc ,tem ,bt );TP +=tp ;NK +=nk ;GT +=gt_of (tem )
    return TP ,NK ,GT 


def f1 (TP ,NK ,GT ):
    p =TP /max (NK ,1 );rr =TP /max (GT ,1 );return 2 *p *rr /max (p +rr ,1e-9 )


W_ ,P_ =WORK &(MF ==1 ),WORK &(MF ==0 )
ens =0.5 *(rank01 (s_rf ,WORK )+rank01 (s_mlp ,WORK ))
cands ={"RF(rich)":s_rf ,"MLP(+emb)":s_mlp ,"SetNet":s_set ,"ens(RF+MLP)":ens }
print (f"\n{'skor':16s} {'WEI F1':>8s} {'PXC F1':>8s} {'ALL(pool) F1':>14s}")
per ={}
for nm ,sc in cands .items ():
    w =nested_counts (sc ,W_ );p =nested_counts (sc ,P_ );a =nested_counts (sc ,WORK )
    per [nm ]=(w ,p )
    print (f"{nm :16s} {f1 (*w ):8.4f} {f1 (*p ):8.4f} {f1 (*a ):14.4f}")

print (f"\n=== URETICI-BASINA YONLENDIRME (WEI ve PXC icin en iyi model AYRI, sayilar HAVUZLANIR) ===")
best =None 
for wn ,(w ,_ )in per .items ():
    for pn ,(_ ,p )in per .items ():
        TP ,NK ,GT =w [0 ]+p [0 ],w [1 ]+p [1 ],w [2 ]+p [2 ]
        a =f1 (TP ,NK ,GT )
        if best is None or a >best [0 ]:best =(a ,wn ,pn ,f1 (*w ),f1 (*p ))
print (f"  EN IYI: WEI<-{best [1 ]} ({best [3 ]:.4f}) + PXC<-{best [2 ]} ({best [4 ]:.4f})  =>  AGREGAT ALL {best [0 ]:.4f}")
base_all =f1 (*nested_counts (s_rf ,WORK ))
print (f"  RF(rich) tek-model agregat: {base_all :.4f}   ->  YONLENDIRME KAZANCI {best [0 ]-base_all :+.4f}")
json .dump ({"best_route":{"WEI":best [1 ],"PXC":best [2 ],"aggregate_ALL":best [0 ],
"WEI_F1":best [3 ],"PXC_F1":best [4 ]},
"single_model_rf_ALL":base_all },open ("results/p2b_route.json","w"),indent =1 )
