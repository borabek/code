# -*- coding: utf-8 -*-
"""FAZ 2 / P1 -- GO/NO-GO degerlendirmesi (kullanici esikleri: ALL +0.015, WEI +0.020, PXC kaybi <=0.005).
Karsilastirma: linear probe (logistic) / RF / small MLP -- AYNI OOF split'lerinde.
Taban: rich (13+konum+very-radius). Aday: rich + embedding feature'lari.
TUM olcumler TEMIZ CALISMA setinde (leakage-muhafizi open, KILITLI holdout haric)."""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .linear_model import LogisticRegression 
from sklearn .neural_network import MLPClassifier 
from sklearn .preprocessing import StandardScaler 
from sklearn .pipeline import make_pipeline 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]

# --- two npz'yi part_id + part-ici order uzerinden hizala ---
rp =[str (x )for x in r ["part_ids"]];ep =[str (x )for x in e ["part_ids"]]
rg ,eg =r ["groups"],e ["groups"]
rmap ={p :i for i ,p in enumerate (rp )}
keep_r ,keep_e =[],[]
for gi_e ,pid in enumerate (ep ):
    if pid not in rmap :continue 
    gi_r =rmap [pid ]
    ir =np .where (rg ==gi_r )[0 ];ie =np .where (eg ==gi_e )[0 ]
    if len (ir )!=len (ie ):continue # candidate count tutmuyorsa parcayi skip (safe)
    keep_r +=list (ir );keep_e +=list (ie )
keep_r =np .array (keep_r );keep_e =np .array (keep_e )
X13 =r ["X13"][keep_r ];XR =r ["XR"][keep_r ];Y =r ["y"][keep_r ];MF =r ["mfg"][keep_r ]
G =r ["groups"][keep_r ];EMB =e ["EMB"][keep_e ]
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt_of =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])
FULL =np .hstack ([RICH ,EMB ])
print (f"hizalandi: {len (Y )} candidate / {len (set (gpid ))} part | rich {RICH .shape [1 ]}d + emb {EMB .shape [1 ]}d = {FULL .shape [1 ]}d")

fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
def enc (a ):
    u ={v :i for i ,v in enumerate (sorted (set (a )))};return np .array ([u [v ]for v in a ])
FG =enc (fam )
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":WORK ,"WEI":WORK &(MF ==1 ),"PXC":WORK &(MF ==0 )}
print (f"WORK: {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part\n")


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );rr =tp /max (gt ,1 );return 2 *p *rr /max (p +rr ,1e-9 )
def gt_of (m ):return int (sum (ngt_of [int (g )]for g in np .unique (G [m ])))
def sc_at (sc ,m ,t ):
    k =m &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (m ))


def clf (kind ,seed =0 ):
    if kind =="rf":return RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed )
    if kind =="lin":return make_pipeline (StandardScaler (),LogisticRegression (max_iter =2000 ,C =0.5 ))
    return make_pipeline (StandardScaler (),MLPClassifier ((128 ,48 ),max_iter =400 ,random_state =seed ,
    early_stopping =True ,n_iter_no_change =15 ))


def oof (Xa ,groups ,kind ,seed =0 ):
    o =np .full (len (Y ),np .nan );idx =np .where (WORK )[0 ]
    Xm ,Ym ,Gm =Xa [idx ],Y [idx ],groups [idx ];om =np .zeros (len (idx ))
    for tr ,te in GroupKFold (5 ).split (Xm ,Ym ,Gm ):
        om [te ]=clf (kind ,seed ).fit (Xm [tr ],Ym [tr ]).predict_proba (Xm [te ])[:,1 ]
    o [idx ]=om ;return o 


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


res ={}
for sp_nm ,groups in (("part-out",G ),("family-out",FG )):
    print (f"================ {sp_nm } ================")
    print (f"{'set':6s} {'clf':4s} {'AUC':>7s} {'ALL':>7s} {'WEI':>7s} {'PXC':>7s} {'topN':>7s}")
    for fn ,Xa in (("rich",RICH ),("+emb",FULL )):
        for kind in ("rf","lin","mlp"):
            sc =oof (Xa ,groups ,kind )
            row ={k :nested (sc ,masks [k ],groups )for k in masks }
            row ["AUC"]=roc_auc_score (Y [WORK ],sc [WORK ]);row ["topN"]=topn (sc ,WORK )
            res [f"{sp_nm }|{fn }|{kind }"]=row 
            print (f"{fn :6s} {kind :4s} {row ['AUC']:7.4f} {row ['ALL']:7.4f} {row ['WEI']:7.4f} {row ['PXC']:7.4f} {row ['topN']:7.4f}",flush =True )
    print ()

print ("=== GO/NO-GO (kullanici esikleri: ALL >=+0.015, WEI >=+0.020, PXC kaybi <=0.005) ===")
go_any =False 
for sp in ("part-out","family-out"):
    for kind in ("rf","lin","mlp"):
        b ,a =res [f"{sp }|rich|{kind }"],res [f"{sp }|+emb|{kind }"]
        dA ,dW ,dP =a ["ALL"]-b ["ALL"],a ["WEI"]-b ["WEI"],a ["PXC"]-b ["PXC"]
        ok =(dA >=0.015 )and (dW >=0.020 )and (dP >=-0.005 )
        go_any |=ok 
        print (f"  {sp :11s} {kind :4s}: dALL {dA :+.4f} | dWEI {dW :+.4f} | dPXC {dP :+.4f}  -> {'GO'if ok else 'no'}")
json .dump (res ,open ("results/p1_embed_eval.json","w"),indent =1 )
print (f"\nKARAR: {'EMBEDDING KOLU OPEN (GO)'if go_any else 'GO SAGLANMADI -> embedding kolunu KAPAT (kullanici kurali)'}")
print ("-> results/p1_embed_eval.json")
