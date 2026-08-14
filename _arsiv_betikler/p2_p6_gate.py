# -*- coding: utf-8 -*-
"""FAZ 2 / P6 -- INSAN ADIMI KAPI-KONTROLU (yet insan isi YOK, only measurement).
Kullanici kurali: "FP'lerin at least %60'i most extra 40 TIPTE toplanmiyorsa adjudication YAPMA."
'aile-opening tipi' = (PRODUCT ailesi, part-ici normalize konum kovasi, opening boyu kovasi):
same ailede same SLOT tekrarlar; a slot for single karar tum aileyi temizler.
NOT: onceki %86 olcumu bbox-proxy aile with yapilmisti -> here GERCEK STEP PRODUCT ailesiyle yenilenir.
En iyi mevcut modelin (rf(rich)+mlp(+emb) ensemble) OOF skorlariyla is computed."""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .neural_network import MLPClassifier 
from sklearn .preprocessing import StandardScaler 
from sklearn .pipeline import make_pipeline 
from sklearn .model_selection import GroupKFold 

r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF ,POS =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"],r ["pos"]
rp =[str (x )for x in r ["part_ids"]];ep =[str (x )for x in e ["part_ids"]]
emap ={p :i for i ,p in enumerate (ep )}
EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]))
for gi ,pid in enumerate (rp ):
    if pid not in emap :continue 
    ir =np .where (G ==gi )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
    if len (ir )==len (ie ):EM [ir ]=e ["EMB"][ie ]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]]);FULL =np .hstack ([RICH ,EM ])
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
FG =np .array ([{v :i for i ,v in enumerate (sorted (set (fam )))}[v ]for v in fam ])


def oof (build ,Xa ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (Xa [idx ],Y [idx ],FG [idx ]):
        o [idx [te ]]=build ().fit (Xa [idx ][tr ],Y [idx ][tr ]).predict_proba (Xa [idx ][te ])[:,1 ]
    return o 


def rank01 (s ):
    o =np .zeros (len (s ));i =np .where (WORK )[0 ]
    o [i ]=np .argsort (np .argsort (s [i ]))/max (len (i )-1 ,1 );return o 


sc =0.5 *(rank01 (oof (lambda :RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ),RICH ))
+rank01 (oof (lambda :make_pipeline (StandardScaler (),MLPClassifier ((128 ,48 ),max_iter =400 ,
random_state =0 ,early_stopping =True ,n_iter_no_change =15 )),FULL )))
# most iyi modelin calisma esigi ~ upper %; FP = y==0 and skor esigi gecen
thr =np .quantile (sc [WORK ],1 -Y [WORK ].mean ())# tahmini pozitif count ~ real pozitif count
surv =WORK &(Y ==0 )&(sc >=thr )
print (f"WORK {int (WORK .sum ())} candidate | {int (Y [WORK ].sum ())} TP | surviving-FP (calisma noktasi) {int (surv .sum ())}")

# aile-opening TIPI: (aile, normalize konum kovasi, boy kovasi)
key =[]
for i in range (len (Y )):
    g =int (G [i ]);m =np .where (G ==g )[0 ];P =POS [m ]
    lo ,hi =P .min (0 ),P .max (0 );nrm =(POS [i ]-lo )/np .maximum (hi -lo ,1e-6 )
    size =X13 [i ,0 ]
    key .append (f"{fam [i ]}|p{np .round (nrm ,1 ).tolist ()}|s{round (float (size )/3 )*3 }")
key =np .array (key )
kk ,cc =np .unique (key [surv ],return_counts =True )
o =np .argsort (-cc );kk ,cc =kk [o ],cc [o ]
cum =np .cumsum (cc )/cc .sum ()
print (f"\n=== FP KUTLESININ 'aile-opening tipi' YOGUNLASMASI (GERCEK PRODUCT ailesi) ===")
print (f"  farkli tip sayisi: {len (kk )}")
for k in (10 ,20 ,40 ,60 ):
    if k <=len (cc ):print (f"  en buyuk {k :3d} tip -> FP kutlesinin %{100 *cum [k -1 ]:.0f}'i")
n40 =cum [min (39 ,len (cum )-1 )]
ok =n40 >=0.60 
print (f"\n=== P6 KAPISI: >=%60 kutle <=40 tipte? -> {'GECTI'if ok else 'KALDI'} (%{100 *n40 :.0f}) ===")
if ok :
    print ("  -> insan adjudication'i MESRU (50-100 hedefli 'wire/tool/unsure' sorusu). Yine de EN SON is done.")
    json .dump ([{"type":k ,"n_fp":int (c )}for k ,c in zip (kk [:60 ],cc [:60 ])],
    open ("results/p6_target_types.json","w"),indent =1 )
    print ("  -> results/p6_target_types.json (hedef list, FAZ B for hazir)")
else :
    print ("  -> RULE GEREGI ADJUDICATION YAPILMAZ (FP dagilmis, single kararlar kutleyi temizlemez).")
