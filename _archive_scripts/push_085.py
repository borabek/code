# -*- coding: utf-8 -*-
"""0.85'E ITME (night 2. hamle, makine-only). Zengin gate 0.783'e getirdi (part-out) / 0.763 (aile-out).
Kazanan blok KONUM'du -> that damari derinlestir + siniflandiriciyi guclendir.
YENI BAGLAM feature'lari (POS'defn turetilir, etiketsiz, inference'ta hesaplanabilir):
  E1 PCA-eksenleri: candidate-bulutunun ana eksenlerinde u,v,w koordinati (ray yonu = most uzun axis)  [3]
  E2 each eksende part-ici RANK (normalize)                                                     [3]
  E3 merkeze uzaklik + most yakin komsu mesafeleri (1/2/3. komsu)                                  [4]
  E4 very-olcekli yerel yogunluk (5/10/20mm icindeki candidate count, normalize)                      [3]
  E5 AYNA-ES: bulut merkezine according to simetrik esi present mi + mesafesi                                [2]
SINIFLANDIRICI: RF vs HistGradientBoosting vs mean-ensemble.
Hepsi PARCA-out + AILE-out olculur (dogrulanmis kazanc = aile-out'ta da stopped)."""
import numpy as np 
from sklearn .ensemble import RandomForestClassifier ,HistGradientBoostingClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

d =np .load ("results/rich_feats.npz",allow_pickle =True )
X13 ,XR ,Y ,G ,MF ,POS =d ["X13"],d ["XR"],d ["y"],d ["groups"],d ["mfg"],d ["pos"]
ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
BASE =np .hstack ([X13 ,XR [:,0 :33 ]])# kazanan set: 13 + konum + very-radius

E =np .zeros ((len (Y ),15 ))
for g in np .unique (G ):
    m =np .where (G ==g )[0 ];P =POS [m ];n =len (m )
    c =P .mean (0 );Q =P -c 
    if n >=3 :
        U ,S ,Vt =np .linalg .svd (Q ,full_matrices =False );A =Q @Vt .T # PCA koordinatlari
    else :
        A =Q .copy ()
    rngA =np .maximum (A .max (0 )-A .min (0 ),1e-6 )
    E [m ,0 :3 ]=(A -A .min (0 ))/rngA # E1
    E [m ,3 :6 ]=np .argsort (np .argsort (A ,axis =0 ),axis =0 )/max (n -1 ,1 )# E2
    dist =np .linalg .norm (Q ,axis =1 );E [m ,6 ]=dist /(dist .max ()+1e-9 )
    DD =np .linalg .norm (P [:,None ]-P [None ],axis =-1 );np .fill_diagonal (DD ,np .inf )
    sd =np .sort (DD ,axis =1 )
    for j in range (3 ):
        E [m ,7 +j ]=sd [:,j ]if n >j +1 else 50.0 # E3
    for j ,r in enumerate ((5.0 ,10.0 ,20.0 )):
        E [m ,10 +j ]=(DD <=r ).sum (1 )/max (n -1 ,1 )# E4
    mir =2 *c -P # E5 ayna-es
    dm =np .linalg .norm (P [:,None ]-mir [None ],axis =-1 );np .fill_diagonal (dm ,np .inf )
    md =dm .min (1 )if n >1 else np .array ([50.0 ]*n )
    E [m ,13 ]=np .minimum (md ,50.0 );E [m ,14 ]=(md <=3.0 ).astype (float )

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
def score_at (sc ,mask ,t ):
    k =mask &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (mask ))


def mk (kind ,seed =0 ):
    if kind =="rf":return RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed )
    return HistGradientBoostingClassifier (max_iter =400 ,learning_rate =0.06 ,min_samples_leaf =10 ,random_state =seed )


def oof (Xa ,groups ,kind ):
    o =np .zeros (len (Y ))
    for tr ,te in GroupKFold (5 ).split (Xa ,Y ,groups ):
        if kind =="ens":
            a =mk ("rf").fit (Xa [tr ],Y [tr ]).predict_proba (Xa [te ])[:,1 ]
            b =mk ("gb").fit (Xa [tr ],Y [tr ]).predict_proba (Xa [te ])[:,1 ]
            o [te ]=0.5 *(a +b )
        else :
            o [te ]=mk (kind ).fit (Xa [tr ],Y [tr ]).predict_proba (Xa [te ])[:,1 ]
    return o 


def nested (sc ,mask ,groups ,K =5 ):
    gp =np .unique (groups [np .where (mask )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,K ):
        te_g =set (f .tolist ())
        trm =mask &np .array ([g not in te_g for g in groups ])
        tem =mask &np .array ([g in te_g for g in groups ])
        bt ,bf =0.35 ,-1 
        for t in THRS :
            ff =score_at (sc ,trm ,t )[2 ]
            if ff >bf :bf ,bt =ff ,t 
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    return prf (TP ,NK ,GT )[2 ]


def topn (sc ,mask ):
    TP =NK =GT =0 
    for g in np .unique (G [mask ]):
        m =np .where (mask &(G ==g ))[0 ];n =ngt_of [g ]
        i2 =m [np .argsort (-sc [m ])[:n ]]
        TP +=int (Y [i2 ].sum ());NK +=len (i2 );GT +=n 
    return prf (TP ,NK ,GT )[2 ]


BE =np .hstack ([BASE ,E ])
print (f"{len (Y )} candidate | {len (ngt_of )} part | BASE {BASE .shape [1 ]}d | +baglam {BE .shape [1 ]}d\n")
for split_nm ,groups in (("PARCA-out",G ),("AILE-out (KATI)",FG )):
    print (f"================ {split_nm } ================")
    for fn ,Xa in (("BASE(46d)",BASE ),("BASE+baglam(61d)",BE )):
        for kind in ("rf","gb","ens"):
            sc =oof (Xa ,groups ,kind )
            print (f"  {fn :17s} {kind :4s} AUC {roc_auc_score (Y ,sc ):.4f} | ALL {nested (sc ,masks ['ALL'],groups ):.4f}"
            f" | WEI {nested (sc ,masks ['WEI'],groups ):.4f} | PXC {nested (sc ,masks ['PXC'],groups ):.4f}"
            f" | topN {topn (sc ,masks ['ALL']):.4f}",flush =True )
    print ()
print ("KIYAS: canonical base 0.750 | night-1 zengin gate: part-out 0.7826 / aile-out 0.7628")
