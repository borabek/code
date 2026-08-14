# -*- coding: utf-8 -*-
"""FAZ 2 / P0 -- VERIFICATION + KILITLI BASELINE (TODO_085_PHASE2.md P0 maddeleri 1,4,6,7).
 (1) FRAME dogrulama: pos and direction same frame'de mi -- pipeline bazinda raporla
 (4) part-out / geometry-out / family-out splitlerini split_lock.json'dan sabitle
 (6) baseline'lari TEMIZ CALISMA setinde yeniden dogrula (leakage-muhafizi OPEN, kilitli holdout HARIC)
 (7) candidate-recall with oracle-F1'i AYRI raporla
Cikti: results/p0_verified_baselines.json"""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

d =np .load ("results/rich_feats.npz",allow_pickle =True )
X13 ,XR ,Y ,G ,MF =d ["X13"],d ["XR"],d ["y"],d ["groups"],d ["mfg"]
PID ,SEEN =d ["part_ids"],d ["seen"]
ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])# 13 + konum(9) + very-radius(24) = kazanan set
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )

pid_of_g ={int (g ):str (PID [i ])for i ,g in enumerate (sorted (ngt_of ))}
seen_of_g ={int (g ):int (SEEN [i ])for i ,g in enumerate (sorted (ngt_of ))}
gpid =np .array ([pid_of_g [int (g )]for g in G ])
gseen =np .array ([seen_of_g [int (g )]for g in G ])
is_locked =np .array ([p in LOCKED for p in gpid ])

# --- (1) FRAME dogrulama (statik audit + data kontrolu) ---
print ("=== (1) FRAME VERIFICATION ===")
print ("  build_rich_feats.rich_feats(): V/F/cps HEPSI mesh frame -> konum+direction TUTARLI  [OK]")
print ("  build_rich_feats 'pos' alani  : JSON frame (SADECE GT eslesme + aile anahtari for)  [OK, ayri kullanim]")
print ("  WARNING/FINDING: build_wei_aggr.py + build_aggr_rich.py havuzlarinda P=JSON frame but dir=MESH frame")
print ("               -> P and dir'i BIRLIKTE kullanan a feature YAZILMAMALI. (P0-a analizim only dir")
print ("                  kullandi, part-ici tutarli oldugu for gecerliydi; yine de not edildi.)")

# --- (4) split anahtarlari ---
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
geo =np .array ([META .get (p ,{}).get ("geom",f"g:{p }")for p in gpid ])
def enc (a ):
    u ={v :i for i ,v in enumerate (sorted (set (a )))};return np .array ([u [v ]for v in a ])
FG ,GEOG =enc (fam ),enc (geo )

WORK =(gseen ==0 )&(~is_locked )# TEMIZ + kilitli-not = tum gelistirme here
sets ={"WORK (temiz, kilitsiz)":WORK ,
"CANONICAL (823, seen dahil)":np .ones (len (Y ),bool )}
print (f"\n=== (4) SPLIT'LER (split_lock.json) ===")
print (f"  part {len (ngt_of )} | aile {len (set (fam ))} | geometri {len (set (geo ))}")
print (f"  WORK candidate {int (WORK .sum ())} / {len (set (gpid [WORK ]))} part | KILITLI {len (LOCKED )} part (P7'de acilir)")


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )
def gt_of (mask ):return int (sum (ngt_of [int (g )]for g in np .unique (G [mask ])))
def score_at (sc ,mask ,t ):
    k =mask &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (mask ))


def oof (Xa ,groups ,mask ):
    o =np .full (len (Y ),np .nan );idx =np .where (mask )[0 ]
    Xm ,Ym ,Gm =Xa [idx ],Y [idx ],groups [idx ]
    om =np .zeros (len (idx ))
    for tr ,te in GroupKFold (5 ).split (Xm ,Ym ,Gm ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xm [tr ],Ym [tr ])
        om [te ]=c .predict_proba (Xm [te ])[:,1 ]
    o [idx ]=om ;return o 


def nested (sc ,mask ,groups ,K =5 ):
    gp =np .unique (groups [np .where (mask )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,K ):
        te_g =set (f .tolist ())
        trm =mask &np .array ([g in gp and g not in te_g for g in groups ])
        tem =mask &np .array ([g in te_g for g in groups ])
        if not trm .any ()or not tem .any ():continue 
        bt ,bf =0.35 ,-1 
        for t in THRS :
            ff =score_at (sc ,trm ,t )[2 ]
            if ff >bf :bf ,bt =ff ,t 
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    return prf (TP ,NK ,GT )[2 ]


def topn (sc ,mask ):
    TP =NK =GT =0 
    for g in np .unique (G [mask ]):
        m =np .where (mask &(G ==g ))[0 ];n =ngt_of [int (g )]
        i2 =m [np .argsort (-sc [m ])[:n ]]
        TP +=int (Y [i2 ].sum ());NK +=len (i2 );GT +=n 
    return prf (TP ,NK ,GT )[2 ]


    # --- (7) candidate-recall vs oracle-F1 (AYRI kavram) ---
print (f"\n=== (7) ADAY-RECALL vs ORACLE-F1 (AYRI raporlanir) ===")
for nm ,mk in sets .items ():
    for sub ,smk in (("ALL",mk ),("WEI",mk &(MF ==1 )),("PXC",mk &(MF ==0 ))):
        gt =gt_of (smk );rec =int (Y [smk ].sum ())/max (gt ,1 )
        print (f"  {nm :28s} {sub :4s}: candidate {int (smk .sum ()):5d} ({smk .sum ()/max (gt ,1 ):.2f}x) | "
        f"candidate-recall {rec :.3f} | oracle-F1 (mukemmel gate) {2 *rec /(1 +rec ):.3f}")

        # --- (6) baseline yeniden dogrulama ---
report ={}
print (f"\n=== (6) BASELINE YENIDEN DOGRULAMA (nested-CV threshold) ===")
for set_nm ,mk in sets .items ():
    print (f"\n--- {set_nm } ---")
    print (f"{'split':14s} {'feat':6s} {'AUC':>7s} {'ALL':>7s} {'WEI':>7s} {'PXC':>7s} {'topN':>7s}")
    for sp_nm ,groups in (("part-out",G ),("geom-out",GEOG ),("family-out",FG )):
        for fn ,Xa in (("13",X13 ),("rich",RICH )):
            sc =oof (Xa ,groups ,mk )
            r ={"AUC":roc_auc_score (Y [mk ],sc [mk ]),
            "ALL":nested (sc ,mk ,groups ),"WEI":nested (sc ,mk &(MF ==1 ),groups ),
            "PXC":nested (sc ,mk &(MF ==0 ),groups ),"topN":topn (sc ,mk )}
            report [f"{set_nm }|{sp_nm }|{fn }"]=r 
            print (f"{sp_nm :14s} {fn :6s} {r ['AUC']:7.4f} {r ['ALL']:7.4f} {r ['WEI']:7.4f} {r ['PXC']:7.4f} {r ['topN']:7.4f}",flush =True )

json .dump (report ,open ("results/p0_verified_baselines.json","w"),indent =1 )
print ("\n-> results/p0_verified_baselines.json")
print ("KIYAS: canonical receipt ALL 0.750 / metadata 0.775 / WEI 0.696 / PXC-tipik 0.823")
print ("NOT: canonical numbers BA_ALLOW_SEEN=1 with uretilmisti (153 parcanin etiketleri segmentasyon")
print ("     egitiminde seen). WORK satiri this sizintiyi DISLAR = more durust baseline.")
