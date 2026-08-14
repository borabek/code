# -*- coding: utf-8 -*-
"""TEL-A: mevcut gate'in calisma noktasinda HAYATTA KALAN FP'leri karakterize et.

WHY ONCE BU: "hayatta kalan FP'ler TP'lerle geometrik as ozdes" yargisi O GUNKU ozelliklerle
verilmisti. Ne olduklarini GORMEDEN new feature icat etmek this projede three times olu kaldiraca path acti.
Once teshis, after feature.

SORULAR:
  S1  Kac FP hayatta kaliyor, hangi parcalarda yogunlasiyor?
  S2  FP'ler TP'lerin YANINDA mi? (alet agzi hipotezi: tel girisiyle karakteristik ofsette eslesir)
  S3  Mevcut 41 ozellikten hangileri FP/TP ayrimini ZATEN yapabiliyor, hangileri kor?
  S4  FP'ler each other benziyor mu (single type mi, otherwise different sebepler mi)?
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

FILES =["results/rich_feats.npz","results/rich_ds3.npz","results/rich_new863.npz"]
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )


def load (paths ):
    X ,Y ,PID ,POS ,NGT ,SEEN =[],[],[],[],{},{}
    got =set ()
    for p in paths :
        r =np .load (p ,allow_pickle =True )
        rp =[str (v )for v in r ["part_ids"]]
        gp =np .array ([rp [int (g )]for g in r ["groups"]])
        keep =np .array ([q not in got for q in gp ])
        got |=set (gp [keep ])
        X .append (np .hstack ([r ["X13"],r ["XR"][:,0 :33 ]])[keep ])
        Y .append (r ["y"][keep ]);PID .append (gp [keep ]);POS .append (r ["pos"][keep ])
        for i ,q in enumerate (rp ):
            if q in set (gp [keep ]):
                SEEN [q ]=int (r ["seen"][i ])
        ngt ={int (g ):int (n )for g ,n in zip (r ["grp_ids"],r ["ngt"])}
        for g in np .unique (r ["groups"][keep ]):
            NGT [rp [int (g )]]=ngt [int (g )]
    return (np .vstack (X ),np .concatenate (Y ),np .concatenate (PID ),
    np .vstack (POS ),NGT ,SEEN )


def main ():
    lock =json .load (open ("results/split_lock.json"))
    LOCK =set (lock ["locked_parts"])
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}

    X ,Y ,PID ,POS ,NGT ,SEEN =load (FILES )
    m =np .array ([q not in LOCK for q in PID ])
    X ,Y ,PID ,POS =X [m ],Y [m ],PID [m ],POS [m ]
    fam =np .array ([fams .get (q ,f"nr:{q }")for q in PID ])
    fid ={v :i for i ,v in enumerate (sorted (set (fam )))}
    FG =np .array ([fid [v ]for v in fam ])

    oof =np .zeros (len (Y ))
    for tr ,te in GroupKFold (5 ).split (X ,Y ,FG ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (X [tr ],Y [tr ])
        oof [te ]=c .predict_proba (X [te ])[:,1 ]

    clean =np .array ([SEEN .get (q ,1 )==0 for q in PID ])# skorlama SADECE temiz parcalarda
    def f1_at (t ,mask ):
        k =mask &(oof >=t )
        tp =int (Y [k ].sum ());nk =int (k .sum ())
        gt =int (sum (NGT .get (q ,0 )for q in set (PID [mask ])))
        p =tp /max (nk ,1 );r =tp /max (gt ,1 )
        return 2 *p *r /max (p +r ,1e-9 ),tp ,nk -tp 
    thr =max (THRS ,key =lambda t :f1_at (t ,clean )[0 ])
    f1 ,n_tp ,n_fp =f1_at (thr ,clean )
    print (f"calisma noktasi: threshold {thr }  F1 {f1 :.4f}  |  TP {n_tp }  HAYATTA KALAN FP {n_fp }\n")

    keep =clean &(oof >=thr )
    is_fp =keep &(Y ==0 )
    is_tp =keep &(Y ==1 )

    # --- S1: FP'ler hangi parcalarda yogunlasiyor -------------------------------------------------
    import collections 
    per =collections .Counter (PID [is_fp ])
    parts_scored =len (set (PID [clean ]))
    print (f"S1  FP'li part {len (per )}/{parts_scored }  |  part basi ort {n_fp /max (parts_scored ,1 ):.2f}")
    print ("    en cok FP ureten 8 part:",", ".join (f"{k }({v })"for k ,v in per .most_common (8 )))
    hi =sum (v for k ,v in per .items ()if v >=3 )
    print (f"    FP'lerin %{100 *hi /max (n_fp ,1 ):.0f}'i, 3+ FP ureten parcalardan geliyor "
    f"({sum (1 for v in per .values ()if v >=3 )} part)\n")

    # --- S2: FP'ler TP'lerin next to mi (alet agzi hipotezi) --------------------------------------
    d_fp ,d_tp =[],[]
    for q in set (PID [keep ]):
        pm =(PID ==q )
        tp_pos =POS [pm &is_tp ];fp_pos =POS [pm &is_fp ]
        if not len (tp_pos ):
            continue 
        for p_ in fp_pos :
            d_fp .append (float (np .linalg .norm (tp_pos -p_ ,axis =1 ).min ()))
        for i ,p_ in enumerate (tp_pos ):# TP'nin most yakin DIGER TP'ye uzakligi (kiyas)
            o =np .delete (tp_pos ,i ,axis =0 )
            if len (o ):d_tp .append (float (np .linalg .norm (o -p_ ,axis =1 ).min ()))
    d_fp =np .array (d_fp );d_tp =np .array (d_tp )
    if len (d_fp ):
        print (f"S2  FP -> en yakin TP mesafesi (mm): medyan {np .median (d_fp ):5.1f} "
        f"| %25 {np .percentile (d_fp ,25 ):5.1f} | %75 {np .percentile (d_fp ,75 ):5.1f}")
        print (f"    TP -> en yakin TP mesafesi (mm): medyan {np .median (d_tp ):5.1f} "
        f"| %25 {np .percentile (d_tp ,25 ):5.1f} | %75 {np .percentile (d_tp ,75 ):5.1f}")
        for lo ,hi_ in [(0 ,3 ),(3 ,6 ),(6 ,10 ),(10 ,20 ),(20 ,1e9 )]:
            n =int (((d_fp >=lo )&(d_fp <hi_ )).sum ())
            print (f"      {lo :>3}-{hi_ if hi_ <1e9 else '..':>3} mm: {n :>4} FP ({100 *n /len (d_fp ):4.1f}%)")
        print ()

        # --- S3: mevcut ozelliklerden hangisi already ayirt ediyor --------------------------------------
    from scipy .stats import mannwhitneyu 
    names =([f"x13_{i }"for i in range (13 )]+[f"rich_{i }"for i in range (X .shape [1 ]-13 )])
    rows =[]
    for j in range (X .shape [1 ]):
        a ,b =X [is_fp ,j ],X [is_tp ,j ]
        if len (a )<10 or len (b )<10 :continue 
        try :
            u ,_ =mannwhitneyu (a ,b ,alternative ="two-sided")
            auc =u /(len (a )*len (b ))# 0.5 = ayirt etmiyor
        except Exception :
            continue 
        rows .append ((abs (auc -0.5 ),auc ,names [j ],float (np .median (a )),float (np .median (b ))))
    rows .sort (reverse =True )
    print ("S3  HAYATTA KALAN FP vs TP ayrimi (mevcut ozellikler, AUC 0.5 = kor):")
    for s ,auc ,nm ,ma ,mb in rows [:8 ]:
        print (f"      {nm :<10} AUC {auc :.3f}   FP med {ma :9.3f}   TP med {mb :9.3f}")
    print (f"      ... en iyi mevcut ozellik bile AUC {rows [0 ][1 ]:.3f} "
    f"({'ZAYIF'if abs (rows [0 ][1 ]-0.5 )<0.15 else 'kayda deger'})\n")

    json .dump ({"threshold":float (thr ),"f1":f1 ,"n_tp":n_tp ,"n_fp":n_fp ,
    "parts_scored":parts_scored ,"parts_with_fp":len (per ),
    "fp_to_tp_dist_median_mm":float (np .median (d_fp ))if len (d_fp )else None ,
    "tp_to_tp_dist_median_mm":float (np .median (d_tp ))if len (d_tp )else None ,
    "top_existing_feature_auc":float (rows [0 ][1 ])if rows else None ,
    "worst_parts":per .most_common (20 )},
    open ("results/tel_a_fp_teshis.json","w"),indent =1 )
    print ("receipt -> results/tel_a_fp_teshis.json")


if __name__ =="__main__":
    main ()
