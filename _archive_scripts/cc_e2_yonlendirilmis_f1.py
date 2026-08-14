# -*- coding: utf-8 -*-
"""CC-E step 2: geometriden regime tahmini + YONLENDIRILMIS CP-F1.

SORU: robot unknown a parcada 'this very-CP' diyebilir mi? Diyebilirse conn_promote'u SADECE
orada acar; diyemezse promote uygulanamaz (kuresel uygulama -0.0087 with ZARARLI).

KIYAS NOKTALARI (same 759 temiz part, same protocol):
  each places mv10          0.8101   <- mevcut most iyi
  ORACLE yonlendirme      0.8177   <- real CP sayisini BILSEYDIK (ulasilamaz upper boundary)
  yonlendirilmis          ?        <- only geometriden

Yonlendirici GIRDILERI (all of them inference aninda elde present, metadata YOK):
  part bbox boyutlari/kosegen/hacim + promote'suz candidate count and yogunlugu.
Yonlendirici AILE-DISI egitilir -- test ailesindeki parts egitimde gorulmez.
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
HIGH_CP =8 
W_HIGH =0.105 # real korpusta very-CP orani


def load (path ):
    r =np .load (path ,allow_pickle =True )
    rp =[str (v )for v in r ["part_ids"]]
    pid =np .array ([rp [int (g )]for g in r ["groups"]])
    X =np .hstack ([r ["X13"],r ["XR"][:,0 :33 ]])
    seen ={rp [i ]:int (r ["seen"][i ])for i in range (len (rp ))}
    ngt_i ={int (g ):int (n )for g ,n in zip (r ["grp_ids"],r ["ngt"])}
    ngt ={rp [int (g )]:ngt_i [int (g )]for g in np .unique (r ["groups"])}
    return X ,r ["y"],pid ,seen ,ngt 


def f1_of (tp ,nk ,gt ):
    p =tp /max (nk ,1 );r =tp /max (gt ,1 )
    return 2 *p *r /max (p +r ,1e-9 )


def oof_scores (X ,y ,G ,seed =0 ):
    o =np .zeros (len (y ))
    for tr ,te in GroupKFold (5 ).split (X ,y ,G ):
        o [te ]=RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =seed ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
    return o 


def scored_f1 (y ,oof ,pid ,G ,NGT ,mask ,seed_split =0 ):
    """Ic-ice esikli F1 (threshold test ailelerinden ogrenilmez)."""
    uf =np .unique (G [mask ]);rs =np .random .RandomState (seed_split )
    uf =uf [rs .permutation (len (uf ))]
    tp =nk =gt =0 
    for fo in np .array_split (uf ,5 ):
        te =mask &np .isin (G ,fo );tr =mask &~np .isin (G ,fo )
        gt_tr =sum (NGT .get (q ,0 )for q in set (pid [tr ]))
        bt ,bf =THRS [0 ],-1.0 
        for t in THRS :
            k =tr &(oof >=t )
            f =f1_of (int (y [k ].sum ()),int (k .sum ()),gt_tr )
            if f >bf :bf ,bt =f ,t 
        k =te &(oof >=bt )
        tp +=int (y [k ].sum ());nk +=int (k .sum ());gt +=sum (NGT .get (q ,0 )for q in set (pid [te ]))
    return tp ,nk ,gt 


def main ():
    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}
    geom =json .load (open ("results/cc_e_partgeom.json"))

    A =load ("results/rich_mv10_all.npz")# promote KAPALI
    B =load ("results/rich_promote.npz")# promote OPEN
    common =({q for q in set (A [2 ])if A [3 ].get (q ,1 )==0 and q not in LOCK }
    &{q for q in set (B [2 ])if B [3 ].get (q ,1 )==0 and q not in LOCK })
    common ={q for q in common if q in geom and "ext"in geom [q ]}
    NGT =A [4 ]
    hi_true ={q for q in common if NGT .get (q ,0 )>=HIGH_CP }
    print (f"ortak+geometrili score kumesi {len (common )} part (gercek very-CP {len (hi_true )})\n")

    # ---- YONLENDIRICI: geometri + promote'suz candidate istatistigi -> 'very-CP mi' ----
    Xa ,ya ,pida ,_ ,_ =A 
    ncand ={q :int ((pida ==q ).sum ())for q in common }
    rows ,lab ,pids =[],[],[]
    for q in sorted (common ):
        g =geom [q ];e =g ["ext"]
        rows .append ([e [0 ],e [1 ],e [2 ],g ["diag"],g ["area_box"],g ["vol_box"],
        ncand [q ],ncand [q ]/max (g ["area_box"],1e-6 )*1e3 ,
        ncand [q ]/max (g ["diag"],1e-6 )])
        lab .append (1 if q in hi_true else 0 );pids .append (q )
    Xr =np .array (rows ,float );yr =np .array (lab );pids =np .array (pids )
    famr =np .array ([fams .get (q ,f"nr:{q }")for q in pids ])
    fid ={v :i for i ,v in enumerate (sorted (set (famr )))}
    Gr =np .array ([fid [v ]for v in famr ])
    pr =np .zeros (len (yr ))
    for tr ,te in GroupKFold (5 ).split (Xr ,yr ,Gr ):
        pr [te ]=RandomForestClassifier (300 ,min_samples_leaf =2 ,n_jobs =-1 ,
        random_state =0 ).fit (Xr [tr ],yr [tr ]).predict_proba (Xr [te ])[:,1 ]
    from sklearn .metrics import roc_auc_score 
    print (f"YONLENDIRICI (geometri->very-CP), aile-disi AUC = {roc_auc_score (yr ,pr ):.4f}")
    for thr in (0.3 ,0.4 ,0.5 ,0.6 ):
        pred =pr >=thr 
        tp =int ((pred &(yr ==1 )).sum ());fp =int ((pred &(yr ==0 )).sum ())
        fn =int ((~pred &(yr ==1 )).sum ())
        print (f"   threshold {thr }: correct very-CP {tp }/{int (yr .sum ())}, wrong sign {fp }, kacan {fn }")
    print ()

    # ---- each two arm for OOF skorlari ----
    out ={}
    res ={}
    for tag ,(X ,y ,pid ,seen ,ngt )in (("promote_kapali",A ),("promote_acik",B )):
        m =np .array ([q in common for q in pid ])
        fam =np .array ([fams .get (q ,f"nr:{q }")for q in pid ])
        fidx ={v :i for i ,v in enumerate (sorted (set (fam )))}
        G =np .array ([fidx [v ]for v in fam ])
        oof =oof_scores (X ,y ,G )
        res [tag ]=(y ,oof ,pid ,G ,m )

    def eval_route (route_hi ):
        """route_hi: very-CP diye ISARETLENEN part kumesi -> orada promote OPEN, digerinde KAPALI."""
        tot_tp =tot_nk =tot_gt =0 
        per ={}
        for tag ,sel in (("promote_kapali",lambda q :q not in route_hi ),
        ("promote_acik",lambda q :q in route_hi )):
            y ,oof ,pid ,G ,m =res [tag ]
            mask =m &np .array ([sel (q )for q in pid ])
            if mask .sum ()==0 :continue 
            tp ,nk ,gt =scored_f1 (y ,oof ,pid ,G ,NGT ,mask )
            tot_tp +=tp ;tot_nk +=nk ;tot_gt +=gt 
            per [tag ]=(tp ,nk ,gt )
        return f1_of (tot_tp ,tot_nk ,tot_gt ),per 

    def weighted (route_hi ):
        """Rejim ayrimli + corpus agirlikli (duz mean very-CP'yi extra temsil eder)."""
        vals ={}
        for reg ,ps in (("low",common -hi_true ),("very",hi_true )):
            tot_tp =tot_nk =tot_gt =0 
            for tag ,sel in (("promote_kapali",lambda q :q not in route_hi ),
            ("promote_acik",lambda q :q in route_hi )):
                y ,oof ,pid ,G ,m =res [tag ]
                mask =m &np .array ([(q in ps )and sel (q )for q in pid ])
                if mask .sum ()==0 :continue 
                tp ,nk ,gt =scored_f1 (y ,oof ,pid ,G ,NGT ,mask )
                tot_tp +=tp ;tot_nk +=nk ;tot_gt +=gt 
            vals [reg ]=f1_of (tot_tp ,tot_nk ,tot_gt )
        return (1 -W_HIGH )*vals ["low"]+W_HIGH *vals ["very"],vals 

    print (f"{'senaryo':<34}{'low':>9}{'very':>9}{'KORPUS-TEMSILI':>17}")
    scen ={"each places promote KAPALI":set (),
    "each places promote OPEN":set (common ),
    "ORACLE yonlendirme":set (hi_true )}
    for thr in (0.3 ,0.4 ,0.5 ,0.6 ):
        scen [f"geometri yonlendirme (threshold {thr })"]={pids [i ]for i in range (len (pids ))if pr [i ]>=thr }
    for name ,rh in scen .items ():
        w ,vals =weighted (rh )
        print (f"{name :<34}{vals ['low']:>9.4f}{vals ['very']:>9.4f}{w :>17.4f}")
        out [name ]={"low":vals ["low"],"high":vals ["very"],"weighted":w ,
        "n_routed_high":len (rh )}
    json .dump ({"router_auc":float (roc_auc_score (yr ,pr )),"scenarios":out },
    open ("results/cc_e_yonlendirme.json","w"),indent =1 )
    print ("\nmakbuz -> results/cc_e_yonlendirme.json")


if __name__ =="__main__":
    main ()
