# -*- coding: utf-8 -*-
"""URUN v2 PAKETI: 2026-07-28/29 gecesinde measured_path +0.053'luk yapilandirmayi TEK KOMUTLA produces.

    .venv/Scripts/python.exe product_v2_paket.py            # yapilandirmayi dogrula + receipt yaz
    .venv/Scripts/python.exe product_v2_paket.py --holdout  # KILITLI HOLDOUT single atis (geri donusu YOK)

YAPILANDIRMA (three measured_path degisiklik, all of them WORK OOF with dogrulandi):
  1. min_v10 post-isleme    (min_v 10 / vertex_conf 0.30 / cluster 3.0 / dedupe 6.0)
  2. ExtraTrees(800, leaf2) gate  -- RandomForest instead of
  3. conn_promote 0.25, SADECE geometri-yonlendiricisi very-CP dedigi parcalarda

TEZ SADAKATI: segmentasyon agi, korpusu and remesh hedefi (6000, izotropik) DEGISMEDI. Uc degisiklik
de tezin ustune bizim muhendisligimiz which is katmanda (post-isleme + gate), tezin own boru hattinda
not. Segmentasyon korpusu yayinlanmis insan-etiketli bolunmesiyle aynen duruyor.

OLCULEN (corpus-temsili CP-F1, 759 temiz part, aile-disi GroupKFold(5), ic-ice threshold, 3 seed):
    dondurulmus urun            0.7892
    + min_v10                   0.8101
    + ExtraTrees + yonlendirme  0.8423     (+0.053 total)
"""
import argparse ,json ,os ,sys ,time 
import numpy as np 
from sklearn .ensemble import ExtraTreesClassifier ,RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import cc_e2_yonlendirilmis_f1 as M 

CONFIG ={
"name":"urun_v2_2026_07_29",
"postproc":{"min_vertices":10 ,"vertex_confidence_mask":0.30 ,
"cluster_mm":3.0 ,"dedupe_mm":6.0 },
"gate":{"model":"ExtraTreesClassifier","n_estimators":800 ,"min_samples_leaf":2 },
"conn_promote":{"threshold":0.25 ,"applied":"only where the geometry router predicts high-CP",
"router_threshold":0.30 },
"unchanged_from_thesis":["segmentation network","segmentation corpus + published split",
"isotropic remesh target 6000"],
}
ARM_OFF ="results/rich_mv10_all.npz"# promote KAPALI
ARM_ON ="results/rich_promote.npz"# promote OPEN
ROUTER_THR =0.30 


def _router (pids_sorted ,geom ,ncand ,hi_true ,fams ,seed =0 ):
    rows ,lab =[],[]
    for q in pids_sorted :
        g =geom [q ];e =g ["ext"]
        rows .append ([e [0 ],e [1 ],e [2 ],g ["diag"],g ["area_box"],g ["vol_box"],ncand [q ],
        ncand [q ]/max (g ["area_box"],1e-6 )*1e3 ,ncand [q ]/max (g ["diag"],1e-6 )])
        lab .append (1 if q in hi_true else 0 )
    Xr =np .array (rows ,float );yr =np .array (lab )
    fam =np .array ([fams .get (q ,f"nr:{q }")for q in pids_sorted ])
    fid ={v :i for i ,v in enumerate (sorted (set (fam )))}
    G =np .array ([fid [v ]for v in fam ])
    pr =np .zeros (len (yr ))
    for tr ,te in GroupKFold (5 ).split (Xr ,yr ,G ):
        pr [te ]=RandomForestClassifier (300 ,min_samples_leaf =2 ,n_jobs =-1 ,
        random_state =seed ).fit (Xr [tr ],yr [tr ]).predict_proba (Xr [te ])[:,1 ]
    return pr ,yr 


def evaluate (score_parts ,seeds =(0 ,1 ,2 ),label =""):
    """Yapilandirmayi verilen part kumesinde olc. Doner: dict."""
    lock =json .load (open ("results/split_lock.json"))
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}
    geom =json .load (open ("results/cc_e_partgeom.json"))
    A =M .load (ARM_OFF );B =M .load (ARM_ON )
    NGT =A [4 ]
    parts ={q for q in score_parts if q in geom and "ext"in geom [q ]
    and q in set (A [2 ])and q in set (B [2 ])}
    hi_true ={q for q in parts if NGT .get (q ,0 )>=M .HIGH_CP }
    ncand ={q :int ((A [2 ]==q ).sum ())for q in parts }
    ps =sorted (parts )

    out ={"n_parts":len (parts ),"n_high_cp":len (hi_true ),"label":label ,"seeds":[]}
    for s in seeds :
        pr ,_ =_router (ps ,geom ,ncand ,hi_true ,fams ,seed =s )
        route ={ps [i ]for i in range (len (ps ))if pr [i ]>=ROUTER_THR }
        res ={}
        for tag ,D in (("off",A ),("ten",B )):
            X ,y ,pid ,_ ,_ =D 
            m =np .array ([q in parts for q in pid ])
            fam =np .array ([fams .get (q ,f"nr:{q }")for q in pid ])
            fx ={v :i for i ,v in enumerate (sorted (set (fam )))}
            G =np .array ([fx [v ]for v in fam ])
            oof =np .zeros (len (y ))
            for tr ,te in GroupKFold (5 ).split (X ,y ,G ):
                oof [te ]=ExtraTreesClassifier (800 ,min_samples_leaf =2 ,n_jobs =-1 ,
                random_state =s ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
            res [tag ]=(y ,oof ,pid ,G ,m )
        vals ={}
        for reg ,sel_parts in (("low",parts -hi_true ),("high",hi_true )):
            T =N =Gt =0 
            for tag ,sel in (("off",lambda q :q not in route ),("ten",lambda q :q in route )):
                y ,oof ,pid ,G ,m =res [tag ]
                mask =m &np .array ([(q in sel_parts )and sel (q )for q in pid ])
                if mask .sum ()==0 :continue 
                tp ,nk ,gt =M .scored_f1 (y ,oof ,pid ,G ,NGT ,mask ,s )
                T +=tp ;N +=nk ;Gt +=gt 
            vals [reg ]=M .f1_of (T ,N ,Gt )
        vals ["corpus_weighted"]=(1 -M .W_HIGH )*vals ["low"]+M .W_HIGH *vals ["high"]
        out ["seeds"].append (vals )
    for k in ("low","high","corpus_weighted"):
        v =[d [k ]for d in out ["seeds"]]
        out [k ]={"mean":float (np .mean (v )),"sd":float (np .std (v ))}
    return out 


def evaluate_holdout (work_parts ,hold_parts ,seeds =(0 ,1 ,2 )):
    """GERCEK holdout protokolu: gate, threshold and router SADECE WORK'te ogrenilir,
    kilitli parcalara BIR KEZ uygulanir.

    NOTE -- this fonksiyon ayri duruyor because evaluate() own inside GroupKFold runs; onu
    holdout'a uygulamak gate'i HOLDOUT ILE egitmek olurdu and single atisi carcur ederdi.
    """
    lock =json .load (open ("results/split_lock.json"))
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}
    geom =json .load (open ("results/cc_e_partgeom.json"))
    A =M .load (ARM_OFF );B =M .load (ARM_ON )
    NGT =A [4 ]
    ok =lambda q :q in geom and "ext"in geom [q ]and q in set (A [2 ])and q in set (B [2 ])
    W ={q for q in work_parts if ok (q )}
    H ={q for q in hold_parts if ok (q )}
    hi_true_W ={q for q in W if NGT .get (q ,0 )>=M .HIGH_CP }
    ncand ={q :int ((A [2 ]==q ).sum ())for q in (W |H )}

    out ={"n_work":len (W ),"n_hold":len (H ),"seeds":[]}
    for s in seeds :
    # --- router: SADECE WORK'te egitilir, holdout'a uygulanir ---
        def rrows (ps ):
            r =[]
            for q in ps :
                g =geom [q ];e =g ["ext"]
                r .append ([e [0 ],e [1 ],e [2 ],g ["diag"],g ["area_box"],g ["vol_box"],ncand [q ],
                ncand [q ]/max (g ["area_box"],1e-6 )*1e3 ,ncand [q ]/max (g ["diag"],1e-6 )])
            return np .array (r ,float )
        wl =sorted (W );hl =sorted (H )
        rt =RandomForestClassifier (300 ,min_samples_leaf =2 ,n_jobs =-1 ,random_state =s )
        rt .fit (rrows (wl ),np .array ([1 if q in hi_true_W else 0 for q in wl ]))
        route ={hl [i ]for i ,p in enumerate (rt .predict_proba (rrows (hl ))[:,1 ])if p >=ROUTER_THR }

        # --- gate: SADECE WORK adaylariyla egitilir; threshold de WORK'te secilir ---
        sc ={}
        for tag ,D in (("off",A ),("ten",B )):
            X ,y ,pid ,_ ,_ =D 
            mw =np .array ([q in W for q in pid ]);mh =np .array ([q in H for q in pid ])
            clf =ExtraTreesClassifier (800 ,min_samples_leaf =2 ,n_jobs =-1 ,random_state =s )
            clf .fit (X [mw ],y [mw ])
            pw =clf .predict_proba (X [mw ])[:,1 ]
            gt_w =sum (NGT .get (q ,0 )for q in W )
            bt ,bf =M .THRS [0 ],-1.0 
            for t in M .THRS :# threshold WORK'te secilir, holdout GORULMEZ
                k =pw >=t 
                f =M .f1_of (int (y [mw ][k ].sum ()),int (k .sum ()),gt_w )
                if f >bf :bf ,bt =f ,t 
            sc [tag ]=(y [mh ],clf .predict_proba (X [mh ])[:,1 ],pid [mh ],bt )

        vals ={}
        for reg ,ps in (("low",H -{q for q in H if NGT .get (q ,0 )>=M .HIGH_CP }),
        ("high",{q for q in H if NGT .get (q ,0 )>=M .HIGH_CP })):
            T =N =Gt =0 
            for tag ,sel in (("off",lambda q :q not in route ),("ten",lambda q :q in route )):
                yh ,ph ,pidh ,thr =sc [tag ]
                m =np .array ([(q in ps )and sel (q )for q in pidh ])
                if m .sum ()==0 :continue 
                k =m &(ph >=thr )
                T +=int (yh [k ].sum ());N +=int (k .sum ())
                Gt +=sum (NGT .get (q ,0 )for q in set (pidh [m ]))
            vals [reg ]=M .f1_of (T ,N ,Gt )if Gt else float ("nan")
        lo =vals ["low"]if vals ["low"]==vals ["low"]else 0.0 
        hg =vals ["high"]if vals ["high"]==vals ["high"]else 0.0 
        vals ["corpus_weighted"]=(1 -M .W_HIGH )*lo +M .W_HIGH *hg 
        out ["seeds"].append (vals )
    for k in ("low","high","corpus_weighted"):
        v =[d [k ]for d in out ["seeds"]]
        out [k ]={"mean":float (np .nanmean (v )),"sd":float (np .nanstd (v ))}
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--holdout",action ="store_true",
    help ="KILITLI HOLDOUT single atis dogrulamasi -- GERI DONUSU YOK")
    a =ap .parse_args ()
    lock =json .load (open ("results/split_lock.json"))
    LOCK =set (lock ["locked_parts"])
    A =M .load (ARM_OFF )
    clean ={q for q in set (A [2 ])if A [3 ].get (q ,1 )==0 }

    t0 =time .time ()
    rec ={"config":CONFIG ,"generated":time .strftime ("%Y-%m-%d %H:%M:%S")}
    print ("=== URUN v2 VERIFICATION (WORK OOF) ===",flush =True )
    work =evaluate (clean -LOCK ,label ="WORK (kilitli haric, temiz)")
    rec ["work"]=work 
    print (f"  part {work ['n_parts']} (very-CP {work ['n_high_cp']})")
    for k in ("low","high","corpus_weighted"):
        print (f"  {k :<18}{work [k ]['mean']:.4f}  (sd {work [k ]['sd']:.4f})")

    if a .holdout :
        print ("\n=== KILITLI HOLDOUT -- TEK ATIS ===",flush =True )
        hp =clean &LOCK 
        if not hp :
            print ("  kilitli holdout parcasi bulunamadi (temiz kesisim empty)")
        else :
        # GERCEK holdout: gate/threshold/router SADECE WORK'te ogrenilir (see evaluate_holdout)
            ho =evaluate_holdout (clean -LOCK ,hp )
            ho ["label"]="KILITLI HOLDOUT (single atis, WORK'te egitildi)"
            rec ["holdout"]=ho 
            print (f"  holdout part {ho ['n_hold']} | gate/threshold/router {ho ['n_work']} WORK parcasinda egitildi")
            for k in ("low","high","corpus_weighted"):
                print (f"  {k :<18}{ho [k ]['mean']:.4f}  (sd {ho [k ]['sd']:.4f})")
            rec ["holdout_spent"]=time .strftime ("%Y-%m-%d %H:%M:%S")

    json .dump (rec ,open ("results/product_v2_receipt.json","w"),indent =1 ,ensure_ascii =False )
    print (f"\nmakbuz -> results/product_v2_receipt.json   ({time .time ()-t0 :.0f}s)")


if __name__ =="__main__":
    main ()
