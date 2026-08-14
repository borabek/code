# -*- coding: utf-8 -*-
"""S1 uctan uca: toplulukta keig96_s0 -> keig128_s0 DEGISIMI kazandiriyor mu?

RATIONALE: k_eig 64 -> 96 bizim single plato kiran hamlemizdi (F1 0.383 -> 0.477, 3/3 seed) and
orada durduk. Tez 128'i arama uzayina koymus (Tablo 3) but §1945'te ">64 asiri ogrenmeye path
acmistir" deyip 64'te kalmis. Yani this arm tezin YONTEMINE sadik, SONUCUNA not; mesruiyeti
bizim own 96 olcumumuzden geliyor.

TASARIM -- single variable:
  * Toplulugun 4 uyesinden only INDEKS 1 degisiyor (keig96_s0 -> keig128_s0). Digerleri same.
  * Ayni seed (0), same label dizinleri, same secim metrigi -> single difference k_eig.
  * Aday turetme and gate AYNI; gate test geometri gruplarini gormuyor.
  * Verimlilik: diger 3 uyenin olasiliklari onbellekte hazir, only new uye for inference.

KILL (onceden yazili): VAL'de uctan uca TESPIT F1 +0.01 gelmezse uye ALINMAZ.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
YENI ="results/seg_extra/recall_hard_keig128_s0.pt"
IDX =1 # cp_config.current_product.checkpoints inside keig96_s0'in yeri


def yeni_uye_probs (cluster ,cache ):
    """Yalniz YENI uye for probability uret -> _probs_<cluster>_k128.pkl (part kimligine according to devam)."""
    import torch 
    import diffusionnet as D 
    from infer_step_cp import load_any 
    out =f"results/_probs_{cluster }_k128.pkl"
    done =pickle .load (open (out ,"rb"))if os .path .exists (out )else {}
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta =load_any (YENI ,dev =dev )[:2 ]
    k =int (meta .get ("k_eig",64 ))
    assert k ==128 ,f"beklenen k_eig 128, gelen {k }"
    for i ,r in enumerate (cache ):
        if r ["pid"]in done :
            continue 
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
        op_cache_dir =f"results/step_infer/ops_k{k }",return_probs =True )
        done [r ["pid"]]=np .asarray (pb ,np .float16 )
        if (i +1 )%25 ==0 :
            pickle .dump (done ,open (out ,"wb"))
            print (f"  inference {i +1 }/{len (cache )}",flush =True )
    pickle .dump (done ,open (out ,"wb"))
    return done 


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import esle ,f1w ,pr 

    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "val").lower ()
    cf =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (cf )and cluster =="dev":
        cf ="results/_h_probs.pkl"
    cache =pickle .load (open (cf ,"rb"))
    for r in cache :
        r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
    k128 =yeni_uye_probs (cluster ,cache )

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"low":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}

    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"absent:"+r ["pid"])for r in cache }
    pids =np .array ([str (x )for x in d ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in pids ]),list (tg ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ][:,:18 ],d ["y"][keep ])
    print (f"KUME={cluster } | {len (cache )} part | gate {int (keep .sum ())} candidate x 18 sutun",flush =True )

    def kos (swap ):
        det ,rob =[],[]
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            if swap :
                plist [IDX ]=np .asarray (k128 [r ["pid"]],np .float64 )
            mk =lambda pr_ ,**kw :cp_openings .connection_points (
            V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"],**kw )
            merge =lambda L :robot_cp ._vote2 (L ,min_votes =1 )
            base =merge ([mk (pb )for pb in plist ])
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if cps :
                Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
                step_path =r ["stp"])
                m =clf .predict_proba (Xc [:,:18 ])[:,1 ]>=(THR ["very"]if is_hi else THR ["low"])
                if m .any ():
                    P =np .array ([c ["point"]for c ,k_ in zip (cps ,m )if k_ ],float )
                    Pd =np .array ([c ["direction"]for c ,k_ in zip (cps ,m )if k_ ],float )
            kk ="very"if r ["n"]>=8 else "low"
            det .append ((kk ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((kk ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'ensemble':<28}{'detection':>9}{'ROBOT':>9}{'conclusive':>9}{'recall':>9}")
    R ={}
    for sw ,lab in ((False ,"mevcut (keig96_s0)"),(True ,"keig128_s0 with degisik")):
        det ,rob =kos (sw )
        R [lab ]=(det ,rob )
        p_ ,r_ =pr (det )
        print (f"{lab :<28}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )

    pickle .dump (R ,open (f"results/s1_parca_{cluster }.pkl","wb"))
    a_ ,b_ ="keig128_s0 with degisik","mevcut (keig96_s0)"
    rng =np .random .RandomState (0 )
    n =len (R [b_ ][0 ]);IX =[rng .randint (0 ,n ,n )for _ in range (2000 )]
    print ("\nESLI BOOTSTRAP (128 - 96):")
    out ={}
    for mi ,mn in ((0 ,"detection"),(1 ,"robot")):
        a ,b =R [a_ ][mi ],R [b_ ][mi ]
        ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])for ix in IX ])
        lo ,hi =np .percentile (ds ,[2.5 ,97.5 ])
        print (f"  {mn :<8}{ds .mean ():>+9.4f}  [{lo :+.4f}, {hi :+.4f}]  "
        f"{'BELIRGIN'if lo >0 or hi <0 else 'noise'}",flush =True )
        out [mn ]=[float (ds .mean ()),float (lo ),float (hi )]
    dd =f1w (R [a_ ][0 ])-f1w (R [b_ ][0 ])
    print (f"\nKILL: detection +0.01 gelmezse ALINMAZ -> {dd :+.4f} => {'ALINIR'if dd >=0.01 else 'ALINMAZ'}")
    json .dump ({"cluster":cluster ,"detection":{k :float (f1w (v [0 ]))for k ,v in R .items ()},
    "robot":{k :float (f1w (v [1 ]))for k ,v in R .items ()},"bootstrap":out },
    open (f"results/s1_keig128_{cluster }.json","w"),indent =1 )
    print (f"receipt -> results/s1_keig128_{cluster }.json")


if __name__ =="__main__":
    main ()
