# -*- coding: utf-8 -*-
"""L1 degerlendirmesi: 5. BAGIMSIZ uye toplulugu iyilestiriyor mu?

RATIONALE: gate'in 13 ozelliginden only `votes` durust bolmede transfer ediyor (AUC dususu
0.018). `votes` = kac bagimsiz uyenin same acikligi gordugu -- i.e. single genelleyen sinyalin
COZUNURLUGU uye sayisiyla sinirli. Turetilmis 5. uye (olasilik ortalamasi) MEASURED and KAYBETTI
because bagimsiz degildi; this run gercekten ayri tohumlu a checkpoint kullanir.

KILL (onceden yazildi): VAL kumesinde tespit F1 +0.01 gelmezse uye ALINMAZ.

Not: gate `votes` ozelligini kullaniyor and training verisi 4 uyeyle uretildi. 5 uyede votes'un
olcegi degisir, that yuzden gate 5-uyeli adaylarla YENIDEN egitilmeden karsilastirma adil olmaz.
Bu yuzden here gate'i each two kolda da AYNI candidate havuzundan yeniden egitiyoruz (gate_refit
dersi: two gate however AYNI havuzda karsilastirilir).
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
NEW ="results/seg_extra/recall_hard_keig96_s3.pt"


def ek_cikarim (cluster ):
    """Onbellekteki each part for YALNIZ new uyenin olasiligini uret -> _probs_<cluster>_u5.pkl"""
    import torch 
    import diffusionnet as D 
    from infer_step_cp import load_any 

    src =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (src )and cluster =="dev":
        src ="results/_h_probs.pkl"
    out =f"results/_probs_{cluster }_u5.pkl"
    cache =pickle .load (open (src ,"rb"))
    done =pickle .load (open (out ,"rb"))if os .path .exists (out )else {}
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta =load_any (NEW ,dev =dev )[:2 ]
    k =int (meta .get ("k_eig",64 ))
    for i ,r in enumerate (cache ):
        pid =r .get ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
        if pid in done :
            continue 
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
        op_cache_dir =f"results/step_infer/ops_k{k }",return_probs =True )
        done [pid ]=np .asarray (pb ,np .float16 )
        if (i +1 )%25 ==0 :
            pickle .dump (done ,open (out ,"wb"))
            print (f"  {i +1 }/{len (cache )}",flush =True )
    pickle .dump (done ,open (out ,"wb"))
    print (f"-> {out }: {len (done )} part",flush =True )
    return cache ,done 


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from j_position_mean import vote_avg 
    from sina_cluster import esle ,f1w ,pr 

    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "val").lower ()
    cache ,u5 =ek_cikarim (cluster )
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"low":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}

    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"]
    pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r .get ("pid",os .path .basename (r ["stp"]).split ("_")[1 ]),
    "absent")for r in cache }
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in pids ]),list (tg ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X [keep ],y [keep ])
    print (f"KUME={cluster } | {len (cache )} part | gate {int (keep .sum ())} candidate",flush =True )

    def kos (n_uye ):
        det ,rob =[],[]
        for r in cache :
            pid =r .get ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            if n_uye ==5 :
                plist =plist +[np .asarray (u5 [pid ],np .float64 )]
            mk =lambda pr_ ,**kw :cp_openings .connection_points (
            V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"],**kw )
            merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
            base =merge ([mk (pb )for pb in plist ])
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if cps :
                Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT )
                sc =clf .predict_proba (Xc )[:,1 ]
                m =sc >=(THR ["very"]if is_hi else THR ["low"])
                if m .any ():
                    P =np .array ([c ["point"]for c ,k_ in zip (cps ,m )if k_ ],float )
                    Pd =np .array ([c ["direction"]for c ,k_ in zip (cps ,m )if k_ ],float )
            k ="very"if r ["n"]>=8 else "low"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'ensemble':<20}{'tespit':>9}{'ROBOT':>9}{'conclusive':>9}{'recall':>9}")
    R ={}
    for n in (4 ,5 ):
        det ,rob =kos (n )
        R [n ]=(det ,rob )
        p_ ,r_ =pr (det )
        print (f"{str (n )+' uye':<20}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )

    rng =np .random .RandomState (0 )
    npart =len (R [4 ][0 ])
    IX =[rng .randint (0 ,npart ,npart )for _ in range (2000 )]
    print (f"\nESLI BOOTSTRAP (5 uye - 4 uye):")
    ok ={}
    for mi ,mn in ((0 ,"tespit"),(1 ,"robot")):
        a ,b =R [5 ][mi ],R [4 ][mi ]
        ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])for ix in IX ])
        lo_ ,hi_ =np .percentile (ds ,[2.5 ,97.5 ])
        print (f"  {mn :<8}{ds .mean ():>+9.4f}  [{lo_ :+.4f}, {hi_ :+.4f}]  "
        f"{'BELIRGIN'if lo_ >0 or hi_ <0 else 'noise'}",flush =True )
        ok [mn ]=[float (ds .mean ()),float (lo_ ),float (hi_ )]

    d_ =f1w (R [5 ][0 ])-f1w (R [4 ][0 ])
    print (f"\nKILL (onceden yazili): tespit +0.01 gelmezse uye ALINMAZ -> "
    f"fark {d_ :+.4f} => {'ALINIR'if d_ >=0.01 else 'ALINMAZ'}")
    json .dump ({"cluster":cluster ,"4uye_tespit":f1w (R [4 ][0 ]),"5uye_tespit":f1w (R [5 ][0 ]),
    "4uye_robot":f1w (R [4 ][1 ]),"5uye_robot":f1w (R [5 ][1 ]),
    "bootstrap":ok ,"karar":"ALINIR"if d_ >=0.01 else "ALINMAZ"},
    open (f"results/uye5_{cluster }.json","w"),indent =1 )
    print (f"receipt -> results/uye5_{cluster }.json")


if __name__ =="__main__":
    main ()
