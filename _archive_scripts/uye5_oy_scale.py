# -*- coding: utf-8 -*-
"""5. uye: loss GERCEK mi, otherwise `votes` OLCEK KAYMASI mi?

MEASURED (results/uye5_{val,dev}.json): 5. bagimsiz uye two kumede de RECALL'i yukseltiyor
(VAL 0.518->0.556, DEV 0.613->0.644) but KESINLIGI dusuruyor (0.679->0.642, 0.828->0.779).
Net detection: VAL +0.0208, DEV -0.0038.

HIPOTEZ: gate `votes` ozelligini 4 UYEYLE uretilmis veriden ogrendi; orada votes in {1,2,3,4}.
5 uyede same candidate {1..5} degerini takes -- i.e. gate'in "3 oy" diye ogrendigi confidence seviyesi
residual different a anlama geliyor. Bu a MODEL kusuru not, EGITIM/CALISMA uyusmazligi
(gate_refit dersi: gate bayatlayinca +0.1273 kaybediliyordu).

UCUZ PROB: `votes` sutununu 4/5 with olcekleyip gate'e oyle ver. Egitimdeki olcege geri returns.
  - precision geri gelir and recall kalirsa -> loss UYUSMAZLIKTAN; gate'i 5 uyeli havuzda yeniden
    egitmek MESRU and degerli (pahali but olculmus rationale with).
  - precision geri gelmezse -> 5. uye gercekten noise ekliyor; is kapanir.

KILL: olcekli 5-uye, 4-uyeyi tespitte gecmezse full gate yeniden egitimi ACILMAZ.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
VOTES =11 # wire_gate.FEAT_NAMES_13.index("votes")


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from j_position_mean import vote_avg 
    from sina_cluster import esle ,f1w ,pr 

    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "val").lower ()
    cf =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (cf )and cluster =="dev":
        cf ="results/_h_probs.pkl"
    cache =pickle .load (open (cf ,"rb"))
    for r in cache :
        r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
    u5 =pickle .load (open (f"results/_probs_{cluster }_u5.pkl","rb"))
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"low":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}

    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"absent:"+r ["pid"])for r in cache }
    pids =np .array ([str (x )for x in d ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in pids ]),list (tg ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
    print (f"KUME={cluster } | {len (cache )} part | gate {int (keep .sum ())} candidate "
    f"(4 uyeli veriyle egitildi, votes in 1..4)",flush =True )

    # --- turetme: 4 uye and 5 uye (each biri BIR times) ---
    DER ={}
    for n_uye in (4 ,5 ):
        rows =[]
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            if n_uye ==5 :
                plist =plist +[np .asarray (u5 [r ["pid"]],np .float64 )]
            mk =lambda pr_ ,**kw :cp_openings .connection_points (
            V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"],**kw )
            merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
            base =merge ([mk (pb )for pb in plist ])
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
            Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT )if cps else None 
            rows .append (dict (X =Xc ,is_hi =is_hi ,
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
            Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
            G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
        DER [n_uye ]=rows 
        print (f"  turetme {n_uye } uye bitti ({sum (len (x ['P'])for x in rows )} candidate)",flush =True )

    def kos (n_uye ,olcek ):
        det ,rob =[],[]
        for r in DER [n_uye ]:
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                Xc =r ["X"]
                if olcek !=1.0 :
                    Xc =Xc .copy ();Xc [:,VOTES ]=Xc [:,VOTES ]*olcek 
                sc =clf .predict_proba (Xc )[:,1 ]
                m =sc >=(THR ["very"]if r ["is_hi"]else THR ["low"])
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            k ="very"if r ["n"]>=8 else "low"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'yapilandirma':<28}{'detection':>9}{'ROBOT':>9}{'conclusive':>9}{'recall':>9}")
    R ={}
    for lab ,n_ ,s_ in (("4 uye (urun)",4 ,1.0 ),("5 uye, olceksiz",5 ,1.0 ),
    ("5 uye, votes x4/5",5 ,0.8 )):
        det ,rob =kos (n_ ,s_ )
        R [lab ]=(det ,rob )
        p_ ,r_ =pr (det )
        print (f"{lab :<28}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )

    rng =np .random .RandomState (0 )
    npart =len (R ["4 uye (urun)"][0 ])
    IX =[rng .randint (0 ,npart ,npart )for _ in range (2000 )]
    print ("\nESLI BOOTSTRAP (4 uyeye according to):")
    out ={}
    for lab in ("5 uye, olceksiz","5 uye, votes x4/5"):
        for mi ,mn in ((0 ,"detection"),(1 ,"robot")):
            a ,b =R [lab ][mi ],R ["4 uye (urun)"][mi ]
            ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])for ix in IX ])
            lo_ ,hi_ =np .percentile (ds ,[2.5 ,97.5 ])
            print (f"  {lab +' ('+mn +')':<32}{ds .mean ():>+9.4f}  [{lo_ :+.4f}, {hi_ :+.4f}]  "
            f"{'BELIRGIN'if lo_ >0 or hi_ <0 else 'noise'}",flush =True )
            out [f"{lab }|{mn }"]=[float (ds .mean ()),float (lo_ ),float (hi_ )]

    dd =f1w (R ["5 uye, votes x4/5"][0 ])-f1w (R ["4 uye (urun)"][0 ])
    print (f"\nKILL: olcekli 5-uye 4-uyeyi tespitte gecmezse tam gate yeniden egitimi ACILMAZ -> "
    f"{dd :+.4f} => {'AC'if dd >0 else 'ACMA'}")
    json .dump ({"cluster":cluster ,
    "detection":{k :float (f1w (v [0 ]))for k ,v in R .items ()},
    "robot":{k :float (f1w (v [1 ]))for k ,v in R .items ()},
    "precision":{k :float (pr (v [0 ])[0 ])for k ,v in R .items ()},
    "recall":{k :float (pr (v [0 ])[1 ])for k ,v in R .items ()},
    "bootstrap":out },open (f"results/uye5_olcek_{cluster }.json","w"),indent =1 )
    print (f"receipt -> results/uye5_olcek_{cluster }.json")


if __name__ =="__main__":
    main ()
