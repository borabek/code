# -*- coding: utf-8 -*-
"""Gate v6 UCTAN UCA: dagitilan gate BAYAT VERIYLE mi egitilmis?

DURUM: dagitilan gate `gate_regrow_data_rt2.npz` (30 Tem 03:04) with egitildi. Bu gecenin parite
duzeltmeleri (3/3 cagriya step_path -> low-CP yolunda da B-rep ekseni, benzersiz oy sayimi,
agirlikli konum) turetmeyi DEGISTIRDI. Yani gate'in gordugu ozelliklerle calisma aninda aldigi
ozellikler same hattan gelmiyor may be -- full as `gate_refit` dersindeki state (+0.1273).

v6 verisi (31 Tem 10:38) guncel hatla uretildi. GATE DUZEYINDE -0.0185 olctum and "olu" dedim;
but gate duzeyi two times yaniltti and dahasi two gate FARKLI candidate havuzlarinda karsilastirilmisti
(rt2 havuzu vs v6 havuzu) -- this, `gate_refit` dersinin acikca yasakladigi kiyas.

BU BETIK correct kiyasi yapar: two gate de AYNI test adaylarinda (guncel hat) uctan uca olculur.
Egitim taraflari different (dogal, karsilastirilan sey this), test tarafi AYNI.

KILL: v6 rt2'yi tespit F1'de gecmezse dagitilan gate does not change.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


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
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "cok":float (cfg ["robot_wire_gate_threshold_highcp"])}

    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in cache }

    CLF ={}
    for tag in ("rt2","v6"):
        d =np .load (f"results/gate_regrow_data_{tag }.npz",allow_pickle =True )
        pids =np .array ([str (x )for x in d ["pids"]])
        keep =~np .isin (np .array ([gk .get (p ,"yok:"+p )for p in pids ]),list (tg ))
        CLF [tag ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
        print (f"gate {tag }: {int (keep .sum ())}/{len (pids )} candidate ile egitildi",flush =True )

        # --- turetme BIR KEZ (guncel urun hatti) ---
    DER =[]
    for r in cache :
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        mk =lambda pr_ ,**kw :cp_openings .connection_points (
        V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"],**kw )
        merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
        base =merge ([mk (pb )for pb in plist ])
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
        Xc =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT )if cps else None 
        DER .append (dict (X =Xc ,is_hi =is_hi ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"{len (DER )} part | {sum (len (r ['P'])for r in DER )} candidate (AYNI pool, iki gate icin de)",
    flush =True )

    def kos (tag ):
        clf =CLF [tag ];det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                sc =clf .predict_proba (r ["X"])[:,1 ]
                m =sc >=(THR ["cok"]if r ["is_hi"]else THR ["dusuk"])
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            k ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'gate training verisi':<24}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R ={}
    for tag in ("rt2","v6"):
        det ,rob =kos (tag )
        R [tag ]=(det ,rob )
        p_ ,r_ =pr (det )
        lab =f"{tag } ({'BAYAT/dagitilan'if tag =='rt2'else 'GUNCEL hat'})"
        print (f"{lab :<24}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )

    rng =np .random .RandomState (0 )
    npart =len (R ["rt2"][0 ])
    IX =[rng .randint (0 ,npart ,npart )for _ in range (2000 )]
    print ("\nESLI BOOTSTRAP (v6 - rt2):")
    out ={}
    for mi ,mn in ((0 ,"tespit"),(1 ,"robot")):
        a ,b =R ["v6"][mi ],R ["rt2"][mi ]
        ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])for ix in IX ])
        lo_ ,hi_ =np .percentile (ds ,[2.5 ,97.5 ])
        print (f"  {mn :<8}{ds .mean ():>+9.4f}  [{lo_ :+.4f}, {hi_ :+.4f}]  "
        f"{'BELIRGIN'if lo_ >0 or hi_ <0 else 'noise'}",flush =True )
        out [mn ]=[float (ds .mean ()),float (lo_ ),float (hi_ )]

    d_ =f1w (R ["v6"][0 ])-f1w (R ["rt2"][0 ])
    print (f"\nKILL: v6 rt2'yi tespitte gecmezse gate DEGISMEZ -> fark {d_ :+.4f} => "
    f"{'DEGISTIR'if d_ >0 else 'DEGISTIRME'}")
    json .dump ({"cluster":cluster ,"rt2_tespit":f1w (R ["rt2"][0 ]),"v6_tespit":f1w (R ["v6"][0 ]),
    "rt2_robot":f1w (R ["rt2"][1 ]),"v6_robot":f1w (R ["v6"][1 ]),"bootstrap":out },
    open (f"results/gate_v6_uctan_uca_{cluster }.json","w"),indent =1 )
    print (f"receipt -> results/gate_v6_uctan_uca_{cluster }.json")


if __name__ =="__main__":
    main ()
