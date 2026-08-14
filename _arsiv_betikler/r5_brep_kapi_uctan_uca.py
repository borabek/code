# -*- coding: utf-8 -*-
"""R5: BREP_MAX_OFF kapisi UCTAN UCA (kodda ASILI kalan "P2 taramasi").

cp_openings.py'deki not aynen soyle diyor:
    "BU IKI SAYI CARPIK OLCUMLERE GORE AYARLANMISTI: radius yay-centroid'inden kestirilirken
     ~3.5 fold KUCUK cikiyordu, i.e. r_range upper siniri 12mm gercekte ~42mm demekti (fiilen
     filtresiz) and axis NOKTASI eksenden ~r up to kaymisti, max_off_mm=5.0 that kaymayi tolere
     ediyordu. Cember oturtma duzeltmesinden after ikisi de yeniden ayarlanmali -> P2 taramasi."

Yaricap hatasi 2026-07-31'de duzeltildi; TARAMA YAPILMADI. Dagitilan value still 5.0.

R4'te (candidate duzeyi, izdusum deneyi) 5.0'in 2.0/3.0'a according to COK kotu oldugu gorundu. Kapi gevsekse
CP KOMSU a silindire kilitleniyor and onun ekseni geliyor -- this, angle hatasindaki ~90 derecelik
populasyonun muhtemel kaynagi.

DURUST WARNING: gate degisince ADAYLAR (konum+direction) degisir, therefore gate'in ozellikleri de
degisir; dagitilan gate 5.0 with turetilmis adaylarla egitildi. Yani this measurement kapinin TEK BASINA
etkisini not, "gate sabitken gate degisirse ne becomes"u gives. Kazanc belirginse full corpus
yeniden uretilip gate yeniden egitilir (P9 dersi).

KILL (onceden yazili): a gate degeri dagitilabilmesi for
  robot-hazir >= +0.02 VE tespit >= -0.01 (ikisi de DEV and VAL'de same yonde).
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
KAPILAR =[5.0 ,3.0 ,2.0 ]


def main ():
    import importlib 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    with open ("results/_dev_val_kume.json",encoding ="utf-8")as f :
        kume_of =json .load (f )

    cache =[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        with open (cf ,"rb")as f :
            for r in pickle .load (f ):
                r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
                cache .append (r )
    print (f"{len (cache )} part",flush =True )

    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"yok:"+r ["pid"])for r in cache }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    M ={"clf":rf (Xtr ),"clf_z":rf (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (M ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    M ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))
    print (f"gate hazir (sizintisiz) | cokus esigi {M ['esik_cokus']:.4f}",flush =True )

    SONUC ={}
    for gate in KAPILAR :
        os .environ ["CP_BREP_MAX_OFF"]=str (gate )
        import cp_openings ,robot_cp 
        importlib .reload (cp_openings )
        assert abs (cp_openings .BREP_MAX_OFF -gate )<1e-9 ,cp_openings .BREP_MAX_OFF 
        from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
        det ={"dev":[],"val":[]};rob ={"dev":[],"val":[]}
        aci_hep =[]
        for i_ ,r in enumerate (cache ,1 ):
            if i_ %50 ==0 :
                print (f"  gate {gate }: {i_ }/{len (cache )}",flush =True )
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
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
                X =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
                step_path =r ["stp"])
                s =wire_gate .decision_score (M ,X )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =np .array ([c ["point"]for c in cps ],float )[k ]
                    Pd =np .array ([c ["direction"]for c in cps ],float )[k ]
            km =kume_of .get (r ["pid"],"dev")
            rj ="cok"if r ["n"]>=8 else "dusuk"
            det [km ].append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob [km ].append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            if len (P )and len (r ["Gd"]):
                a =np .degrees (np .arccos (np .clip (np .abs (Pd @np .asarray (r ["Gd"]).T ),0 ,1 )))
                aci_hep .extend (a .min (1 ).tolist ())
        hep_d =det ["dev"]+det ["val"];hep_r =rob ["dev"]+rob ["val"]
        ah =np .array (aci_hep )
        SONUC [gate ]={"tespit":float (f1w (hep_d )),"robot":float (f1w (hep_r )),
        "tespit_dev":float (f1w (det ["dev"])),"tespit_val":float (f1w (det ["val"])),
        "robot_dev":float (f1w (rob ["dev"])),"robot_val":float (f1w (rob ["val"])),
        "aci_45_ustu":float ((ah >45 ).mean ()),"aci_10_alti":float ((ah <=10 ).mean ())}
        s_ =SONUC [gate ]
        print (f"\nKAPI {gate }: tespit {s_ ['tespit']:.4f} (dev {s_ ['tespit_dev']:.4f} / "
        f"val {s_ ['tespit_val']:.4f}) | ROBOT {s_ ['robot']:.4f} "
        f"(dev {s_ ['robot_dev']:.4f} / val {s_ ['robot_val']:.4f}) | "
        f"aci<=10 {s_ ['aci_10_alti']:.1%} | aci>45 {s_ ['aci_45_ustu']:.1%}\n",flush =True )

    t =SONUC [5.0 ]
    print (f"\n{'gate':<8}{'tespit':>9}{'robot':>9}{'d_tespit':>10}{'d_robot':>9}"
    f"{'aci<=10':>9}{'KILL':>10}")
    kazanan =None 
    for gate in KAPILAR :
        s_ =SONUC [gate ]
        dt =s_ ["tespit"]-t ["tespit"];dr =s_ ["robot"]-t ["robot"]
        ok =(dr >=0.02 and dt >=-0.01 and gate !=5.0 
        and (s_ ["robot_dev"]-t ["robot_dev"])>0 and (s_ ["robot_val"]-t ["robot_val"])>0 )
        print (f"{gate :<8.1f}{s_ ['tespit']:>9.4f}{s_ ['robot']:>9.4f}{dt :>+10.4f}{dr :>+9.4f}"
        f"{s_ ['aci_10_alti']:>9.1%}{'GECTI'if ok else ('baseline'if gate ==5.0 else 'gecmedi'):>10}")
        if ok and (kazanan is None or s_ ["robot"]>SONUC [kazanan ]["robot"]):
            kazanan =gate 
    print (f"\nSONUC: {('KAPI '+str (kazanan )+' DAGITILABILIR')if kazanan else 'KAPI 5.0 KALIR'}")
    with open ("results/r5_brep_kapi.json","w",encoding ="utf-8")as f :
        json .dump ({str (k ):v for k ,v in SONUC .items ()}|{"kazanan":kazanan },f ,indent =1 )
    print ("receipt -> results/r5_brep_kapi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
