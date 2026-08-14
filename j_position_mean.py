# -*- coding: utf-8 -*-
"""J: UNION temsilcisinin konumunu ANLASAN UYELERIN ORTALAMASI yap.

WHY: `robot_cp._vote2` anlasan modelleri gruplayip EN YUKSEK GUVENLI uyeyi temsilci aliyor
and digerlerinin konumunu ATIYOR. Toplulugun most klasik faydasi -- bagimsiz hatalarin ortalamada
sonmesi -- so never kullanilmiyor.

WHY SIMDI: durust (geometri) bolmede ayrisim degisti:
    detection 0.6250 -> lateral<=2mm 0.4715 -> robot-hazir 0.4159
i.e. KONUM kaybi -0.154, EKSEN kaybi -0.056. Eksen I with large olcude closed; asil kalem residual
konum. B (silindir eksenine izdusurme) olculup became but that DIS a referansa tasima denemesiydi;
this different: own uyelerimizin ortalamasi, dis referans absent.

KILL: lateral<=2mm ya da detection gerilerse duser.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"very":0.105 }


def vote_avg (cp_lists ,cluster_mm =5.0 ,min_votes =1 ,mode ="rep",dir_mode ="rep"):
    """mode='rep'   -> mevcut davranis (most safe uye temsilci)
       mode='mean'  -> anlasan uyelerin KONUM ortalamasi
       mode='wmean' -> confidence agirlikli konum ortalamasi"""
    allc =[c for lst in cp_lists for c in lst ]
    allc .sort (key =lambda c :-float (c .get ("confidence",0.0 )))
    kept =[]
    for c in allc :
        p =np .asarray (c ["point"],float )
        hit =None 
        for k in kept :
            if np .linalg .norm (p -np .asarray (k ["point"],float ))<=cluster_mm :
                hit =k 
                break 
        if hit is None :
            c =dict (c )
            c ["_votes"]=1 
            c ["_pts"]=[p ]
            c ["_ws"]=[float (c .get ("confidence",1.0 ))]
            c ["_ds"]=[np .asarray (c ["direction"],float )]
            kept .append (c )
        else :
            hit ["_votes"]+=1 
            hit ["_pts"].append (p )
            hit ["_ws"].append (float (c .get ("confidence",1.0 )))
            hit ["_ds"].append (np .asarray (c ["direction"],float ))
    out =[c for c in kept if c ["_votes"]>=min_votes ]
    if mode !="rep":
        for c in out :
            P =np .array (c ["_pts"],float )
            if mode =="mean":
                c ["point"]=P .mean (0 )
            else :
                w =np .array (c ["_ws"],float )
                w =w /max (w .sum (),1e-9 )
                c ["point"]=(P *w [:,None ]).sum (0 )
    if dir_mode !="rep":
        for c in out :
            D =np .array (c ["_ds"],float )
            D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )
            if len (D )>1 :
            # ISARETSIZ mean: uyeler same ekseni ZIT isaretle verebilir; ham mean
            # onlari birbirini goturur. Once first uyeye according to sign hizalanir.
                D =D *np .sign (D @D [0 ])[:,None ]
                if dir_mode =="wmean":
                    w =np .array (c ["_ws"],float );w =w /max (w .sum (),1e-9 )
                    v =(D *w [:,None ]).sum (0 )
                else :
                    v =D .mean (0 )
                nv =np .linalg .norm (v )
                if nv >1e-9 :
                    v =v /nv 
                    if float (np .dot (v ,np .asarray (c ["direction"],float )))<0 :
                        v =-v # urunun ISARET konvansiyonunu koru (disari)
                    c ["direction"]=v 
    for c in out :
        c .pop ("_pts",None )
        c .pop ("_ws",None )
        c .pop ("_ds",None )
    return out 


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from big_arbiter import eligible 

    cfg =json .load (open ("cp_config.json"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    cache =pickle .load (open ("results/_h_probs.pkl","rb"))
    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"];groups =d ["groups"]
    pids =np .array ([str (x )for x in d ["pids"]])
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    # KESKIN geometri anahtari: kaba bbox anahtari IKIZ OLMAYAN parcalari ikiz saniyordu
    # (775 grup vs keskin 1220). Bu, sizintiyi and therefore kaybi ABARTIYORDU.
    _kf ="results/_strict_geometry_keys.json"
    gk =json .load (open (_kf ))if os .path .exists (_kf )else json .load (open ("results/_geometry_keys.json"))
    _SET =os .environ .get ("OLCUM_KUME","")# "" = old 100, ya da dev/val/locked

    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :
            continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (202 )
    lo =[x for x in parts if x [4 ]<8 ]
    hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    if _SET :
        want =set (json .load (open ("results/split3.json"))[_SET ]["parts"])
        sel =[p for p in parts if p [1 ]in want ]
        print (f"KUME={_SET }: {len (sel )} part",flush =True )
    tg ={gk .get (p [1 ],"none:"+p [1 ])for p in sel }
    Gg =np .array ([gk .get (p ,"none:"+p )for p in pids ])
    keep =~np .isin (Gg ,list (tg ))
    reg_all =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "dusuk")for g in groups ])
    Xk ,yk ,gg ,reg =X [keep ],y [keep ],Gg [keep ],reg_all [keep ]
    tot ={"dusuk":0 ,"very":0 }
    for g in {int (g )for g in groups [keep ]}:
        n =int (ngt .get (g ,0 ))
        if n >0 :
            tot ["very"if n >=8 else "dusuk"]+=n 
    o =np .zeros (len (yk ))
    for tr ,te in GroupKFold (n_splits =5 ).split (Xk ,yk ,gg ):
        o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Xk [tr ],yk [tr ]).predict_proba (Xk [te ])[:,1 ]
        # PARITE: esikler DAGITILAN urunun okudugu yerden gelir. Kendi secmek, measured_path urunu
        # dagitilandan different a isletim noktasinda calistirir (2026-07-31 denetimi: measurement
        # 0.35/0.20 secerken urun 0.40/0.35 kosuyordu).
    thr ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Xk ,yk )
    print (f"GEOMETRI bolmesi | gate {int (keep .sum ())} candidate | esikler {thr }",flush =True )
    print (f"{len (cache )} test part",flush =True )

    KEYS =(("det",0.0 ,180.0 ,True ),("lat",2.0 ,180.0 ,False ),("rob",2.0 ,10.0 ,False ))

    def run (mode ):
        res ={k :{"dusuk":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}for k ,_ ,_ ,_ in KEYS }
        LAT =[]
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            if mode [2 ]>=5 :
                plist =plist +[sum (plist )/len (plist )]# 5. uye = ORTALAMA harita
            der =[cp_openings .connection_points (
            V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =pb ,vertex_conf =VC ,
            ct_depth_min_mm =1.0 ,cluster_mm =CL ,step_path =r ["stp"])for pb in plist ]
            base =robot_cp ._vote2 (der ,min_votes =1 )# PARITE: urunun own fonksiyonu
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            if is_hi :
                der2 =[cp_openings .connection_points (
                V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),
                dedupe_mm =10.0 ,probs =pb ,vertex_conf =VC ,
                ct_depth_min_mm =1.0 ,cluster_mm =CL ,conn_promote =0.25 ,
                step_path =r ["stp"])for pb in plist ]
                cps =robot_cp ._vote2 (der2 ,min_votes =1 )
            else :
                cps =base 
            kept =[]
            if cps :
                probs =sum (np .asarray (p_ ,np .float64 )for p_ in r ["pbs"])/len (r ["pbs"])
                sc =clf .predict_proba (wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ))[:,1 ]
                t_ =thr ["very"]if is_hi else thr ["dusuk"]
                kept =[c for c ,s_ in zip (cps ,sc )if s_ >=t_ ]
            P =np .array ([c ["point"]for c in kept ],float )if kept else np .zeros ((0 ,3 ))
            Pd =np .array ([c ["direction"]for c in kept ],float )if kept else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"]
            for key ,tol ,am ,pct in KEYS :
                hit =np .zeros (len (G ),bool )
                used =set ()
                if len (P )and len (G ):
                    diff =P [:,None ,:]-G [None ,:,:]
                    al =(diff *Gd [None ,:,:]).sum (-1 )
                    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                    an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
                    tt =max (3.0 ,0.06 *r ["diag"])if pct else tol 
                    pe2 =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
                    for d_ ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )
                    for a in range (len (P ))for b in range (len (G ))):
                        if d_ >tt or a_ in used or hit [b_ ]:
                            continue 
                        hit [b_ ]=True 
                        used .add (a_ )
                        if key =="det":
                            LAT .append (d_ )
                tp =int (hit .sum ())
                k ="very"if r ["n"]>=8 else "dusuk"
                res [key ][k ][0 ]+=tp 
                res [key ][k ][1 ]+=len (P )-tp 
                res [key ][k ][2 ]+=len (G )-tp 
        out ={}
        for key in res :
            o_ ={}
            for k ,(T ,Fp ,Fn )in res [key ].items ():
                p_ =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
                o_ [k ]=2 *p_ *rc /max (p_ +rc ,1e-9 )
            out [key ]=sum (W [k ]*o_ [k ]for k in W )
        return out ,(np .array (LAT )if LAT else np .array ([9.0 ]))

    print ()
    print (f"{'arm':<26}{'detection':>9}{'yanal2':>9}{'ROBOT':>9}{'lateral med':>11}{'<=2mm':>8}")
    out ={}
    for mode ,lab in ((("wmean","rep",4 ),"URUN (parite + benzersiz oy)"),):
        o_ ,L =run (mode )
        out [lab ]=dict (o_ ,yanal_medyan =float (np .median (L )),oran_2mm =float ((L <=2 ).mean ()))
        print (f"{lab :<26}{o_ ['det']:>9.4f}{o_ ['lat']:>9.4f}{o_ ['rob']:>9.4f}"
        f"{np .median (L ):>11.2f}{100 *(L <=2 ).mean ():>7.0f}%",flush =True )
    json .dump (out ,open ("results/j_position_mean.json","w"),indent =1 )
    print ("\nKILL: lateral<=2mm ya da detection gerilerse duser.")
    print ("receipt -> results/j_position_mean.json")


if __name__ =="__main__":
    main ()
