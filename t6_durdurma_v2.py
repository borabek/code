# -*- coding: utf-8 -*-
"""T6: durdurma kararini IKI SEYLE iyilestir (t5 +0.0142'de kaldi).

T5 formulasyon degisikligiyle -0.0148'den +0.0142'ye output (part basina regresyon instead of
ADAY BASINA dur/devam karari; training satiri 200 -> 3277). Kahinin %14'u.

Iki missing vardi, ikisi de ucuz:

  1. MODEL SIFIRDAN OGRENIYOR. Mevcut kuralin karari feature as VERILMIYORDU; model
     "kesim nerede" sorusunu bastan cozmeye calisiyordu. Oysa istedigimiz sey mevcut kuralin
     DUZELTMESI. Mevcut kararin bayragi and mevcut K'ya according to goreli order eklendi -> model
     residual "here kuraldan sapmali miyim" sorusunu ogreniyor.

  2. HER SATIR ESIT AGIRLIKTAYDI. Oysa a adayin alinip alinmamasi parcanin F1'ini sometimes
     never degistirmiyor, sometimes very. Siniflandirici, never difference etmeyen satirlari correct bilmek
     for difference eden satirlari feda edebiliyordu. Satirlar residual MARJINAL F1 ETKISIYLE
     agirliklandiriliyor -- ogrenilen sey urun metrigine baglandi.

KILL (t3/t4/t5 with AYNI, degistirilmedi): tespit >= +0.02 VE DEV with VAL same yonde VE
gorulmemis manufacturer ortalamasi dusmeyecek. Gecerse eslestirilmis bootstrap with dogrulanir.
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


def match3 (P ,Pd ,G ,Gd ,diag ):
    hit =np .zeros (len (G ),bool );used =set ()
    if len (P )and len (G ):
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *diag )
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >tt or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ )
    tp =int (hit .sum ())
    return tp ,len (P )-tp ,len (G )-tp 


def main ():
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    with open ("results/_dev_val_kume.json",encoding ="utf-8")as f :
        kume_of =json .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"none:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"none:"+r ["pid"])for r in DER }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf0 =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf0 (Xtr ),"clf_z":rf0 (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))

    PAR =[]
    for r in DER :
        if r ["X"]is None or not len (r ["G"]):
            continue 
        s =np .asarray (wire_gate .decision_score (m ,r ["X"]),float )
        o =np .argsort (-s )
        PAR .append ({"geo":gk .get (r ["pid"],"none:"+r ["pid"]),
        "cluster":kume_of .get (r ["pid"],"dev"),"mfg":mfg_of .get (r ["pid"],"?"),
        "rj":"very"if r ["n"]>=8 else "dusuk",
        "s":s [o ],"X":np .asarray (r ["X"],float )[o ],
        "P":r ["P"][o ],"Pd":r ["Pd"][o ],
        "G":np .asarray (r ["G"],float ),"Gd":np .asarray (r ["Gd"],float ),
        "diag":float (r ["diag"]),"hi":r ["n"]>=8 })
    for p in PAR :
        f1s =[]
        for K in range (0 ,len (p ["s"])+1 ):
            tp ,fp ,fn =match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"])
            f1s .append (2 *tp /max (2 *tp +fp +fn ,1 ))
        p ["f1s"]=np .array (f1s )
        p ["kahin_K"]=int (np .argmax (p ["f1s"]))
        p ["su_K"]=int (wire_gate .decision_mask (p ["s"]).sum ())

    R ,Y ,W ,GRP =[],[],[],[]
    for p in PAR :
        v =p ["s"];n =len (v );kum =0.0 
        for k in range (n ):
            kum +=float (v [k ])
            R .append ([float (v [k ]),float (v [k ]/max (v [0 ],1e-9 )),float (k +1 ),
            float ((k +1 )/n ),
            float (v [k -1 ]-v [k ])if k >0 else 0.0 ,
            float (v [k ]-v [k +1 ])if k +1 <n else 0.0 ,
            float (kum ),float (n ),float (v [0 ]),float (v .mean ()),float (v .std ()),
            p ["diag"],float (p ["hi"]),
            # IYILESTIRME 1: mevcut kuralin karari + ona according to goreli konum
            float ((k +1 )<=p ["su_K"]),float ((k +1 )-p ["su_K"]),
            float (p ["su_K"]),float (p ["su_K"]/n )]+p ["X"][k ].tolist ())
            Y .append (1 if (k +1 )<=p ["kahin_K"]else 0 )
            # IYILESTIRME 2: this adayi almanin MARJINAL F1 etkisi = ogrenmenin agirligi
            W .append (abs (float (p ["f1s"][k +1 ]-p ["f1s"][k ])))
            GRP .append (p ["geo"])
    R =np .array (R ,float );Y =np .array (Y );W =np .array (W );GRP =np .array (GRP )
    W =W /max (W .mean (),1e-9 )
    W =np .clip (W ,0.05 ,None )# sifir agirlikli row ogrenmeyi tamamen kaybetmesin
    print (f"{len (PAR )} part | {len (Y )} satir | pozitif {Y .mean ():.1%} | "
    f"agirlik ort {W .mean ():.2f} (medyan {np .median (W ):.2f})",flush =True )

    KOLLAR ={"v1 (t5 tekrari)":dict (w =False ,ek =False ),
    "v2a +mevcut rule":dict (w =False ,ek =True ),
    "v2b +agirlik":dict (w =True ,ek =False ),
    "v2c ikisi":dict (w =True ,ek =True )}
    N_EK =4 
    SON ={}
    for ad ,kfg in KOLLAR .items ():
        M =R if kfg ["ek"]else np .hstack ([R [:,:13 ],R [:,13 +N_EK :]])
        oof =np .zeros (len (Y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,Y ,GRP ):
            clf =RandomForestClassifier (n_estimators =500 ,min_samples_leaf =5 ,n_jobs =-1 ,
            random_state =0 )
            clf .fit (M [tr ],Y [tr ],sample_weight =W [tr ]if kfg ["w"]else None )
            oof [te ]=clf .predict_proba (M [te ])[:,1 ]
        i0 =0 
        for p in PAR :
            n =len (p ["s"]);p ["pr"]=oof [i0 :i0 +n ];i0 +=n 

        def puanla (Kf ,sec =None ):
            rows =[]
            for p in PAR :
                if sec is not None and not sec (p ):
                    continue 
                K =int (max (0 ,min (len (p ["s"]),Kf (p ))))
                rows .append ((p ["rj"],)+match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"]))
            return f1w (rows )if rows else float ("nan")

        a =puanla (lambda p :p ["su_K"])
        a_dev =puanla (lambda p :p ["su_K"],lambda p :p ["cluster"]=="dev")
        a_val =puanla (lambda p :p ["su_K"],lambda p :p ["cluster"]=="val")
        a_mfg =np .mean ([puanla (lambda p :p ["su_K"],lambda p ,x =x :p ["mfg"]==x )
        for x in ("PXC","WEI")])

        def dur (p ,t ):
            pr =p ["pr"];k =0 
            while k <len (pr )and pr [k ]>=t :
                k +=1 
            return k 
        en =None 
        for t in (0.4 ,0.5 ,0.55 ,0.6 ,0.65 ,0.7 ):
            Kf =lambda p ,t =t :dur (p ,t )
            v =puanla (Kf )
            dv =puanla (Kf ,lambda p :p ["cluster"]=="dev")-a_dev 
            vl =puanla (Kf ,lambda p :p ["cluster"]=="val")-a_val 
            mf =np .mean ([puanla (Kf ,lambda p ,x =x :p ["mfg"]==x )
            for x in ("PXC","WEI")])-a_mfg 
            ok =(v -a >=0.02 )and dv >0 and vl >0 and mf >=0 
            if en is None or v >en [1 ]:
                en =(t ,v ,v -a ,dv ,vl ,mf ,ok )
        SON [ad ]=en 
        print (f"{ad :<20} en iyi t={en [0 ]:.2f} | tespit {en [1 ]:.4f} ({en [2 ]:+.4f}) | "
        f"DEV {en [3 ]:+.4f} | VAL {en [4 ]:+.4f} | mfg {en [5 ]:+.4f} | "
        f"{'GECTI'if en [6 ]else ''}",flush =True )

    iyi =max (SON ,key =lambda k :SON [k ][2 ])
    print (f"\nEN IYI KOL: {iyi } -> {SON [iyi ][2 ]:+.4f}")
    print (f"KILL: tespit >= +0.02 VE DEV/VAL ayni yonde VE manufacturer ort dusmeyecek -> "
    f"{'GECTI'if SON [iyi ][6 ]else 'GECMEDI'}")
    with open ("results/t6_durdurma_v2.json","w",encoding ="utf-8")as f :
        json .dump ({k :{"threshold":v [0 ],"tespit":v [1 ],"fark":v [2 ],"dev":v [3 ],
        "val":v [4 ],"mfg":v [5 ],"gecti":bool (v [6 ])}
        for k ,v in SON .items ()},f ,indent =1 )
    print ("receipt -> results/t6_durdurma_v2.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
