# -*- coding: utf-8 -*-
"""T7: durdurma kuralinin +0.0150'si GERCEK mi, GURULTU mu?

Dalin gidisati: -0.0148 (part-regresyonu) -> +0.0089 (only-azalt) -> +0.0142 (dur/devam)
-> +0.0150 (F1-agirlikli). Kahinin %15'i, bar +0.02'nin under.

Karar vermeden before two sey lazim:
  1. ESLESTIRILMIS BOOTSTRAP: difference sifiri iceriyor mu? Iceriyorsa arm OLU, "bar under"
     demeye bile gerek absent.
  2. TOHUM DAYANIKLILIGI: single a random_state with alinan +0.0150, seed degisince duruyor mu?
     (this projede more before "+0.053" diye a kazanc seed/split degisince erimisti)

Ikisi de gecerse arm GERCEK but KUCUK demektir -> istiflenebilir candidate as kayda gecer.
Gecmezse dal KAPANIR.
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
ESIK =0.60 


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
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }))
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
        PAR .append ({"geo":gk .get (r ["pid"],"yok:"+r ["pid"]),
        "rj":"cok"if r ["n"]>=8 else "dusuk",
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
            p ["diag"],float (p ["hi"])]+p ["X"][k ].tolist ())
            Y .append (1 if (k +1 )<=p ["kahin_K"]else 0 )
            W .append (abs (float (p ["f1s"][k +1 ]-p ["f1s"][k ])))
            GRP .append (p ["geo"])
    R =np .array (R ,float );Y =np .array (Y );W =np .array (W );GRP =np .array (GRP )
    W =np .clip (W /max (W .mean (),1e-9 ),0.05 ,None )

    def kur (seed ):
        oof =np .zeros (len (Y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (R ,Y ,GRP ):
            clf =RandomForestClassifier (n_estimators =500 ,min_samples_leaf =5 ,n_jobs =-1 ,
            random_state =seed )
            clf .fit (R [tr ],Y [tr ],sample_weight =W [tr ])
            oof [te ]=clf .predict_proba (R [te ])[:,1 ]
        return oof 

    def satirlar (Kf ):
        out =[]
        for p in PAR :
            K =int (max (0 ,min (len (p ["s"]),Kf (p ))))
            out .append ((p ["rj"],)+match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"]))
        return out 

    baseline =satirlar (lambda p :p ["su_K"])
    print (f"TABAN {f1w (baseline ):.4f}\n")
    print (f"{'seed':<8}{'tespit':>9}{'fark':>9}")
    farklar ,kollar =[],[]
    for seed in (0 ,1 ,2 ,3 ,4 ):
        oof =kur (seed )
        i0 =0 
        for p in PAR :
            n =len (p ["s"]);p ["pr"]=oof [i0 :i0 +n ];i0 +=n 

        def dur (p ):
            pr =p ["pr"];k =0 
            while k <len (pr )and pr [k ]>=ESIK :
                k +=1 
            return k 
        yeni =satirlar (dur )
        kollar .append (yeni )
        farklar .append (f1w (yeni )-f1w (baseline ))
        print (f"{seed :<8}{f1w (yeni ):>9.4f}{farklar [-1 ]:>+9.4f}",flush =True )

    f =np .array (farklar )
    print (f"\nTOHUM DAYANIKLILIGI: ort {f .mean ():+.4f} | sd {f .std ():.4f} | "
    f"min {f .min ():+.4f} | maks {f .max ():+.4f} | {int ((f >0 ).sum ())}/5 pozitif")

    rng =np .random .default_rng (0 )
    yeni0 =kollar [0 ]
    v =[]
    for _ in range (4000 ):
        i =rng .integers (0 ,len (baseline ),len (baseline ))
        v .append (f1w ([yeni0 [k ]for k in i ])-f1w ([baseline [k ]for k in i ]))
    v =np .array (v )
    lo ,hi =np .percentile (v ,2.5 ),np .percentile (v ,97.5 )
    gercek =lo >0 
    print (f"ESLESTIRILMIS BOOTSTRAP (seed 0): {v .mean ():+.4f} [{lo :+.4f}, {hi :+.4f}] -> "
    f"{'GERCEK'if gercek else 'GURULTU (GA sifiri iceriyor)'}")

    print (f"\nKARAR:")
    print (f"  bar (+0.02)          : {'GECTI'if f .mean ()>=0.02 else 'GECMEDI'}")
    print (f"  gercek mi (GA)       : {'EVET'if gercek else 'HAYIR'}")
    print (f"  seed dayanikli mi   : {'EVET'if (f >0 ).all ()else 'HAYIR'}")
    sonuc =("ISTIFLENEBILIR ADAY (gercek ama bar alti)"if (gercek and (f >0 ).all ()
    and f .mean ()<0.02 )else 
    "DAGITILABILIR"if f .mean ()>=0.02 and gercek else "OLU")
    print (f"  -> {sonuc }")
    with open ("results/t7_durdurma_dogrula.json","w",encoding ="utf-8")as fh :
        json .dump ({"baseline":float (f1w (baseline )),"farklar":[float (x )for x in f ],
        "ort":float (f .mean ()),"sd":float (f .std ()),
        "bootstrap":[float (v .mean ()),float (lo ),float (hi )],
        "gercek":bool (gercek ),"sonuc":sonuc },fh ,indent =1 )
    print ("receipt -> results/t7_durdurma_dogrula.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
