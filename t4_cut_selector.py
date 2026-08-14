# -*- coding: utf-8 -*-
"""T4: part kesimini SECICI degistir (t3'un dogrudan regresyonu dustu).

T3 oracle K'yi dogrudan kestirdi and KAYBETTI (-0.0148; DEV/VAL/manufacturer all of them negatif). Sebep
hedefin gurultulu olmasi: same F1'i veren bircok K present, regresyon ortalamaya kaciyor and
mevcut kuraldan HER PARCADA sapiyor.

Bugun ISE YARAYAN desen farkliydi: kurali herkese uygulamak instead of YALNIZ gerektigi places
uygulamak (cokus yonlendirmesi, yarik secicisi). Burada da same:

  A  mevcut rule (baseline)
  B  BUZULME: K = round(alfa*K_tahmin + (1-alfa)*K_mevcut)   -- alfa taranir
  C  SECICI : only |K_tahmin - K_mevcut| >= threshold ISE degistir, otherwise mevcut rule
  D  TEK YON: only AZALTMA yonunde degistir (oracle K medyani mevcuttan KUCUK: 3 vs 4)

KILL (t3 with AYNI, degistirilmedi): detection >= +0.02 VE DEV with VAL same yonde VE unseen
manufacturer ortalamasi dusmeyecek.
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


def ozellik (s ,diag ,is_hi ):
    v =np .sort (np .asarray (s ,float ))[::-1 ]
    n =len (v )
    pad =lambda i :float (v [i ])if i <n else 0.0 
    dif =np .diff (v )*-1.0 if n >1 else np .array ([0.0 ])
    return [float (n ),float (v .max ()),float (v .mean ()),float (v .std ()),float (np .median (v )),
    pad (0 )-pad (1 ),pad (0 )-pad (2 ),pad (1 )-pad (2 ),
    float ((v >=0.25 ).sum ()),float ((v >=0.35 ).sum ()),
    float ((v >=0.50 ).sum ()),float ((v >=0.70 ).sum ()),
    float (dif .max ()),float (int (np .argmax (dif ))+1 )if n >1 else 1.0 ,
    float (diag ),float (bool (is_hi ))]


def main ():
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 
    from sklearn .model_selection import GroupKFold 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    with open ("results/_dev_val_cluster.json",encoding ="utf-8")as f :
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
        "s":s [o ],"P":r ["P"][o ],"Pd":r ["Pd"][o ],
        "G":np .asarray (r ["G"],float ),"Gd":np .asarray (r ["Gd"],float ),
        "diag":float (r ["diag"]),"hi":r ["n"]>=8 })
    for p in PAR :
        en ,enk =-1 ,1 
        for K in range (0 ,len (p ["s"])+1 ):
            tp ,fp ,fn =match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"])
            f =2 *tp /max (2 *tp +fp +fn ,1 )
            if f >en :
                en ,enk =f ,K 
        p ["kahin_K"]=enk 
        p ["X"]=ozellik (p ["s"],p ["diag"],p ["hi"])
    X =np .array ([p ["X"]for p in PAR ],float )
    yk =np .array ([p ["kahin_K"]for p in PAR ],float )
    grp =np .array ([p ["geo"]for p in PAR ])
    suK =np .array ([int (wire_gate .decision_mask (p ["s"]).sum ())for p in PAR ],float )
    print (f"{len (PAR )} part | oracle K medyan {np .median (yk ):.0f} | mevcut K medyan "
    f"{np .median (suK ):.0f}",flush =True )

    oof =np .zeros (len (PAR ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,yk ,grp ):
        oof [te ]=RandomForestRegressor (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],yk [tr ]).predict (X [te ])

    def puanla (Ks ,sec =None ):
        rows =[]
        for p ,K in zip (PAR ,Ks ):
            if sec is not None and not sec (p ):
                continue 
            K =int (max (0 ,min (len (p ["s"]),round (K ))))
            rows .append ((p ["rj"],)+match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"]))
        return f1w (rows )if rows else float ("nan")

    a =puanla (suK )
    a_dev =puanla (suK ,lambda p :p ["cluster"]=="dev")
    a_val =puanla (suK ,lambda p :p ["cluster"]=="val")
    a_mfg =np .mean ([puanla (suK ,lambda p ,x =x :p ["mfg"]==x )for x in ("PXC","WEI")])
    print (f"\nTABAN: detection {a :.4f} | DEV {a_dev :.4f} | VAL {a_val :.4f} | manufacturer ort {a_mfg :.4f}")
    print (f"KAHIN: {puanla (yk ):.4f} (+{puanla (yk )-a :.4f})")

    KOL ={}
    for al in (0.15 ,0.3 ,0.5 ,0.75 ,1.0 ):
        KOL [f"B buzulme alfa={al :.2f}"]=al *oof +(1 -al )*suK 
    for es in (1.5 ,2.5 ,4.0 ):
        K =suK .copy ()
        f =np .abs (oof -suK )>=es 
        K [f ]=oof [f ]
        KOL [f"C selector |diff|>={es }"]=K 
    for es in (0.5 ,1.5 ,2.5 ):
        K =suK .copy ()
        f =(suK -oof )>=es # only AZALTMA
        K [f ]=oof [f ]
        KOL [f"D only azalt >={es }"]=K 

    print (f"\n{'arm':<24}{'detection':>9}{'diff':>9}{'DEV':>9}{'VAL':>9}{'mfg ort':>10}{'KILL':>9}")
    kazanan ,en =None ,-10 
    for ad ,K in KOL .items ():
        t =puanla (K )
        dv =puanla (K ,lambda p :p ["cluster"]=="dev")-a_dev 
        vl =puanla (K ,lambda p :p ["cluster"]=="val")-a_val 
        mf =np .mean ([puanla (K ,lambda p ,x =x :p ["mfg"]==x )for x in ("PXC","WEI")])-a_mfg 
        ok =(t -a >=0.02 )and dv >0 and vl >0 and mf >=0 
        print (f"{ad :<24}{t :>9.4f}{t -a :>+9.4f}{dv :>+9.4f}{vl :>+9.4f}{mf :>+10.4f}"
        f"{'GECTI'if ok else '':>9}")
        if ok and t >en :
            kazanan ,en =ad ,t 
    print (f"\nSONUC: {kazanan if kazanan else 'HICBIRI GECMEDI -> mevcut rule KALIR'}")
    with open ("results/t4_cut_selector.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":float (a ),"oracle":float (puanla (yk )),
        "kollar":{k :float (puanla (v ))for k ,v in KOL .items ()},
        "kazanan":kazanan },f ,indent =1 )
    print ("receipt -> results/t4_cut_selector.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
