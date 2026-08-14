# -*- coding: utf-8 -*-
"""T5: kesimi DURDURMA KARARI as kur (regresyonun coktugu yeri duzelt).

T3/T4 kahin K'yi PARCA BASINA single a number as kestirmeye calisti and kaybetti:
    dogrudan regresyon  -0.0148 | buzulme +0.0001 | selector -0.0005 | only-azalt +0.0089

Iki yapisal kusuru vardi:
  1. EGITIM SATIRI SAYISI 200 (part basina a hedef). Ogrenilecek sey for very few.
  2. TARGET GURULTULU: same F1'i veren bircok K present; kare-error most aza indiren prediction,
     F1'i most aza indiren prediction DEGIL.

DOGRU KURULUM: kesim, part basina single number not, ADAY BASINA a DUR/DEVAM kararidir.
Adaylar skora according to sirali; k. candidate for label "kahin K >= k mi" (i.e. this adayi almali miyiz).
Boylece training satiri 200 -> 3277 becomes and each satirin etiketi TEMIZ.

CIKARIM: siradaki candidates for olasilik is computed, first P < threshold which is places DURULUR (onek
kuralina saygi). Esik taranir.

OZELLIKLER (all of them calisma aninda, GT'siz):
  candidate: skor, skor/part-maks, order, order/n, onceki skorla difference, sonraki skorla difference,
        that ana kadarki kumulatif skor
  part: n_aday, maks, mean, std, kosegen, very-CP bayragi
  (+ istege bagli: adayin own 22 gate ozelligi)

KILL (t3/t4 with AYNI, degistirilmedi): tespit >= +0.02 VE DEV with VAL same yonde VE
gorulmemis manufacturer ortalamasi dusmeyecek.
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
        "cluster":kume_of .get (r ["pid"],"dev"),"mfg":mfg_of .get (r ["pid"],"?"),
        "rj":"cok"if r ["n"]>=8 else "dusuk",
        "s":s [o ],"X":np .asarray (r ["X"],float )[o ],
        "P":r ["P"][o ],"Pd":r ["Pd"][o ],
        "G":np .asarray (r ["G"],float ),"Gd":np .asarray (r ["Gd"],float ),
        "diag":float (r ["diag"]),"hi":r ["n"]>=8 })
    for p in PAR :
        en ,enk =-1 ,0 
        for K in range (0 ,len (p ["s"])+1 ):
            tp ,fp ,fn =match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"])
            f =2 *tp /max (2 *tp +fp +fn ,1 )
            if f >en :
                en ,enk =f ,K 
        p ["kahin_K"]=enk 
        p ["su_K"]=int (wire_gate .decision_mask (p ["s"]).sum ())

        # ADAY BASINA satirlar: k. adayi ALMALI MIYIZ?
    R ,Y ,GRP =[],[],[]
    for p in PAR :
        v =p ["s"];n =len (v )
        kum =0.0 
        for k in range (n ):
            kum +=float (v [k ])
            R .append ([float (v [k ]),float (v [k ]/max (v [0 ],1e-9 )),float (k +1 ),
            float ((k +1 )/n ),
            float (v [k -1 ]-v [k ])if k >0 else 0.0 ,
            float (v [k ]-v [k +1 ])if k +1 <n else 0.0 ,
            float (kum ),float (n ),float (v [0 ]),float (v .mean ()),float (v .std ()),
            p ["diag"],float (p ["hi"])]+p ["X"][k ].tolist ())
            Y .append (1 if (k +1 )<=p ["kahin_K"]else 0 )
            GRP .append (p ["geo"])
    R =np .array (R ,float );Y =np .array (Y );GRP =np .array (GRP )
    print (f"{len (PAR )} part | {len (Y )} ADAY satiri (regresyon 200 satirla calisiyordu) | "
    f"pozitif {Y .mean ():.1%}",flush =True )

    oof =np .zeros (len (Y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (R ,Y ,GRP ):
        oof [te ]=RandomForestClassifier (n_estimators =500 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (R [tr ],Y [tr ]).predict_proba (R [te ])[:,1 ]

        # part basina olasilik dizisine geri dagit
    i0 =0 
    for p in PAR :
        n =len (p ["s"])
        p ["pr"]=oof [i0 :i0 +n ]
        i0 +=n 

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
    kah =puanla (lambda p :p ["kahin_K"])
    print (f"\nTABAN {a :.4f} (DEV {a_dev :.4f} / VAL {a_val :.4f} / manufacturer ort {a_mfg :.4f}) | "
    f"KAHIN {kah :.4f} (+{kah -a :.4f})")

    def dur (p ,t ):
        """ILK P < t which is places DUR (onek kurali)."""
        pr =p ["pr"]
        k =0 
        while k <len (pr )and pr [k ]>=t :
            k +=1 
        return k 

    def say (p ,t ):
        """P >= t olanlarin SAYISI (onek zorunlulugu absent)."""
        return int ((p ["pr"]>=t ).sum ())

    print (f"\n{'arm':<26}{'tespit':>9}{'fark':>9}{'DEV':>9}{'VAL':>9}{'mfg ort':>10}{'KILL':>8}")
    KOL ={}
    for t in (0.3 ,0.4 ,0.5 ,0.6 ,0.7 ):
        KOL [f"A dur-ilk P<{t :.1f}"]=lambda p ,t =t :dur (p ,t )
        KOL [f"B say P>={t :.1f}"]=lambda p ,t =t :say (p ,t )
    kazanan ,en =None ,-10.0 
    SON ={}
    for ad ,Kf in KOL .items ():
        v =puanla (Kf )
        dv =puanla (Kf ,lambda p :p ["cluster"]=="dev")-a_dev 
        vl =puanla (Kf ,lambda p :p ["cluster"]=="val")-a_val 
        mf =np .mean ([puanla (Kf ,lambda p ,x =x :p ["mfg"]==x )for x in ("PXC","WEI")])-a_mfg 
        ok =(v -a >=0.02 )and dv >0 and vl >0 and mf >=0 
        print (f"{ad :<26}{v :>9.4f}{v -a :>+9.4f}{dv :>+9.4f}{vl :>+9.4f}{mf :>+10.4f}"
        f"{'GECTI'if ok else '':>8}")
        SON [ad ]={"tespit":float (v ),"fark":float (v -a ),"dev":float (dv ),
        "val":float (vl ),"mfg":float (mf ),"gecti":bool (ok )}
        if ok and v >en :
            kazanan ,en =ad ,v 
    print (f"\nSONUC: {kazanan if kazanan else 'HICBIRI GECMEDI'}")
    if not kazanan :
        iyi =max (SON ,key =lambda k :SON [k ]["fark"])
        print (f"  en iyi arm {iyi }: {SON [iyi ]['fark']:+.4f} "
        f"(kahinin {SON [iyi ]['fark']/max (kah -a ,1e-9 ):.0%}'i)")
    with open ("results/t5_durdurma.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":float (a ),"kahin":float (kah ),"kollar":SON ,
        "kazanan":kazanan ,"n_satir":int (len (Y ))},f ,indent =1 )
    print ("receipt -> results/t5_durdurma.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
