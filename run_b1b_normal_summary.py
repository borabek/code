# -*- coding: utf-8 -*-
"""B1b: mesh normallerini OZETLE -- secenek sayisini dusur, sinyali koru.

B1 BASARISIZ OLDU (+0.0023) and sebebi measured: candidate basina 2mm icindeki HER
tepenin +/- normali AYRI secenekti; 58582 secenekte pozitif orani only
%6.4 -- selector samanlikta igne ariyordu. Sinyal VAR (results/b2_yon_isaret.json:
normaller YON_YOK'un %59.1'ini tasiyor), GURULTULU which is SECENEK KUMESIYDI.

DUZELTME: ham normaller instead of SUMMARY yonler --
  agirlikli : p_pos agirlikli mean normal
  pca       : normallerin birinci ana bileseni (sign-hizali)
  enyuksek  : most high p_pos'lu single tepenin normali
Her biri +/- with; mevcut yonle birlikte candidate basina most extra ~7 secenek.

TABAN ARTIK ISARET DUZELTMELI arm (B2a, 0.2773 -> 0.3090); B1b onun USTUNE gelir.
TEZE SADIK: konum and candidate count does not change, only direction.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import brep_pool # noqa: E402
import connector3d # noqa: E402
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import product_zinciri # noqa: E402
import wire_gate # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ={"d7":"results/_p1_olasilik_d7","tam":"results/_p1_olasilik_brepegit"}
KAYNAKLAR =(0 ,1 )
ESIK =0.05 
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
GIRME ,ERISIM =3 ,5 
NR =2.0 
AYIRT =8.0 
CE ,CT =int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )


def birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def yukle (on ,pid ):
    z =np .load (f"{OZ }/{on }_{pid }.npz")
    m =np .isin (np .asarray (z ["source"],int ),KAYNAKLAR )
    T =np .asarray (np .load (f"{TAN }/{on }_{pid }.npz")["T"],float )
    X =np .hstack ([np .asarray (z ["X"],float ),T ])[m ]
    if len (X )<2 :
        return None 
    f =f"{OB [on ]}/{pid }.npz"
    if not os .path .exists (f ):
        return None 
    zz =np .load (f )
    V =np .ascontiguousarray (zz ["V"],np .float64 )
    F =np .ascontiguousarray (zz ["F"],np .int64 )
    pb =np .asarray (zz ["pbs"],float ).mean (0 )
    return {"X":X ,"T":T [m ],"P":np .asarray (z ["P"],float )[m ],
    "D":np .asarray (z ["D"],float )[m ],"V":V ,"F":F ,"pb":pb ,
    "NV":brep_pool .vertex_normals_at (V ,F ),
    "ppos":pb [:,CE ]+pb [:,CT ]}


def pick_gate (gate ,d ):
    s =np .asarray (gate .predict_proba (
    wire_gate .within_part (d ["X"],"zskor"))[:,1 ],float )
    k =s >=ESIK 
    if not k .any ():
        return d ["P"][:0 ],d ["D"][:0 ],d ["T"][:0 ],s [:0 ]
    P ,D ,T ,sk =d ["P"][k ],d ["D"][k ],d ["T"][k ],s [k ]
    if len (P )>1 :
        nm =wire_gate .crowd_mask (P ,sk )
        P ,D ,T ,sk =P [nm ],D [nm ],T [nm ],sk [nm ]
        # ISARET DUZELTMESI (B2a, +0.0316) ARTIK TABANIN PARCASI
    D =np .where ((T [:,ERISIM ]<T [:,GIRME ])[:,None ],-D ,D )
    return P ,D ,T ,sk 


def ozet_yonler (p ,V ,NV ,ppos ):
    uz =np .linalg .norm (V -p ,axis =1 )
    k =uz <=NR 
    if not k .any ():
        return []
    N =NV [k ]
    w =np .maximum (ppos [k ],1e-6 )
    ref =N [int (np .argmax (w ))]
    Nh =N *np .sign (N @ref )[:,None ]
    out =[birim ([(Nh *w [:,None ]).sum (0 )])[0 ]]
    if len (Nh )>2 :
        Q =Nh -Nh .mean (0 )
        out .append (birim ([np .linalg .svd (Q ,full_matrices =False )[2 ][0 ]])[0 ])
    out .append (birim ([ref ])[0 ])
    return out 


def secenekler (p ,d ,V ,NV ,ppos ,gs ,n ):
    d =birim ([d ])[0 ]
    ad =[(d ,1.0 ,0.0 ,0.0 )]
    for v in ozet_yonler (p ,V ,NV ,ppos ):
        for s in (1.0 ,-1.0 ):
            ad .append ((s *v ,0.0 ,float (s ),1.0 ))
    sec ,oz =[],[]
    for v ,mev ,isr ,ozet in ad :
        v =birim ([v ])[0 ]
        if any (np .degrees (np .arccos (np .clip (abs (float (v @w )),-1 ,1 )))<AYIRT 
        for w ,_ ,_ ,_ in sec ):
            continue 
        sec .append ((v ,mev ,isr ,ozet ))
    for v ,mev ,isr ,ozet in sec :
        oz .append ([mev ,
        float (np .degrees (np .arccos (np .clip (float (v @d ),-1 ,1 )))),
        isr ,ozet ,float (gs ),float (n ),
        float (np .max (np .abs (v ))),float (len (sec ))])
    return [x [0 ]for x in sec ],np .asarray (oz ,float )


def main ():
    gate =pickle .load (open ("results/kazanan_hgb_derin.pkl","rb"))["HGB-derin"]
    S =K .step_map ()
    d6 ={str (p ):r for p ,r in 
    d6_record .yukle (set (d6_record .exam ()["pidler"])).items ()}
    Rk =K .yukle (None )

    def rec_ (p ):
        return Rk .get (p )or d6 .get (p )

    X ,Y =[],[]
    for pid in [f [4 :-4 ]for f in sorted (os .listdir (OZ ))
    if f .startswith ("tam_")]:
        r =rec_ (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        d =yukle ("tam",pid )
        if d is None :
            continue 
        P ,D ,T ,sk =pick_gate (gate ,d )
        if not len (P ):
            continue 
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        for a in range (len (P )):
            j =-1 
            for t in range (len (G )):
                n =np .linalg .norm (Gd [t ])
                if n <1e-9 :
                    continue 
                u =Gd [t ]/n 
                w =P [a ]-G [t ]
                e =float (w @u )
                if np .linalg .norm (w -e *u )<=YANAL and abs (e )<=EKSENEL :
                    j =t 
                    break 
            if j <0 :
                continue 
            V_ ,F_ =secenekler (P [a ],D [a ],d ["V"],d ["NV"],d ["ppos"],
            sk [a ],len (P ))
            if len (V_ )<2 :
                continue 
            u =Gd [j ]/max (np .linalg .norm (Gd [j ]),1e-12 )
            for q ,v in enumerate (V_ ):
                ac =float (np .degrees (np .arccos (np .clip (float (v @u ),-1 ,1 ))))
                X .append (F_ [q ])
                Y .append (int (ac <=ACI ))
    X =np .asarray (X ,float )
    Y =np .asarray (Y ,int )
    print (f"OZET secenek {X .shape } | pozitif {Y .mean ():.4f}   "
    f"(B1 ham: 58582 secenek, pozitif %6.4)",flush =True )
    yc =HistGradientBoostingClassifier (max_iter =400 ,learning_rate =0.08 ,
    max_leaf_nodes =31 ,
    random_state =0 ).fit (X ,Y )
    pickle .dump (yc ,open ("results/b1b_yon_secici.pkl","wb"))

    out ={}
    for ad ,kul in (("TABAN (sign duzeltmeli)",False ),("+OZET NORMAL",True )):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
        tes =[]
        for pid in [f [3 :-4 ]for f in sorted (os .listdir (OZ ))
        if f .startswith ("d7_")]:
            r =rec_ (pid )
            if r is None or not len (r .get ("G",[])):
                continue 
            d =yukle ("d7",pid )
            if d is None :
                continue 
            P ,D ,T ,sk =pick_gate (gate ,d )
            if kul and len (P ):
                D =D .copy ()
                for a in range (len (P )):
                    V_ ,F_ =secenekler (P [a ],D [a ],d ["V"],d ["NV"],
                    d ["ppos"],sk [a ],len (P ))
                    if len (V_ )<2 :
                        continue 
                    s2 =yc .predict_proba (F_ )[:,1 ]
                    b =int (np .argmax (s2 ))
                    if b !=0 and s2 [b ]>s2 [0 ]:
                        D [a ]=V_ [b ]
            if len (P ):
                P ,D =product_zinciri .tam_poz (d ["V"],d ["F"],d ["pb"],P ,D ,
                step_path =S .get (pid ))
            G =np .asarray (r ["G"],float )
            Gd =np .asarray (r ["Gd"],float )
            dg =float (r ["diag"])
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,YANAL ,ACI ,False ,
            signed =True )[:3 ]
            a_ =rob [r ["mfg"]]
            a_ [0 ]+=tp 
            a_ [1 ]+=fp 
            a_ [2 ]+=fn 
            tes .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,dg ,
            max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ])
        pm ={m :2 *v [0 ]/max (2 *v [0 ]+v [1 ]+v [2 ],1 )
        for m ,v in rob .items ()}
        mi =float (2 *sum (v [0 ]for v in rob .values ())/
        max (sum (2 *v [0 ]+v [1 ]+v [2 ]for v in rob .values ()),1 ))
        out [ad ]={"robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"brand":pm }
        print (f"{ad :<26} robot {mi :.4f} | tespit {out [ad ]['tespit']:.4f} | "
        f"makro {out [ad ]['makro']:.4f} | en kotu {out [ad ]['en_kotu']:.4f}",
        flush =True )
    a =out ["TABAN (sign duzeltmeli)"]["robot"]
    b =out ["+OZET NORMAL"]["robot"]
    print (f"\nFARK {b -a :+.4f} | KAPI >= +0.02")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,"fark":b -a ,
    "not":"Mesh normalleri OZETLENDI (agirlikli/pca/en-yuksek). "
    "Taban ISARET DUZELTMELI arm. D7 brand-disi, TAM ZINCIR."},
    open ("results/b1b_ozet_normal.json","w"),indent =1 )
    print ("receipt -> results/b1b_ozet_normal.json")


if __name__ =="__main__":
    main ()
