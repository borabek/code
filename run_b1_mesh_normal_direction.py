# -*- coding: utf-8 -*-
"""B1: adayin YONUNU yakin MESH NORMALLERINDEN sec (konum DEGISMEZ).

DAYANAK (`results/b2_yon_isaret.json`, D7 brand-disi): YON_YOK kovasindaki 705
GT'nin **%59.1'inde (417)** adaya 2mm mesafedeki a mesh tepesinin normali
correct yonu ZATEN tasiyor. Ayrica %44.3'u duz 180 derece ISARET hatasi.

Kova aritmetigi (`results/kazanan_hata_bankasi.json`): TP 710 / FN 2377 / FP 1323.
417 kurtarilirsa F1 0.2773 -> **0.407**. Kol candidate EKLEMEZ -> FP artamaz,
tespit metrigi (angle serbest) DEGISMEZ; this, olcumun own kontrolu becomes.

ONCEKI YON KOLU WHY BASARISIZDI: `yon_odunc.py` secenekleri komsu ADAY
yonleri + silindir eksenleri + baskin yondu; **mesh normali YOKTU**. B2 olcumu
missing kaynagin full da that oldugunu showed.

TEZE SADIK: `v_o` KONUMU and segmentasyon does not change; tezin own yonu HER ZAMAN
0. secenek and esitlikte KAZANIR.
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
OB ={"d7":"results/_p1_olasilik_d7","d6":"results/_p1_olasilik_g7",
"tam":"results/_p1_olasilik_brepegit"}
KAYNAKLAR =(0 ,1 )
ESIK =0.05 
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
NORMAL_R =2.0 # adaya this mesafedeki tepelerin normalleri secenek becomes
AYIRT =5.0 # this aciya yakin secenekler AYNI sayilir
CE ,CT =int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )
_D6 =None 


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def secenekler (p ,d ,V ,NV ,ppos ,gate_s ,n_aday ):
    """Bu candidate for (direction, feature) listesi. 0. oge HER ZAMAN MEVCUT direction."""
    d =_birim ([d ])[0 ]
    V =np .asarray (V ,float )
    uz =np .linalg .norm (V -p ,axis =1 )
    k =uz <=NORMAL_R 
    candidates =[(d ,0.0 ,float (gate_s ),1.0 ,0.0 )]# mevcut
    if k .any ():
        for i in np .where (k )[0 ]:
            n =NV [i ]
            for s in (1.0 ,-1.0 ):
                candidates .append ((s *n ,float (uz [i ]),float (ppos [i ]),0.0 ,
                float (s )))
    secili ,oz =[],[]
    for v ,mes ,pp ,mevcut ,sign in candidates :
        v =_birim ([v ])[0 ]
        if any (float (np .degrees (np .arccos (np .clip (abs (float (v @w )),-1 ,1 ))))
        <AYIRT for w ,_ ,_ ,_ ,_ in secili ):
            continue 
        secili .append ((v ,mes ,pp ,mevcut ,sign ))
    for v ,mes ,pp ,mevcut ,sign in secili :
        aci_mevcut =float (np .degrees (np .arccos (np .clip (float (v @d ),-1 ,1 ))))
        oz .append ([mevcut ,aci_mevcut ,mes ,pp ,sign ,float (gate_s ),
        float (n_aday ),float (np .max (np .abs (v ))),
        float (len (secili ))])
    return [x [0 ]for x in secili ],np .asarray (oz ,float )


def parca_yukle (on ,pid ,r ):
    z =np .load (f"{OZ }/{on }_{pid }.npz")
    m =np .isin (np .asarray (z ["kaynak"],int ),KAYNAKLAR )
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
    return {"X":X ,"P":np .asarray (z ["P"],float )[m ],
    "D":np .asarray (z ["D"],float )[m ],"V":V ,"F":F ,"pb":pb ,
    "NV":brep_pool .vertex_normals_at (V ,F ),
    "ppos":pb [:,CE ]+pb [:,CT ]}


def pick_gate (model ,d ):
    s =np .asarray (model .predict_proba (
    wire_gate .within_part (d ["X"],"zskor"))[:,1 ],float )
    k =s >=ESIK 
    P ,D ,sk =((d ["P"][k ],d ["D"][k ],s [k ])if k .any ()
    else (d ["P"][:0 ],d ["D"][:0 ],s [:0 ]))
    if len (P )>1 :
        nm =wire_gate .crowd_mask (P ,sk )
        P ,D ,sk =P [nm ],D [nm ],sk [nm ]
    return P ,D ,sk 


def warn_position (p ,g ,gd ):
    n =np .linalg .norm (gd )
    if n <1e-9 :
        return False ,None 
    u =gd /n 
    w =p -g 
    e =float (w @u )
    return (np .linalg .norm (w -e *u )<=YANAL and abs (e )<=EKSENEL ),u 


def main ():
    gate =pickle .load (open ("results/kazanan_hgb_derin.pkl","rb"))["HGB-derin"]
    S =K .step_map ()
    global _D6 
    _D6 ={str (p ):r for p ,r in 
    d6_record .yukle (set (d6_record .exam ()["pidler"])).items ()}
    Rk =K .yukle (None )

    def rec_ (pid ):
        return Rk .get (pid )or _D6 .get (pid )

        # --- EGITIM: corpus
    X ,Y =[],[]
    pid_tam =[f [4 :-4 ]for f in sorted (os .listdir (OZ ))if f .startswith ("tam_")]
    for i ,pid in enumerate (pid_tam ):
        r =rec_ (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        d =parca_yukle ("tam",pid ,r )
        if d is None :
            continue 
        P ,D ,sk =pick_gate (gate ,d )
        if not len (P ):
            continue 
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        for a in range (len (P )):
            j =-1 
            for t in range (len (G )):
                ok ,_u =warn_position (P [a ],G [t ],Gd [t ])
                if ok :
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
                X .append (F_ [q ]);Y .append (int (ac <=ACI ))
        if (i +1 )%400 ==0 :
            print (f"  training {i +1 }/{len (pid_tam )} secenek {len (X )}",flush =True )
    X =np .asarray (X ,float );Y =np .asarray (Y ,int )
    print (f"YON SECENEGI {X .shape } | pozitif {Y .mean ():.4f}",flush =True )
    yc =HistGradientBoostingClassifier (max_iter =400 ,learning_rate =0.08 ,
    max_leaf_nodes =63 ,random_state =0 
    ).fit (X ,Y )
    pickle .dump (yc ,open ("results/b1_yon_secici.pkl","wb"))

    # --- SINAV: D7 full zincir
    out ={}
    for ad ,kullan in (("YON SABIT",False ),("MESH NORMALI",True )):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
        tes =[]
        for pid in [f [3 :-4 ]for f in sorted (os .listdir (OZ ))
        if f .startswith ("d7_")]:
            r =rec_ (pid )
            if r is None or not len (r .get ("G",[])):
                continue 
            d =parca_yukle ("d7",pid ,r )
            if d is None :
                continue 
            P ,D ,sk =pick_gate (gate ,d )
            if kullan and len (P ):
                D =D .copy ()
                for a in range (len (P )):
                    V_ ,F_ =secenekler (P [a ],D [a ],d ["V"],d ["NV"],d ["ppos"],
                    sk [a ],len (P ))
                    if len (V_ )<2 :
                        continue 
                    s2 =yc .predict_proba (F_ )[:,1 ]
                    b =int (np .argmax (s2 ))
                    if b !=0 and s2 [b ]>s2 [0 ]:
                        D [a ]=V_ [b ]
            if len (P ):
                P ,D =product_zinciri .tam_poz (d ["V"],d ["F"],d ["pb"],P ,D ,
                step_path =S .get (pid ))
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            dg =float (r ["diag"])
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,YANAL ,ACI ,False ,
            signed =True )[:3 ]
            a_ =rob [r ["mfg"]]
            a_ [0 ]+=tp ;a_ [1 ]+=fp ;a_ [2 ]+=fn 
            tes .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,dg ,
            max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ])
        pm ={m :2 *v [0 ]/max (2 *v [0 ]+v [1 ]+v [2 ],1 )
        for m ,v in rob .items ()}
        mi =float (2 *sum (v [0 ]for v in rob .values ())/
        max (sum (2 *v [0 ]+v [1 ]+v [2 ]for v in rob .values ()),1 ))
        out [ad ]={"robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"brand":pm }
        print (f"{ad :<14} robot {mi :.4f} | tespit {out [ad ]['tespit']:.4f} | "
        f"makro {out [ad ]['makro']:.4f} | en kotu {out [ad ]['en_kotu']:.4f}",
        flush =True )
    a ,b =out ["YON SABIT"]["robot"],out ["MESH NORMALI"]["robot"]
    print (f"\nFARK {b -a :+.4f} | kova aritmetigi beklentisi ~0.407")
    print ("KAPI: >= +0.02 whereas KABUL")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,"fark":b -a ,
    "not":"Yon secenekleri = mevcut + 2mm icindeki mesh tepelerinin "
    "+/- normalleri. Konum ve candidate sayisi DEGISMEZ. "
    "D7 brand-disi, TAM ZINCIR, MIKRO."},
    open ("results/b1_mesh_normal.json","w"),indent =1 )
    print ("receipt -> results/b1_mesh_normal.json")


if __name__ =="__main__":
    main ()
