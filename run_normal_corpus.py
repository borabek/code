# -*- coding: utf-8 -*-
"""MESH NORMALI KORPUSU: each adaya TEK direction (own surface normali)

FINDING (2026-08-12). Mesh tepesinin own normali GT yonunu ISARETLI as
NIT'te %80, SUPU'da %93.6 tutuyor. Yani direction OGRENILMESI gereken a karar
not, GEOMETRIDEN OKUNAN a buyukluk.

Bugun each adaya `direction_bank` ~24 option takiyor and selector 4857 option
icinden ~24 dogruyu bulmaya calisiyor (pozitif yogunlugu 1/202). Bu corpus
each adaya TEK option birakir -> ~350 option, yogunluk 1/15 (13.5 KAT).

WHY YENIDEN CIKARIM DEGIL DE SUZGEC: feature sutunlarinin C and D
bloklari SECENEK YONUNE bagli hesaplanmis durumda. Yeniden inference ~4 saat.
Onun instead of each adayin seceneklerinden, yonu mesh normaline EN YAKIN olani
TUTULUR -- features that option for already correct hesaplanmistir.

Kullanim:
    NK_ON=d6 NK_CIK=results/_p6_oz_normal python run_normal_corpus.py
"""
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")

KAYNAK =os .environ .get ("NK_KAYNAK","results/_p6_oz_tam4")
CIK =os .environ .get ("NK_CIK","results/_p6_oz_normal")
ON =os .environ .get ("NK_ON","d6")
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit",
"d7":"results/_p1_olasilik_d7"}[ON ]
SHARD =os .environ .get ("NK_SHARD")


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def main ():
    import trimesh 
    os .makedirs (CIK ,exist_ok =True )
    fs =sorted (f for f in os .listdir (KAYNAK )
    if f .startswith (ON +"_")and f .endswith (".npz"))
    if SHARD :
        i_ ,n_ =(int (x )for x in SHARD .split ("/"))
        fs =[f for k ,f in enumerate (fs )if k %n_ ==i_ ]
        print (f"  PAY {i_ }/{n_ }",flush =True )
    print (f"{ON }: {len (fs )} part | {KAYNAK } -> {CIK }",flush =True )
    t0 =time .time ()
    yaz =skip =empty_ =0 
    pre_ =post_ =0 
    for i ,f in enumerate (fs ,1 ):
        hedef =f"{CIK }/{f }"
        if os .path .exists (hedef ):
            skip +=1 
            continue 
        pid =f [len (ON )+1 :-4 ]
        mf =f"{MESH }/{pid }.npz"
        if not os .path .exists (mf ):
            empty_ +=1 
            continue 
        z =np .load (f"{KAYNAK }/{f }")
        X ,idx ,YD =z ["X"],np .asarray (z ["idx"],int ),np .asarray (z ["YD"],float )
        P ,D ,kay =z ["P"],z ["D"],z ["source"]
        if not len (idx ):
            empty_ +=1 
            continue 
        zz =np .load (mf )
        V =np .ascontiguousarray (zz ["V"],np .float64 )
        Fc =np .ascontiguousarray (zz ["F"],np .int64 )
        mesh =trimesh .Trimesh (V ,Fc ,process =False )
        VN =_birim (np .asarray (mesh .vertex_normals ,float ))
        Pc =np .asarray (P ,float )
        # each ADAYIN normali: most yakin mesh tepesinin normali
        yak =np .argmin (np .linalg .norm (Pc [:,None ,:]-V [None ,:,:],
        axis =-1 ),axis =1 )
        AN =VN [yak ]# (candidate, 3)
        YDn =_birim (YD )
        # each option: own adayinin normaliyle ISARETLI angle
        cos =(YDn *AN [idx ]).sum (1 )
        # candidate basina EN YAKIN secenegi tut
        tut =np .zeros (len (idx ),bool )
        rank_ =np .lexsort ((-cos ,idx ))
        _ ,first_ =np .unique (idx [rank_ ],return_index =True )
        tut [rank_ [first_ ]]=True 
        pre_ +=len (idx )
        post_ +=int (tut .sum ())
        np .savez_compressed (hedef +".tmp",
        X =X [tut ],idx =idx [tut ],YD =YD [tut ],
        P =P ,D =D ,src_ =kay )
        os .replace (hedef +".tmp.npz",hedef )
        yaz +=1 
        if i %50 ==0 :
            print (f"  {i }/{len (fs )} written {yaz } | option "
            f"{pre_ /max (yaz ,1 ):.0f} -> {post_ /max (yaz ,1 ):.0f} "
            f"({time .time ()-t0 :.0f} s)",flush =True )
    print (f"\nBITTI: written {yaz } | skipped {skip } | bos {empty_ }")
    if yaz :
        print (f"option/part: {pre_ /yaz :.0f} -> {post_ /yaz :.0f} "
        f"({pre_ /max (post_ ,1 ):.1f}x azalma)")


if __name__ =="__main__":
    main ()
