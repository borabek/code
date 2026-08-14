# -*- coding: utf-8 -*-
"""ADAYIN KENDI NORMALI: yeniden cikarimi hak eden a ceiling present mi?

DURUM. "Mesh normali yonu %80 tutuyor" sondasi, kutudaki 6000 TEPENIN
HERHANGI BIRINI kabul ediyordu -- extra comert a criterion. Korpus suzgeci
denendiginde (each adaya bankadaki normale EN YAKIN secenek) yonlu recall
0.8926 -> 0.3679 dustu, NIT'te 0.8429 -> 0.027.

Iki ayri sebep vardi:
  (a) candidates 6000 tepeden ~490'a SEYRELTILMIS
  (b) "bankadaki normale most yakin secenek" != "normalin kendisi"

BU SONDA (b)'yi kaldirir: each ADAYA KENDI normali verilir and yonlu recall
olculur. Bu, yeniden cikarimin (approximately 4 saat) TAVANIDIR.

  ceiling >= ~0.80 whereas yeniden inference HAK EDILIR
  ceiling dusukse mekanizma OLU -- 4 saat harcanmaz

Ozniteliklere ihtiyac absent; only geometri. D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

KORPUS =os .environ .get ("AN_KORPUS","results/_p6_oz_tam4")
MESH =os .environ .get ("AN_MESH","results/_p1_olasilik")
ON =os .environ .get ("AN_ON","d6")
YANAL ,EKSENEL =2.0 ,40.0 


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def main ():
    import trimesh 
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    fs =sorted (f for f in os .listdir (KORPUS )
    if f .startswith (ON +"_")and f .endswith (".npz"))
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    t0 =time .time ()
    n =0 
    for f in fs :
        pid =f [len (ON )+1 :-4 ]
        r =kay .get (pid )
        if not r or not len (r .get ("G",[])):
            continue 
        mf =f"{MESH }/{pid }.npz"
        if not os .path .exists (mf ):
            continue 
        z =np .load (f"{KORPUS }/{f }")
        P =np .asarray (z ["P"],float )
        idx =np .asarray (z ["idx"],int )
        YD =_birim (np .asarray (z ["YD"],float ))
        if not len (P ):
            continue 
        zz =np .load (mf )
        V =np .ascontiguousarray (zz ["V"],np .float64 )
        Fc =np .ascontiguousarray (zz ["F"],np .int64 )
        mesh =trimesh .Trimesh (V ,Fc ,process =False )
        VN =_birim (np .asarray (mesh .vertex_normals ,float ))
        yak =np .argmin (np .linalg .norm (P [:,None ,:]-V [None ,:,:],
        axis =-1 ),axis =1 )
        AN =VN [yak ]# ADAYIN KENDI normali
        G =np .asarray (r ["G"],float )
        Gn =_birim (np .asarray (r ["Gd"],float ))
        v =P [:,None ,:]-G [None ,:,:]
        al =(v *Gn [None ,:,:]).sum (-1 )
        yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
        konum =(yan <=YANAL )&(np .abs (al )<=EKSENEL )# (candidate, gt)
        aci_n =np .degrees (np .arccos (np .clip (AN @Gn .T ,-1 ,1 )))
        a =ist [r ["mfg"]]
        a ["gt"].append (len (G ))
        a ["konum"].append (int (konum .any (0 ).sum ()))
        a ["normal"].append (int ((konum &(aci_n <=K .ACI )).any (0 ).sum ()))
        # KIYAS: mevcut BANKA (tum secenekler)
        kb =np .zeros (len (G ),bool )
        for j in range (len (G )):
            ad =np .where (konum [:,j ])[0 ]
            if not len (ad ):
                continue 
            m_ =np .isin (idx ,ad )
            if not m_ .any ():
                continue 
            aci =np .degrees (np .arccos (np .clip (YD [m_ ]@Gn [j ],-1 ,1 )))
            if (aci <=K .ACI ).any ():
                kb [j ]=True 
        a ["banka"].append (int (kb .sum ()))
        a ["candidate"].append (len (P ))
        n +=1 
        if n %60 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{'brand':<7}{'GT':>7}{'candidate/p':>8}{'KONUM':>8}"
    f"{'ADAY NORMALI':>14}{'banka (24 sec)':>16}")
    out ={}
    tg =tn =tb =0 
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =max (sum (a ["gt"]),1 )
        k =sum (a ["konum"])/g 
        nn =sum (a ["normal"])/g 
        bb =sum (a ["banka"])/g 
        tg +=g ;tn +=sum (a ["normal"]);tb +=sum (a ["banka"])
        out [m_ ]={"gt":g ,"konum":k ,"aday_normali":nn ,"banka":bb ,
        "aday_parca":float (np .mean (a ["candidate"]))}
        print (f"{m_ :<7}{g :>7}{np .mean (a ['candidate']):>8.0f}{k :>8.3f}"
        f"{nn :>14.3f}{bb :>16.3f}")
    print (f"{'TOPLAM':<7}{tg :>7}{'':>8}{'':>8}{tn /tg :>14.3f}{tb /tg :>16.3f}")
    json .dump ({"corpus":KORPUS ,"brand":out ,
    "toplam":{"aday_normali":tn /tg ,"banka":tb /tg },
    "not":"ADAYIN KENDI mesh normali vs 24 secenekli direction bankasi. "
    "Yeniden cikarimin TAVANI. D7'ye BAKILMADI."},
    open ("results/aday_normal_tavan.json","w"),indent =1 )
    print ("\nmakbuz -> results/aday_normal_tavan.json")
    print ("DECISION: candidate normali ~banka'ya yakinsa yeniden inference HAK EDILIR")
    print ("       (24 fold few secenek, same recall). Cok dusukse mekanizma OLU.")


if __name__ =="__main__":
    main ()
