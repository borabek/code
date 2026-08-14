# -*- coding: utf-8 -*-
"""TTA SONDASI: inference aninda DONDURME ortalamasi havuzu iyilestirir mi?

WHY BU KOL FARKLI: bugun denenen nine kolun all of them SECICI tarafindaydi and
all of them same takas egrisi on gezindi (precision artarken geri cagirma
dusuyor). Bu arm GIRDI KALITESINI changes: `p_pos` iyilesirse hem ADAYLAR
hem de 58 gate ozniteliginin most birden iyilesir -- egriyi YUKARI kaydirabilir.

DAYANAK: `results/_p1_olasilik_d7` **single checkpoint** with uretilmis
(`pbs` sekli (1, N, 5)); urun tarihsel as ensemble kullaniyordu. Ayrica
agin input ozniteligi `xyz` (ckpt cfg), i.e. DONDURMEYE DUYARLI -- same mesh
different yonelimlerde different prediction gives. TTA this varyansi ortalar.

FARKI: [[b9-rotation-augmentation]] EGITIMDE dondurme artirmasiydi and kahini
dusurmustu. Bu, EGITIME DOKUNMAZ; only cikarimda K yonelimin ortalamasi
alinir. Tez degismezleri (5 sinif, ~6000 remesh, `v_o`) AYNEN korunur.

OLCUT: HAVUZ RECALL (F1 not) -- new a seg ciktisi before bununla sinanir
([[kismi-ce-denetimi-recall-dusuruyor]] dersi). Ornek: D7'den 100 part.
"""
import json 
import os 
import sys 
import time 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import diffusionnet as D_ # noqa: E402
import canonical_d7 as K # noqa: E402
import robot_cp # noqa: E402
from infer_step_cp import load_any # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OB ="results/_p1_olasilik_d7"
CKPT ="results/seg_g7/g7_s0.pt"
N_PARCA =int (os .environ .get ("TTA_N","100"))
ACILAR =(0.0 ,15.0 ,30.0 ,45.0 )# derece, z ekseni etrafinda


def _dondur (V ,aci_derece ,axis ):
    a =np .radians (aci_derece )
    k =np .asarray (axis ,float )
    k =k /np .linalg .norm (k )
    K_ =np .array ([[0 ,-k [2 ],k [1 ]],[k [2 ],0 ,-k [0 ]],[-k [1 ],k [0 ],0 ]])
    R =np .eye (3 )+np .sin (a )*K_ +(1 -np .cos (a ))*(K_ @K_ )
    return V @R .T 


def main ():
    dev ="cuda"
    try :
        import torch 
        dev ="cuda"if torch .cuda .is_available ()else "cpu"
    except Exception :
        dev ="cpu"
    model ,meta =load_any (CKPT ,dev =dev )[:2 ]
    S =K .step_map ()
    pids =sorted (f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz"))
    kay =K .yukle (pids )
    secili =[p for p in pids if p in kay and len (kay [p ].get ("G",[]))][:N_PARCA ]
    print (f"ckpt {CKPT } | cihaz {dev } | part {len (secili )} | "
    f"aci {ACILAR }",flush =True )

    top ={"tek":[0 ,0 ],"tta":[0 ,0 ]}
    t0 =time .time ()
    for i ,pid in enumerate (secili ,1 ):
        r =kay [pid ]
        z =np .load (f"{OB }/{pid }.npz")
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (r ["diag"])
        tol =max (3.0 ,0.06 *dg )
        pb_tek =np .asarray (z ["pbs"],float ).mean (0 )
        # TTA: same mesh, K yonelim; olasiliklar ORTALANIR (vertex order korunur,
        # only input koordinatlari returns -> output vertex basina hizali kalir)
        pbs =[]
        for a in ACILAR :
            Vr =V if a ==0.0 else _dondur (V -V .mean (0 ),a ,(0 ,0 ,1 ))+V .mean (0 )
            _lb ,pb =D_ .predict (model ,meta ,np .ascontiguousarray (Vr ),F ,
            device =dev ,op_cache_dir =None ,
            return_probs =True )
            pbs .append (np .asarray (pb ,float ))
        pb_tta =np .mean (pbs ,axis =0 )
        for ad ,pb in (("tek",pb_tek ),("tta",pb_tta )):
            cps ,_o ,_c ,_p =robot_cp .derive_candidates (V ,F ,[pb ],S .get (pid ))
            P =np .asarray ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            Dd =np .asarray ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            tp ,_f ,fn =match_hungarian (P ,Dd ,G ,Gd ,dg ,tol ,180.0 ,True )[:3 ]
            top [ad ][0 ]+=tp 
            top [ad ][1 ]+=fn 
        if i %20 ==0 :
            print (f"  {i }/{len (secili )} ({(time .time ()-t0 )/i :.1f}s/part)",
            flush =True )
    out ={}
    for ad ,v in top .items ():
        out [ad ]=v [0 ]/max (v [0 ]+v [1 ],1 )
        print (f"{ad :<5} pool recall {out [ad ]:.4f}  (TP {v [0 ]} FN {v [1 ]})")
    d =out ["tta"]-out ["tek"]
    print (f"\nFARK {d :+.4f} | KAPI: >= +0.02 ise TAM OLCEGE gecilir")
    json .dump ({"damga":receipt_hash .damga (),"recall":out ,"diff":d ,
    "acilar":list (ACILAR ),"n_parca":len (secili ),
    "not":"TTA sondasi: cikarimda dondurme ortalamasi. Olcut HAVUZ "
    "RECALL. Egitime DOKUNULMADI, tez degismezleri korundu."},
    open ("results/tta_pool.json","w"),indent =1 )
    print ("receipt -> results/tta_pool.json")


if __name__ =="__main__":
    main ()
