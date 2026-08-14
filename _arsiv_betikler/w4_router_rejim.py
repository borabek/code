# -*- coding: utf-8 -*-
"""W4: ROUTER'IN GERCEK REJIM KARARI (denetimin P1.1 maddesi).

SORUN: very-CP multires kolu (s9) hedef parcalari `r["n"] >= 8` with secmisti -- i.e.
GT'DEKI CP SAYISIYLA. Urun calisma aninda bunu BILMEZ; rejimi `robot_cp._highcp_router`
geometriden prediction eder. Dolayisiyla s9'un "+0.0283 very-CP" count MUKEMMEL REJIM BILGISI
varsayan a TAVAN'dir, dagitilabilir a kazanc not.

BU BETIK: router'i 194 measurement parcasinda URUNUN KENDI YOLUNDAN calistirir (derive_candidates ->
n_union -> _highcp_router) and GT rejimiyle karsilastirir. Cikan karsiliklilik tablosu
multires kolunun real uygulama kumesini gives:
    router+ & GT very  -> kazanc buraya gelir
    router+ & GT low-> BEDEL buraya gelir (s9 this parcalari never olcmedi)
    router- & GT very   -> kazanc KACIRILIR

Cikti: results/w4_router_rejim.json (part basina karar + karsiliklilik)
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/w4_router_rejim.json"
ONBELLEK ="results/_w4_router.pkl"


def main ():
    import diffusionnet as D 
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )

    K ={}
    if os .path .exists (ONBELLEK ):
        with open (ONBELLEK ,"rb")as f :
            K =pickle .load (f )
        print (f"onbellekten {len (K )} part",flush =True )

    kalan =[r for r in DER if r ["pid"]not in K ]
    if kalan :
        cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
        dev ="cuda"if torch .cuda .is_available ()else "cpu"
        models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
        t0 =time .time ()
        for i_ ,r in enumerate (kalan ,1 ):
            if i_ %20 ==0 :
                print (f"  {i_ }/{len (kalan )}  {time .time ()-t0 :.0f}s",flush =True )
                with open (ONBELLEK ,"wb")as f :
                    pickle .dump (K ,f )
            try :
                stp =stp_of .get (r ["pid"])
                Vr ,Fr =step_to_mesh (stp )
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
                pbs =[]
                for model ,meta in models :
                    _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                    op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                    return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
                cps ,_ ,is_hi ,_ =RC .derive_candidates (V ,F ,pbs ,stp ,cfg =cfg )
                K [r ["pid"]]={"router":bool (is_hi ),"n_cand":len (cps )}
            except Exception as e :
                K [r ["pid"]]={"router":None ,"error":f"{type (e ).__name__ }: {e }"}
        with open (ONBELLEK ,"wb")as f :
            pickle .dump (K ,f )

            # --- KARSILIKLILIK
    sat =[]
    for r in DER :
        k =K .get (r ["pid"],{})
        if k .get ("router")is None :
            continue 
        sat .append ({"pid":r ["pid"],"geo":r ["geo"],"gt_cok":bool (r ["n"]>=8 ),
        "router":bool (k ["router"]),"n_gt":int (r ["n"]),
        "n_cand":int (k .get ("n_cand",0 ))})
    tp =sum (1 for s in sat if s ["router"]and s ["gt_cok"])
    fp =sum (1 for s in sat if s ["router"]and not s ["gt_cok"])
    fn =sum (1 for s in sat if not s ["router"]and s ["gt_cok"])
    tn =sum (1 for s in sat if not s ["router"]and not s ["gt_cok"])
    print (f"\nROUTER vs GT REJIM ({len (sat )} part)")
    print (f"{'':>14}{'GT cok-CP':>12}{'GT dusuk-CP':>13}")
    print (f"{'router +':>14}{tp :>12}{fp :>13}   <- multires BURAYA uygulanir")
    print (f"{'router -':>14}{fn :>12}{tn :>13}")
    kes =tp /max (tp +fp ,1 );rec =tp /max (tp +fn ,1 )
    print (f"\n  precision {kes :.3f} | recall {rec :.3f} | dogruluk {(tp +tn )/max (len (sat ),1 ):.3f}")
    print (f"  s9 hedefi (GT cok-CP)      : {tp +fn } part")
    print (f"  URUNUN hedefi (router +)   : {tp +fp } part "
    f"({fp } tanesi GERCEKTE dusuk-CP -> s9 bunlari HIC olcmedi)")
    yanlis =[s ["pid"]for s in sat if s ["router"]and not s ["gt_cok"]]
    kacan =[s ["pid"]for s in sat if not s ["router"]and s ["gt_cok"]]
    if yanlis :
        print (f"  yanlis yonlendirilen: {yanlis [:12 ]}")
    if kacan :
        print (f"  kacirilan cok-CP    : {kacan [:12 ]}")
    with io .open (OUT ,"w",encoding ="utf-8")as f :
        json .dump ({"karsiliklilik":{"tp":tp ,"fp":fp ,"fn":fn ,"tn":tn },
        "precision":kes ,"recall":rec ,"satirlar":sat ,
        "yanlis_yonlendirilen":yanlis ,"kacirilan":kacan },f ,indent =1 )
    print (f"receipt -> {OUT }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
