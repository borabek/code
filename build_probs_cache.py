# -*- coding: utf-8 -*-
"""Bir KUME for inference onbellegi kur (VAL / LOCKED / DEV).

Neden ayri betik: `_h_probs.pkl` only DEV kumesini tasiyor. VAL on karar SINAMAK
for that parcalarin olasiliklari is required; LOCKED for de single atislik final olcumde gerekecek.

IKI KORUMA (ikisi de this gece yasanmis hatalardan):
 1) ZEHIRLI PARCA: gmsh single a STEP'te asilabiliyor. Islenmeden ONCE name INFLIGHT dosyasina
    yazilir; yeniden baslatmada orada duran part kalici as atlanir.
 2) DEVAM PARCA KIMLIGINE according to: loop indeksine according to devam etmek, pool degisince wrong
    parcalari skips and metrigi SESSIZCE breaks (2026-07-29'da yasandi).

Kullanim:  python build_probs_cache.py val
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "val").lower ()
    OUT =f"results/_probs_{cluster }.pkl"
    INFLIGHT =f"results/_probs_{cluster }.inflight"
    SKIP =f"results/_probs_{cluster }.skip"

    import torch 
    import diffusionnet as D 
    import thesis_remesh 
    from infer_step_cp import step_to_mesh ,load_any 
    from cad_eval import align_frames 
    from big_arbiter import eligible 

    cfg =json .load (open ("cp_config.json"))
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    want =set (json .load (open ("results/split3.json"))[cluster ]["parts"])

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p not in want :
            continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    print (f"KUME={cluster }: {len (parts )}/{len (want )} part bulundu | cihaz={dev }",flush =True )

    cache =pickle .load (open (OUT ,"rb"))if os .path .exists (OUT )else []
    done ={r ["pid"]for r in cache }
    skip ={x .strip ()for x in open (SKIP )}if os .path .exists (SKIP )else set ()
    if os .path .exists (INFLIGHT ):
        st =open (INFLIGHT ).read ().strip ()
        if st :
            skip .add (st )
            with open (SKIP ,"a")as fh :
                fh .write (st +"\n")
            print (f"  [zehirli part] {st } kalici atlandi",flush =True )
        os .remove (INFLIGHT )

    todo =[p for p in parts if p [1 ]not in done and p [1 ]not in skip ]# KIMLIGE according to
    print (f"  bitmis {len (done )} | atlanan {len (skip )} | kalan {len (todo )}",flush =True )
    if not todo :
        print ("yapacak is yok");return 

    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["current_product"]["checkpoints"]]
    for k ,(mfg ,pid ,jf ,stp ,n )in enumerate (todo ,1 ):
        with open (INFLIGHT ,"w")as fh :
            fh .write (pid )
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,np .float32 ))
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (pid =pid ,V =V .astype (np .float32 ),F =F .astype (np .int32 ),
            pbs =[p_ .astype (np .float16 )for p_ in pbs ],stp =stp ,n =n ,
            G =(G -t )@R ,Gd =Gd @R ,
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))))
        except Exception as e :
            print (f"  [error] {pid }: {type (e ).__name__ }",flush =True )
        if os .path .exists (INFLIGHT ):
            os .remove (INFLIGHT )
        if k %10 ==0 :
            pickle .dump (cache ,open (OUT ,"wb"))
            print (f"  {k }/{len (todo )} (cache {len (cache )})",flush =True )
    pickle .dump (cache ,open (OUT ,"wb"))
    print (f"-> {OUT }: {len (cache )} part",flush =True )


if __name__ =="__main__":
    main ()
