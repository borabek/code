# -*- coding: utf-8 -*-
"""DECISION DENEYI: robotun buldugu acikliklarda TEL-girisi with TOOL/mount agzi GEOMETRIYLE ayriliyor mu?

Sinif (Contact) ayirt edemiyor (tez: Contact=Kontaktierung bzw. Werkzeugeinschub). Peki geometri?
Uretici ConnectionPoint'i (= real tel-kontakt) AYRAC-ETIKETI as kullan: a robot-CP'si a
manufacturer CP'sine axis-aware eslesiyorsa = TEL-girisi; eslesmiyorsa = TOOL/mount/extra.
Sonra two grubun size_mm / depth_mm / votes / confidence dagilimlarini karsilastir.
  Ayriliyorsa (or. tel more large/derin) -> GEOMETRIK KAPI genel fix (yeniden training absent).
  Ayrilmiyorsa -> lower-sinif etiketi (tool'u Contact'tan ayir) + yeniden training is required.
Sizinti-siz 145 WEI + PXC ornegi. Cikti: results/recon_wire_tool.json + ayrilabilirlik ozeti.
"""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 

HELD =set (open ("_hw_r3.txt").read ().split ())


def wire_flags (P ,G ,Gd ,tol ,axis_tol =40.0 ):
    """each robot-CP for: a manufacturer CP'ye eslesiyor mu (=tel-girisi)."""
    flag =np .zeros (len (P ),bool )
    if not len (P )or not len (G ):return flag 
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )<=axis_tol ,pe ,np .inf )
    order =sorted ((pe [i ,k ],i ,k )for i in range (len (P ))for k in range (len (G ))if pe [i ,k ]<=tol )
    up ,ug =set (),set ()
    for d ,i ,k in order :
        if i in up or k in ug :continue 
        up .add (i );ug .add (k );flag [i ]=True 
    return flag 


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    parts =[p for p in eligible ()if (p [0 ]=="WEI"and p [1 ]in HELD )]
    parts +=[p for p in eligible ()if p [0 ]=="PXC"][:90 ]
    print (f"{len (parts )} part | vote>=2 robot",flush =True )

    recs =[]# (wire?, size_mm, depth_mm, votes, conf, mfg)
    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            per =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
                pb =np .asarray (pb ,float );lab =pb .argmax (-1 )
                per .append (cp_openings .connection_points (V ,F ,lab ,min_v =30 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =pb ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 ))
            cps =_vote2 (per )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            fl =wire_flags (P ,G ,Gd ,tol )
            for c ,w in zip (cps ,fl ):
                area =float (c .get ("area",0.0 ));size =2.0 *(area /np .pi )**0.5 if area >0 else 0.0 
                recs .append ((int (w ),size ,float (c .get ("insertion_depth_mm",0.0 )),
                int (c .get ("_votes",1 )),float (c .get ("confidence",0.0 )),1 if mfg =="WEI"else 0 ))
        except Exception :
            continue 
        if k %25 ==0 :print (f"  {k }/{len (parts )}  {time .time ()-t0 :.0f}s",flush =True )

    A =np .array (recs ,float )
    json .dump (recs ,open ("results/recon_wire_tool.json","w"))
    wire =A [A [:,0 ]==1 ];tool =A [A [:,0 ]==0 ]
    print (f"\n=== KARAR DENEYI ({len (A )} robot-CP: {len (wire )} tel-girisi / {len (tool )} tool/fazla) ===")
    for name ,col in (("size_mm",1 ),("depth_mm",2 ),("votes",3 ),("confidence",4 )):
        w =wire [:,col ];tl =tool [:,col ]
        # ayrilabilirlik: AUC benzeri (P(wire>tool))
        import itertools 
        auc =float ((w [:,None ]>tl [None ,:]).mean ())if len (w )and len (tl )else 0.5 
        print (f"  {name :11s}: tel medyan {np .median (w ):.2f}  tool medyan {np .median (tl ):.2f}  | ayrilabilirlik(AUC) {auc :.2f}")
    print ("\n  AUC ~0.5 = ayirt edilemez; >0.65 = zayif sinyal; >0.75 = geometrik gate ise yarar")


if __name__ =="__main__":
    main ()
