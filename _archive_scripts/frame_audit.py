# -*- coding: utf-8 -*-
"""EVDE-2a: WEI measurement butunlugu -- align_frames residual audit.

FBI supheci D2: WEI direction hatasi 24-90 derece kararsizdi (PXC 0). Eger STEP<->JSON frame hizalamasi
WEI'de bozuksa, axis-aware esleme SISTEMATIK kaciriyor -> FN yapay sisiyor (PHANTOM recall kaybi).
Bu script each WEI held-out parcasinda align_frames residual_mm'i olcer. Kucuk residual (~tessellation,
<1mm) = hizalama saglam; large (>2mm) = same geometri DEGIL / hizalama supheli -> that parcanin FN'leri
sayilmamali. GPU gerekmez.
"""
import os ,sys ,json ,glob 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh 
from big_arbiter import eligible 

HELD =set (open ("_hw_r3.txt").read ().split ())


def main ():
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =[p for p in eligible ()if p [0 ]=="WEI"and p [1 ]in HELD ]
    print (f"{len (raw )} WEI held-out | frame residual audit (GPU'suz)",flush =True )
    rows =[]
    for k ,(mfg ,pid ,jf ,stp )in enumerate (raw ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            ncp =len (j .get ("ConnectionPoints",[]))
            if not ncp :continue 
            Vr ,_ =step_to_mesh (stp )
            R ,t ,res =align_frames (Vr ,Vj )
            rows .append ({"pid":pid ,"residual_mm":float (res ),"mfg_cps":ncp ,
            "n_step":int (len (Vr )),"n_json":int (len (Vj ))})
        except Exception as e :
            rows .append ({"pid":pid ,"residual_mm":None ,"err":str (e )[:60 ]})
        if k %25 ==0 :print (f"  {k }/{len (raw )}",flush =True )

    ok =[r for r in rows if r .get ("residual_mm")is not None ]
    res =np .array ([r ["residual_mm"]for r in ok ])
    bad =[r for r in ok if r ["residual_mm"]>2.0 ]
    susp =[r for r in ok if 1.0 <r ["residual_mm"]<=2.0 ]
    print (f"\n=== FRAME AUDIT ({len (ok )} part) ===")
    print (f"  residual_mm: medyan {np .median (res ):.3f} | ort {res .mean ():.3f} | max {res .max ():.3f}")
    print (f"  SAGLAM (<=1mm):  {len (ok )-len (susp )-len (bad )}")
    print (f"  SUPHELI (1-2mm): {len (susp )}")
    print (f"  BOZUK  (>2mm):   {len (bad )}  <- this parcalarin FN'leri PHANTOM olabilir")
    if bad :
        tot_bad_cp =sum (r ["mfg_cps"]for r in bad )
        print (f"  bozuk parcalardaki manufacturer CP sayisi: {tot_bad_cp } (total recall paydasindan cikarsa recall yukselir)")
        for r in sorted (bad ,key =lambda x :-x ["residual_mm"])[:10 ]:
            print (f"    {r ['pid']}: {r ['residual_mm']:.2f}mm  ({r ['mfg_cps']} CP)")
    json .dump (rows ,open ("results/wei_frame_audit.json","w"),indent =1 )
    print ("  -> results/wei_frame_audit.json")


if __name__ =="__main__":
    main ()
