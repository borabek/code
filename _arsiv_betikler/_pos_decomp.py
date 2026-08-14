# -*- coding: utf-8 -*-
"""Robot konum hatasini axis-boyu (depth/konvansiyon) vs DIK (real nisan hatasi) ayristir.
Ops cache full oldugu for fast. Eslesenlerde: along = |proj ten mfg InsertDir|, perp = kalan.
"""
import os ,sys ,glob ,json 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
import robot_cp ,robot_e2e 

def run (mfg ,limit ,models ,dev ):
    parts =robot_e2e .eligible (mfg )[:limit ]
    along =[];perp =[];dirs =[]
    for m ,pid ,jf ,stp in parts :
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            cps =robot_cp .extract (models ,stp ,dev ,0.75 )
            if not cps :continue 
            Vr ,_ =step_to_mesh (stp );R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([c ["point"]for c in cps ],float )@R .T +t 
            Pd =np .array ([c ["direction"]for c in cps ],float )@R .T 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )# DIK distance
            pe_g =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            order =sorted ((pe_g [i ,k ],i ,k )for i in range (len (P ))for k in range (len (G ))if pe_g [i ,k ]<=tol )
            up ,ug =set (),set ()
            for d ,i ,k in order :
                if i in up or k in ug :continue 
                up .add (i );ug .add (k )
                along .append (abs (float (al [i ,k ])));perp .append (float (pe [i ,k ]))
                c =float (np .clip (Pd [i ]@Gd [k ]/(np .linalg .norm (Pd [i ])*np .linalg .norm (Gd [k ])+1e-9 ),-1 ,1 ))
                dirs .append (float (np .degrees (np .arccos (abs (c )))))
        except Exception :
            continue 
    if along :
        print (f"  {mfg }: {len (along )} eslesme | axis-boyu (depth/konvansiyon) medyan {np .median (along ):.1f}mm"
        f" | DIK nisan hatasi medyan {np .median (perp ):.1f}mm | direction medyan {np .median (dirs ):.1f} deg")
    else :
        print (f"  {mfg }: eslesme yok")

if __name__ =="__main__":
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    print (f"URUN: {len (models )} model cogunluk oyu  (konum hatasi ayristirmasi)")
    run ("PXC",40 ,models ,dev )
    run ("WEI",40 ,models ,dev )
