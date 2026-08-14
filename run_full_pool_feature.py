# -*- coding: utf-8 -*-
"""TAM HAVUZ (seg + B-rep + mesh tepeleri) for gate ozniteliklerini uret.

Olculdu (`results/tavan_080_eksensiz.json`, D7 brand-disi, signed angle,
mukemmel selector): this havuzun robot F1 TAVANI **0.8347**, part basina ~318 candidate.
Karsilastirma: urunun bugunku havuzu 13.4 candidate / ceiling 0.3430; this gece kazanan
B-rep havuzu 98 candidate / ceiling 0.5761.

Yeniden baslatilabilir: each part own npz'sine yazilir, present which is atlanir.
Diske `X` (58 column), `y` (PROJENIN label tanimi), `P`, `D`, `source` yazilir.

ETIKET: `build_zengin_parite.py` with BIREBIR -- lateral distance + axial 40mm
kapisi + acgozlu BIRE-BIR eslesme. (Oklid + coka-a etiketi gate'i 0.08
bozuyordu, see. gate-label-tanimi-hatasi-and-v6-kunyesi.)
"""
import argparse 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import brep_pool # noqa: E402
import connector3d # noqa: E402
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import wire_gate # noqa: E402

CE ,CT =int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )
CIK ="results/_tam_oz"
EKSENEL =40.0 


def project_label (P ,G ,Gd ,diag ):
    y =np .zeros (len (P ),int )
    if not len (P )or not len (G ):
        return y 
    tol =max (3.0 ,0.06 *float (diag ))
    Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gn [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gn [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )<=EKSENEL ,pe ,np .inf )
    up ,ug =set (),set ()
    for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
    for a in range (len (P ))for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :
            continue 
        up .add (a_ );ug .add (b_ );y [a_ ]=1 
    return y 


def kayitlar (ad ):
    if ad =="d6":
        return d6_record .yukle (set (d6_record .exam ()["pidler"]))
    if ad =="d7":
        return K .yukle (json .load (open ("results/d7_sinav_kumesi.json"))["pidler"])
    return K .yukle ([str (p )for p in json .load (
    open ("results/brep_egitim_kumesi.json"))["pidler"]])


ISLER ={
"d7":("results/_p1_olasilik_d7","results/_d7_silindirler.pkl",
"results/_d7_acikliklar.pkl"),
"d6":("results/_p1_olasilik_g7","results/_d6_silindirler.pkl",
"results/_d6_acikliklar.pkl"),
"tam":("results/_p1_olasilik_brepegit","results/_brepegit_silindirler.pkl",
"results/_brepegit_acikliklar.pkl"),
}


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--cluster",nargs ="+",default =["d7","d6","tam"])
    a =ap .parse_args ()
    os .makedirs (CIK ,exist_ok =True )
    S =K .step_map ()
    for ad in a .cluster :
        ob ,cylf ,acf =ISLER [ad ]
        cy =pickle .load (open (cylf ,"rb"))
        ac =pickle .load (open (acf ,"rb"))
        kay =kayitlar (ad )
        t0 =time .time ()
        yazilan =atlanan =yok =0 
        for i ,(pid ,r )in enumerate (sorted (kay .items ()),1 ):
            pid =str (pid )
            yol =f"{CIK }/{ad }_{pid }.npz"
            if os .path .exists (yol ):
                atlanan +=1 
                continue 
            G =np .asarray (r .get ("G",[]),float )
            f =f"{ob }/{pid }.npz"
            if not len (G )or not os .path .exists (f ):
                yok +=1 
                continue 
            z =np .load (f )
            V =np .ascontiguousarray (z ["V"],np .float64 )
            F =np .ascontiguousarray (z ["F"],np .int64 )
            pb =np .asarray (z ["pbs"],float ).mean (0 )
            ppos =pb [:,CE ]+pb [:,CT ]
            P ,D ,kayn =brep_pool .tam_havuz (
            np .asarray (r ["P"],float ),np .asarray (r ["Pd"],float ),
            cy .get (pid ),ac .get (pid ),V =V ,F =F ,ppos =ppos )
            if len (P )<2 :
                yok +=1 
                continue 
            X =np .asarray (wire_gate .feats_for (
            V ,F ,pb ,[{"point":P [j ],"direction":D [j ]}
            for j in range (len (P ))],
            CE ,CT ,step_path =S .get (pid )),float )
            y =project_label (P ,G ,np .asarray (r ["Gd"],float ),float (r ["diag"]))
            np .savez_compressed (yol ,X =X .astype (np .float32 ),y =y .astype (np .int8 ),
            P =P .astype (np .float32 ),D =D .astype (np .float32 ),
            kaynak =kayn .astype (np .int8 ))
            yazilan +=1 
            if yazilan %50 ==0 :
                h =(time .time ()-t0 )/yazilan 
                print (f"  {ad } {i }/{len (kay )} yazilan={yazilan } {h :.2f}s/part "
                f"kalan ~{h *(len (kay )-i )/60 :.0f}dk",flush =True )
        print (f"{ad } BITTI: yazilan {yazilan } | onbellekte {atlanan } | "
        f"girdisi yok {yok }",flush =True )
    print ("->",CIK )


if __name__ =="__main__":
    main ()
