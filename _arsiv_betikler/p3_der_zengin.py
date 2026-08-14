# -*- coding: utf-8 -*-
"""P3: OLCUM onbellegine ZENGIN sutunlari ekle (uctan uca A/B for).

P2 candidate duzeyinde +0.0231 showed (konum9 +0.0114, cokyaricap24 +0.0169, all of them +0.0231; ucu de
HER IKI ureticide pozitif). Aday duzeyi this arastirmada ALTI KEZ yaniltti -> karar uctan uca.

Uctan uca measurement `results/_u4_der.pkl` onbellegini kullaniyor but that only 22 column tasiyor.
Bu betik AYNI candidates for 33 zengin sutunu hesaplayip new a cache writes. Olasilik
haritalari already diskte (results/_probs_*.pkl) -- network cikarimi TEKRARLANMAZ, ~5 dk.

NOTE: candidates DER onbellegindekilerle AYNI must be. Bu yuzden yeniden turetilmiyor;
onbellekteki P/Pd/X aynen korunuyor, only XR ekleniyor.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
GIRDI ="results/_u4_der.pkl"
CIKTI ="results/_der_zengin.pkl"


def main ():
    from build_zengin_parite import _normaller ,zengin 

    with open (GIRDI ,"rb")as f :
        DER =pickle .load (f )
    VF ={}
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        with open (cf ,"rb")as f :
            for r in pickle .load (f ):
                pid =os .path .basename (r ["stp"]).split ("_")[1 ]
                VF [pid ]=(np .ascontiguousarray (r ["V"],np .float64 ),
                np .ascontiguousarray (r ["F"],np .int64 ),
                [np .asarray (pb ,float )for pb in r ["pbs"]])
    print (f"{len (DER )} kayit | {len (VF )} parcanin mesh+olasiligi var",flush =True )

    out ,atlanan =[],0 
    for i ,r in enumerate (DER ,1 ):
        if i %50 ==0 :
            print (f"  {i }/{len (DER )}",flush =True )
        pid =r ["pid"]
        if pid not in VF or r ["X"]is None or not len (r ["P"]):
            atlanan +=1 
            r =dict (r );r ["XR"]=None 
            out .append (r );continue 
        V ,F ,pbs =VF [pid ]
        probs =sum (pbs )/len (pbs )
        cps =[{"point":p ,"direction":d }for p ,d in zip (r ["P"],r ["Pd"])]
        try :
            xr =zengin (V ,F ,probs ,cps ,_normaller (V ,F ))
        except Exception as e :
            print (f"    {pid }: {type (e ).__name__ }: {e }")
            atlanan +=1 ;xr =None 
        r =dict (r );r ["XR"]=xr 
        out .append (r )
    with open (CIKTI ,"wb")as f :
        pickle .dump (out ,f )
    n =sum (1 for r in out if r .get ("XR")is not None )
    print (f"\n-> {CIKTI } | zengin sutunlu kayit {n }/{len (out )} (atlanan {atlanan })")
    ok =next ((r ["XR"].shape for r in out if r .get ("XR")is not None ),None )
    print (f"   XR sekli ornegi: {ok }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
