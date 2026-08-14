# -*- coding: utf-8 -*-
"""CC-E: 'this part very-CP mi' sorusunu METADATA OLMADAN, only geometriden cevaplayabilir miyiz?

WHY: conn_promote very-CP'de +0.072, low-CP'de -0.018. Kuresel uygulanamaz. Kosullu uygulamak
for robotun BILINMEYEN a parcada rejimi own anlamasi is required -- manufacturer CP count elde YOK.

KISAYOL: mv10 and mv10+promote cikarimlarinin IKISI DE same 1882 part for elde. Yonlendirmeyi
GERIYE DONUK simule edebiliriz; new inference gerekmiyor.

ADIM 1 (this file): part-duzeyi geometri (STEP bounding box -- meshlemeden, fast) + promote'suz
candidate count. Bunlarla 'very-CP mi' siniflandiricisi kurulur and AILE-DISI dogrulanir.
ADIM 2: yonlendirilmis F1 vs (a) each places mv10, (b) ORACLE yonlendirme.
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/cc_e_partgeom.json"


def part_geometry ():
    """STEP bbox -- gmsh.model.mesh.generate() CAGRILMAZ, that yuzden ~0.3s/part."""
    import gmsh 
    from big_arbiter import eligible 
    z =np .load ("results/rich_mv10_all.npz",allow_pickle =True )
    want =[str (x )for x in z ["part_ids"]]
    el ={p :s for m ,p ,jf ,s in eligible ()}
    out ={}
    if os .path .exists (OUT ):
        out =json .load (open (OUT ))
        print (f"  [devam] {len (out )} part already olculmus",flush =True )
    t0 =time .time ()
    for k ,pid in enumerate (want ,1 ):
        if pid in out or pid not in el :
            continue 
        gmsh .initialize ();gmsh .option .setNumber ("General.Terminal",0 )
        try :
            gmsh .open (el [pid ])
            b =gmsh .model .getBoundingBox (-1 ,-1 )
            ext =np .array (b [3 :])-np .array (b [:3 ])
            ext =np .sort (ext )[::-1 ]
            out [pid ]={"ext":[float (x )for x in ext ],
            "diag":float (np .linalg .norm (ext )),
            "area_box":float (2 *(ext [0 ]*ext [1 ]+ext [0 ]*ext [2 ]+ext [1 ]*ext [2 ])),
            "vol_box":float (ext [0 ]*ext [1 ]*ext [2 ])}
        except Exception as e :
            out [pid ]={"error":str (e )[:40 ]}
        finally :
            gmsh .finalize ()
        if k %200 ==0 :
            json .dump (out ,open (OUT ,"w"))
            print (f"  {k }/{len (want )}  {time .time ()-t0 :.0f}s",flush =True )
    json .dump (out ,open (OUT ,"w"))
    print (f"-> {OUT }  {len (out )} part  {time .time ()-t0 :.0f}s")
    return out 


if __name__ =="__main__":
    part_geometry ()
