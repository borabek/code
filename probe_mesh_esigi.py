# -*- coding: utf-8 -*-
"""MESH OLASILIK ESIGI: CWT'de pool why still %55'te tikaniyor?

FINDING. CWT'de TUM mesh tepeleriyle bile konum recall 0.5532 -- NIT'te same
kurulum 1.0000 veriyordu. Mesh adaylari `p_pos >= MESH_ESIK (0.50)` sartiyla
seciliyor. CWT'de segmentasyon zayifsa CP bolgelerindeki vertices this esigi
GECEMEZ and pool that bolgeleri never gormez.

Bu betik esigi tarar: 0.50 / 0.30 / 0.20 / 0.10 / 0.05. Her esikte
  * only-KONUM recall'u
  * candidate count (maliyet)
olculur. Amac, esigi dusurmenin CWT like markalarda havuzu acip acmadigini
gormek; bedeli candidate patlamasi olacaktir and regime kapisi + seyreltme onu
yonetmek zorundadir.

D7 SINAV KUMESIDIR: here YALNIZCA teshis is done. Esik secimi `full`
korpusunda yapilacak.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import brep_pool # noqa: E402
import connector3d # noqa: E402
import thin_pool # noqa: E402
import canonical_d7 as K # noqa: E402

YANAL ,EKSENEL =2.0 ,40.0 
ESIKLER =(0.50 ,0.30 ,0.20 ,0.10 ,0.05 )
KUME =os .environ .get ("MESH_KUME","d7")
MESH ={"d7":"results/_p1_olasilik_d7","d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
MARKALAR =[m for m in os .environ .get ("MESH_MARKA","").split (",")if m ]
N =int (os .environ .get ("SONDA_N","0"))


def konum_rec (P ,G ,Gd ):
    if not len (P )or not len (G ):
        return 0 
    Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    df =np .asarray (P ,float )[:,None ,:]-np .asarray (G ,float )[None ,:,:]
    al =(df *Gn [None ,:,:]).sum (-1 )
    yan =np .linalg .norm (df -al [...,None ]*Gn [None ,:,:],axis =-1 )
    return int (((yan <=YANAL )&(np .abs (al )<=EKSENEL )).any (0 ).sum ())


def main ():
    import d6_record 
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    on =KUME +"_"
    fs =sorted (f for f in os .listdir ("results/_tam_oz")if f .startswith (on ))
    if N :
        fs =fs [:N ]
    agg =collections .defaultdict (collections .Counter )
    for i ,f in enumerate (fs ,1 ):
        pid =f [len (on ):-4 ]
        r =kay .get (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        if MARKALAR and r ["mfg"]not in MARKALAR :
            continue 
        mf =f"{MESH }/{pid }.npz"
        if not os .path .exists (mf ):
            continue 
        z =np .load (f"results/_tam_oz/{f }")
        kk =np .asarray (z ["source"],int )
        P01 =np .asarray (z ["P"],float )[np .isin (kk ,(0 ,1 ))]
        zz =np .load (mf )
        V =np .ascontiguousarray (zz ["V"],np .float64 )
        Fc =np .ascontiguousarray (zz ["F"],np .int64 )
        pb =np .asarray (zz ["pbs"],float ).mean (0 )
        pp =thin_pool .ppos (pb ,connector3d .CABLE_ENTRY ,
        connector3d .CONTACT )
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        a =agg [r ["mfg"]]
        a ["part"]+=1 
        a ["gt"]+=len (G )
        a ["n01"]+=len (P01 )
        for e in ESIKLER :
            Pm ,_ =brep_pool .mesh_adaylari (V ,Fc ,pp ,e ,
            brep_pool .MESH_DEDUPE_MM )
            Pu =np .vstack ([P01 ,Pm ])if len (Pm )else P01 
            a [f"rec_{e }"]+=konum_rec (Pu ,G ,Gd )
            a [f"n_{e }"]+=len (Pu )
        if i %100 ==0 :
            print (f"  {i }/{len (fs )}",flush =True )

    print (f"\n{'brand':<7}{'part':>6}{'GT':>7}"+
    "".join (f"{('p>='+str (e )):>12}"for e in ESIKLER ))
    out ={}
    for m ,a in sorted (agg .items (),key =lambda x :-x [1 ]["gt"]):
        g =max (a ["gt"],1 )
        out [m ]={"part":a ["part"],"gt":a ["gt"],
        "recall":{str (e ):a [f"rec_{e }"]/g for e in ESIKLER },
        "aday_parca":{str (e ):a [f"n_{e }"]/max (a ["part"],1 )
        for e in ESIKLER }}
        print (f"{m :<7}{a ['part']:>6}{a ['gt']:>7}"+
        "".join (f"{a [f'rec_{e }']/g :>12.4f}"for e in ESIKLER ))
    print (f"\n{'brand':<7}{'':>13}"+
    "".join (f"{('candidate@'+str (e )):>12}"for e in ESIKLER ))
    for m ,a in sorted (agg .items (),key =lambda x :-x [1 ]["gt"]):
        p =max (a ["part"],1 )
        print (f"{m :<7}{'':>13}"+
        "".join (f"{a [f'n_{e }']/p :>12.0f}"for e in ESIKLER ))
    json .dump ({"damga":receipt_hash .damga (),"cluster":KUME ,"esikler":ESIKLER ,
    "sonuc":out ,
    "not":"Mesh probability esigi taramasi (YALNIZ KONUM recall'u). "
    "D7'de TESHIS amaclidir; threshold secimi `tam` korpusunda "
    "yapilir."},
    open (f"results/mesh_esigi_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/mesh_esigi_{KUME }.json")


if __name__ =="__main__":
    main ()
