# -*- coding: utf-8 -*-
"""IKI DOKUMU ESLI KIYASLA -- genel amacli.

Kullanim:
    python probe_dump_compare.py TABAN.json KOL.json [path]

Ortak parcalarda, three metrikte (detection / robot unsigned / robot ISARETLI)
part duzeyi ESLI bootstrap yapar. Kiyas same parcalarda oldugu for
correct test marjinal confidence araligi not FARKIN dagilimidir.
"""
import json 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import match_hungarian # noqa: E402

OLCUTLER =(("detection",0.0 ,180.0 ,True ,False ),
("rob",2.0 ,10.0 ,False ,False ),
("rbi",2.0 ,10.0 ,False ,True ))


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def yukle (fp ,path ):
    return {str (r ["pid"]):r for r in json .load (open (fp ))
    if r .get ("path")==path }


def olc (kayitlar ,pidler ):
    tot ={k [0 ]:[0 ,0 ,0 ]for k in OLCUTLER }
    part =[]
    for pid in pidler :
        r =kayitlar [pid ]
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        if not len (G ):
            continue 
        Gd =_birim (r ["Gd"])
        P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
        D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
        line_ ={}
        for ad ,tol ,am ,pct ,isr in OLCUTLER :
            tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,float (r ["diag"]),
            tol ,am ,pct ,signed =isr )
            tot [ad ][0 ]+=tp 
            tot [ad ][1 ]+=fp 
            tot [ad ][2 ]+=fn 
            line_ [ad ]=(tp ,fp ,fn )
        part .append (line_ )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    return {k :f1 (v )for k ,v in tot .items ()},part 


def boot (pa ,pb ,ad ,n =4000 ,seed =0 ):
    rng =np .random .default_rng (seed )
    A =np .asarray ([r [ad ]for r in pa ],float )
    B =np .asarray ([r [ad ]for r in pb ],float )
    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    d =[]
    for _ in range (n ):
        i =rng .integers (0 ,len (A ),len (A ))
        d .append (f1 (B [i ].sum (0 ))-f1 (A [i ].sum (0 )))
    d =np .asarray (d )
    return d .mean (),np .percentile (d ,2.5 ),np .percentile (d ,97.5 ),(d >0 ).mean ()


def main ():
    if len (sys .argv )<3 :
        print (__doc__ )
        return 2 
    fa ,fb =sys .argv [1 ],sys .argv [2 ]
    path =sys .argv [3 ]if len (sys .argv )>3 else "field"
    A ,B =yukle (fa ,path ),yukle (fb ,path )
    ortak =sorted (set (A )&set (B ))
    print (f"path={path } | {os .path .basename (fa )} ({len (A )}) vs "
    f"{os .path .basename (fb )} ({len (B )}) -> ortak {len (ortak )} part")
    if not ortak :
        print ("ORTAK PARCA YOK")
        return 1 
    ma ,pa =olc (A ,ortak )
    mb ,pb =olc (B ,ortak )
    print (f"\n{'':22s} {'detection':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    print (f"{'TABAN':22s} {ma ['detection']:8.4f} {ma ['rob']:8.4f} "
    f"{ma ['rbi']:8.4f}")
    print (f"{'KOL':22s} {mb ['detection']:8.4f} {mb ['rob']:8.4f} "
    f"{mb ['rbi']:8.4f}")
    print (f"\n{'metrik':>8s} {'diff':>9s} {'%95 GA':>22s} {'poz%':>6s}")
    for ad ,*_r in OLCUTLER :
        f ,lo ,hi ,pz =boot (pa ,pb ,ad )
        yz =" *"if (lo >0 or hi <0 )else ""
        print (f"{ad :>8s} {f :+9.4f} [{lo :+.4f},{hi :+.4f}]{yz :>3s} "
        f"{100 *pz :5.1f}")
    print ("\n(* = %95 GA sifiri icermiyor)")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
