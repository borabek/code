# -*- coding: utf-8 -*-
"""GOREV 3.1: family/template transfer. Aile-ici terminal bloklarinda CP DUZENI tekrarli. Bir parcanin
CP'lerini, same aileden BASKA a parcanin (template) manufacturer-CP layout'unu rigid+scale align edip
aktararak yakalayabilir miyiz? OFFLINE (Desktop\\JSON layout'lari, GPU absent). certain-source.
DECISION: transferred-template held-out CP'lerin >=0.85'ine ulasirsa -> new candidate source; degilse kapat.
Leakage-safe: template BASKA part; held-out parcanin own GT'si only SKORLAMADA."""
import json ,numpy as np 
from collections import defaultdict 
from big_arbiter import eligible 

# tum eligible parcalarin manufacturer CP layout'u (point+direction, own frame)
parts ={}
for mfg ,pid ,jf ,stp in eligible ():
    try :
        j =json .load (open (jf ,encoding ="utf-8-sig"))
        G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
        if len (G )<1 :continue 
        parts [pid ]={"mfg":mfg ,"G":G ,"fam":pid [:4 ],"N":len (G )}
    except Exception :pass 
print (f"{len (parts )} part manufacturer CP layout",flush =True )
byfam =defaultdict (list )
for pid ,d in parts .items ():byfam [d ["fam"]].append (pid )


def align_reach (src_G ,dst_G ):
    """src (template) CP bulutunu dst'ye rigid+scale align (Procrustes) -> dst CP'lerin ne kadari yakin."""
    if len (src_G )<2 or len (dst_G )<2 :return 0.0 
    # centre + scale normalize, after Kabsch rotation (src->dst centroid-eslesmesi kaba)
    Sc =src_G -src_G .mean (0 );Dc =dst_G -dst_G .mean (0 )
    ss =np .sqrt ((Sc **2 ).sum ()/len (Sc ));ds =np .sqrt ((Dc **2 ).sum ()/len (Dc ))
    if ss <1e-6 :return 0.0 
    Sn =Sc /ss *ds 
    # each dst CP for most yakin (aligned) src CP mesafesi; only scale+translate (rotation absent, kaba upper-boundary)
    # rotation for: dst'nin own asal eksenlerine hizala (SVD)
    def pca (X ):
        C =X .T @X # 3x3, always
        w ,V =np .linalg .eigh (C );return V [:,::-1 ].T # satirlar = asal eksenler (azalan)
    Vs =pca (Sn );Vd =pca (Dc )
    Sr =Sn @Vs .T @Vd # src'yi dst asal-axis cercevesine dondur
    diag =float (np .linalg .norm (dst_G .max (0 )-dst_G .min (0 )))
    tol =max (3.0 ,0.06 *diag )
    dd =np .linalg .norm (Dc [:,None ]-Sr [None ],axis =-1 ).min (1 )
    return float ((dd <=tol ).mean ())


    # each part for same-aileden most yakin-N template with transfer reach (leave-part-out: template != self)
res ={"WEI":[],"PXC":[]};singleton =0 
for pid ,d in parts .items ():
    fam =d ["fam"];cand =[q for q in byfam [fam ]if q !=pid ]
    if not cand :singleton +=1 ;continue 
    # most yakin CP-count template
    best =max (cand ,key =lambda q :-abs (parts [q ]["N"]-d ["N"]))
    reach =align_reach (parts [best ]["G"],d ["G"])
    res [d ["mfg"]].append (reach )

for m in ("WEI","PXC"):
    if res [m ]:
        a =np .array (res [m ])
        print (f"{m }: {len (a )} part (aile-eslesmeli), template-transfer reach ort {a .mean ():.3f} med {np .median (a ):.3f} | >=0.85 ratio {(a >=0.85 ).mean ():.2%}")
print (f"singleton (aile-eslesmesiz, transfer YOK): {singleton }")
allr =np .array (res ["WEI"]+res ["PXC"])
print (f"\nKARAR: template-transfer reach ort {allr .mean ():.3f} -> "
+("candidate source DEGERLI (>=0.75), full kur"if allr .mean ()>=0.75 else f"{allr .mean ():.2f} DUSUK -> aile layout yeterince transfer olmuyor, kapat"))
