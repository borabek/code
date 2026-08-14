# -*- coding: utf-8 -*-
"""K3: RECALL TABANI -- 'no parcada N'den few candidate kabul etme'.

Top-K CLOSED because TAM K secmeye zorluyordu: K yanlissa esikten kotu (-0.0145).
TABAN different -- wrong K'nin cezasi YOK, only asiri silent parts kurtulur.
Gerekce: gate'in reddettigi 248 correct adayin %77'si YOGUN parcalarda and that parcalarda
goreli threshold (0.5*max) single a guclu adayin golgesinde kaliyor.

DURUST SECIM: N a yarida secilir, diger yaride olculur.
"""
import io ,json ,sys 
import numpy as np 
sys .path .insert (0 ,".")
import tezgah2 as T2 
from sina_cluster import f1w 

DER ,gate ,ek =T2 .yukle ()
d0 ,r0 ,gg =T2 .puanla (DER ,gate )
print (f"\n{'arm':<30}{'tespit':>9}{'robot':>9}")
T2 .bas ("TABAN (goreli threshold)",d0 ,r0 )

def taban_karar (N ):
    def f (sk ,r ,X ):
        m =(sk >=0.5 *max (sk .max (),1e-9 ))&(sk >=0.25 )
        if m .sum ()<N and len (sk ):
            m =np .zeros (len (sk ),bool )
            m [np .argsort (sk )[::-1 ][:min (N ,len (sk ))]]=True 
        return m 
    return f 

TAM ={}
for N in (2 ,3 ,4 ,5 ):
    d ,r ,_ =T2 .puanla (DER ,gate ,karar =taban_karar (N ))
    TAM [N ]=(f1w (d ),f1w (r ))
    T2 .bas (f"N={N } tabani",d ,r ,baseline =(d0 ,r0 ),gg =gg )

    # DURUST CAPRAZ SECIM
gruplar =sorted ({r ["geo"]for r in DER });rng =np .random .default_rng (0 )
kar =list (gruplar );rng .shuffle (kar );A =set (kar [:len (kar )//2 ])
altA =[r for r in DER if r ["geo"]in A ];altB =[r for r in DER if r ["geo"]not in A ]
dc ,dt ,gg2 =[],[],[]
for sec ,olc in ((altA ,altB ),(altB ,altA )):
    en ,ea =-1 ,2 
    for N in (2 ,3 ,4 ,5 ):
        dd ,_ ,_ =T2 .puanla (sec ,gate ,karar =taban_karar (N ))
        if f1w (dd )>en :en ,ea =f1w (dd ),N 
    d1 ,_ ,g1 =T2 .puanla (olc ,gate ,karar =taban_karar (ea ))
    d2 ,_ ,_ =T2 .puanla (olc ,gate )
    dc +=d1 ;dt +=d2 ;gg2 +=g1 
    print (f"  yarida secilen N={ea } -> diger yaride measured ({len (olc )} part)")
lo ,hi =T2 .ga (dt ,dc ,gg2 )
print (f"\nDURUST: {f1w (dt ):.4f} -> {f1w (dc ):.4f}  ({f1w (dc )-f1w (dt ):+.4f})  GA[{lo :+.4f},{hi :+.4f}]")
gecti =(f1w (dc )-f1w (dt ))>=0.01 and lo >0 
print (f"KILL: tespit +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'}")
json .dump ({"tam":{str (k ):{"tespit":v [0 ],"robot":v [1 ]}for k ,v in TAM .items ()},
"durust_taban":f1w (dt ),"durust_aday":f1w (dc ),"ga":[lo ,hi ],
"gecti":bool (gecti )},io .open ("results/k3_recall_tabani.json","w"),indent =1 )
print ("receipt -> results/k3_recall_tabani.json")
