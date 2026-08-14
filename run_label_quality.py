# -*- coding: utf-8 -*-
"""Y8 — ETIKET KALITESI AGIRLIKLANDIRMA

Secicinin etiketi `project_label` with uretiliyor: lateral distance + axial
40mm kapisi + acgozlu BIRE-BIR eslesme. Ama eslesmelerin GUVENILIRLIGI
equal not: GT'ye 0.2mm'de oturan a candidate with 1.9mm'de (tolerans siniri)
oturan a candidate AYNI agirlikta ogretiliyor.

FIKIR: pozitif orneklere, GT'ye YAKINLIGIYLA orantili weight ver.
Sinirdaki ambiguous eslesmeler modeli more few cekistirir.

Bu, [[metrik-cerrahisi-a1-reddedildi]]'den FARKLIDIR: orada METRIK
degistirilmeye calisilmisti (reddedildi); here metrik AYNI kalir,
only EGITIM agirligi degisir.

KOLLAR: baseline / yakinlik agirligi (lineer) / yakinlik agirligi (karesel)
KOSUL: TANIDIK brand (rastgele katlar). KAPI: +0.01. D7'ye BAKILMAZ.
"""
import collections ,json ,os ,sys ,time 
import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K 
import p6_decision 
from run_p6_ortak import yukle 
from sina_cluster import match_hungarian 

NEG_KAT =12 ;ITER ,LR ,YAPRAK ,L2R =200 ,0.06 ,63 ,1.0 
NMS =5.0 ;KURAL =("goreli",0.85 ,0.20 )

def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (np .float32 )

def yakinlik (d ):
    """each POZITIF secenek for GT'ye YANAL uzaklik (0 = full ustunde)."""
    P =np .asarray (d ["P"],float )[np .asarray (d ["idx"],int )]
    G =np .asarray (d ["G"],float );Gd =np .asarray (d ["Gd"],float )
    Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    if not len (G ):return np .zeros (len (P ))
    v =P [:,None ,:]-G [None ,:,:]
    al =np .einsum ("pgc,gc->pg",v ,Gn )
    yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
    return yan .min (1 )

def yap ():
    return HistGradientBoostingClassifier (max_iter =ITER ,learning_rate =LR ,
    max_leaf_nodes =YAPRAK ,l2_regularization =L2R ,random_state =0 )

def main ():
    t0 =time .time ()
    data_ =yukle ("d6",0 )
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int );d ["_M"]=temel (d )
        d ["_yak"]=yakinlik (d )
    print (f"{len (data_ )} part",flush =True )
    # KAT TOHUMU: first run +0.0144 with kapiyi gecti but TEK tohumluydu.
    # Uc tohumda da gecerse karar verilebilir.
    _t =int (os .environ .get ("Y8_TOHUM","1"))
    rng =np .random .default_rng (_t );pay =rng .permutation (len (data_ ))%3 
    KOLLAR =("baseline","lineer","karesel")
    agg ={k :collections .Counter ()for k in KOLLAR }
    for f_ in range (3 ):
        ic =[i for i in range (len (data_ ))if pay [i ]!=f_ ]
        dis =[i for i in range (len (data_ ))if pay [i ]==f_ ]
        M =np .vstack ([data_ [i ]["_M"]for i in ic ])
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        YK =np .concatenate ([data_ [i ]["_yak"]for i in ic ])
        rr =np .random .default_rng (0 )
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rr .choice (neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        for arm in KOLLAR :
            if arm =="baseline":
                W =np .ones (len (sec ),np .float32 )
            else :
            # yakinlik 0 -> weight 1 ; tolerans (2mm) -> weight 0.25
                t =np .clip (YK [sec ]/K .YANAL ,0.0 ,1.0 )
                W =(1.0 -0.75 *t )if arm =="lineer"else (1.0 -0.75 *t **2 )
                W =np .where (Y [sec ]==1 ,W ,1.0 ).astype (np .float32 )
            m =yap ().fit (M [sec ],Y [sec ],sample_weight =W )
            for i in dis :
                d =data_ [i ]
                s =m .predict_proba (d ["_M"])[:,1 ]
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,nms_mm =NMS )
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
                K .YANAL ,K .ACI ,False ,signed =True )[:3 ]
                c =agg [arm ];c ["tp"]+=tp ;c ["fp"]+=fp ;c ["fn"]+=fn 
        del M 
        print (f"  fold{f_ } bitti ({time .time ()-t0 :.0f} s)",flush =True )
    def f1 (c ):return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )
    last_ ={k :f1 (agg [k ])for k in KOLLAR }
    print (f"\n=== TABAN {last_ ['baseline']:.4f} ===")
    for k in KOLLAR [1 :]:
        fark =last_ [k ]-last_ ["baseline"]
        print (f"  {k :<10}{last_ [k ]:.4f}   {fark :+.4f}"
        +("  <- KAPI GECTI"if fark >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"kat_tohumu":_t ,"toplam":last_ ,
    "not":"Etiket kalitesi agirliklandirma: pozitife GT'ye "
    "YAKINLIGIYLA orantili agirlik. Metrik DEGISMEZ, "
    "yalnizca training agirligi. D7'ye BAKILMADI."},
    open (f"results/etiket_kalitesi_t{_t }.json","w"),indent =1 )
    print ("receipt -> results/etiket_kalitesi.json")

if __name__ =="__main__":
    main ()
