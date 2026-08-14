# -*- coding: utf-8 -*-
"""A/B: channel_axis (direction duzeltmesi) URUN F1'ini degistiriyor mu?

WHY ZORUNLU: direction duzeltmesi cizim for not, cp_openings'in ICINE kondu. Orada direction only
raporlanmiyor -- `outward_min` kapisinda ADAY ELEMEK for de is used. Aday sayilari fiilen
degisti (3270115 CP13->CP14, 2770943 CP41->CP40), i.e. F1 yukari da asagi da gidebilir.
"Gorsel a correction" diye olcmeden gecmek, this depoda more before full as boyle patlamisti.

YONTEM: results/pitstop2_cache icindeki V/F/olasiliklar yeniden is used -- GPU'ya gerek absent,
segmentasyon ciktisi ikisinde de AYNI. Degisen single sey cp_openings'in direction hesabi.
Skorlama big_arbiter konvansiyonu: eksene dik distance + +-40mm axial pencere, regime ayrimli
and corpus-agirlikli (%89.5 low / %10.5 very).

KILL (olcumden before yazildi): corpus-agirlikli CP-F1 kaybi > 0.005 whereas direction duzeltmesi URUNDEN
GERI ALINIR (cizime ozel a bayrak arkasina tasinir).
"""
import os ,sys ,json ,importlib 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CACHE ="results/pitstop2_cache"
W ={"low":0.895 ,"very":0.105 }


def derive (rows ,use_axis ):
    """Tum parts for adaylari YENIDEN derive (direction duzeltmesi open/closed)."""
    os .environ ["CP_CHANNEL_AXIS"]="1"if use_axis else "0"
    import cp_openings 
    importlib .reload (cp_openings )
    import robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json"))
    pp =cfg ["prediction_postproc"]
    out ={}
    for r in rows :
        per =[]
        pb =r ["probs"]
        for _ in range (1 ):# onbellekte ORTALAMA probability present -> single turetme
            per .append (cp_openings .connection_points (
            r ["V"],r ["F"],pb .argmax (-1 ),min_v =int (pp ["min_vertices"]),classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
            ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
            conn_promote =(0.25 if r ["regime"]=="very"else 0.0 )))
        cps =per [0 ]
        thr =float (cfg .get ("robot_wire_gate_threshold_highcp",0.25 ))if r ["regime"]=="very"else float (cfg .get ("robot_wire_gate_threshold",0.35 ))
        if cps :
            cps =wire_gate .apply (r ["V"],r ["F"],pb ,cps ,CE ,CT ,threshold =thr ,top_n =None )
        out [r ["pid"]]=np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
    return out 


def score (rows ,preds ):
    agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
    for r in rows :
        P =preds [r ["pid"]];G =r ["G"];Gd =r ["Gd"];tol =r ["tol"]
        hit =np .zeros (len (G ),bool );used =set ()
        if len (P )and len (G ):
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
                if d_ >tol or a_ in used or hit [b_ ]:
                    continue 
                hit [b_ ]=True ;used .add (a_ )
        tp =int (hit .sum ())
        agg [r ["regime"]][0 ]+=tp 
        agg [r ["regime"]][1 ]+=len (P )-tp 
        agg [r ["regime"]][2 ]+=len (G )-tp 
    res ={}
    for k ,(T ,Fp ,Fn )in agg .items ():
        p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
        res [k ]=2 *p *rc /max (p +rc ,1e-9 )
    res ["weighted"]=sum (W [k ]*res [k ]for k in W )
    return res 


def main ():
    import pickle ,glob 
    rows =[]
    for f in sorted (glob .glob (f"{CACHE }/*.npz")):
        pid =os .path .basename (f )[:-4 ]
        d =np .load (f ,allow_pickle =True )
        rows .append ({"pid":pid ,"V":d ["V"],"F":d ["F"],"probs":d ["probs"],
        "G":d ["G"],"Gd":d ["Gd"],"n":int (d ["n"]),"tol":float (d ["tol"]),
        "regime":"very"if int (d ["n"])>=8 else "low"})
    if not rows :
        print ("cache empty -- before pitstop2_gate_nested.py cikarimini kos");return 1 
    print (f"{len (rows )} part ({sum (r ['regime']=='low'for r in rows )} dusuk / "
    f"{sum (r ['regime']=='very'for r in rows )} very)\n",flush =True )

    off =score (rows ,derive (rows ,False ))
    on =score (rows ,derive (rows ,True ))
    print (f"{'':<14}{'low-CP':>10}{'very-CP':>10}{'agirlikli':>12}")
    print (f"{'direction KAPALI':<14}{off ['low']:>10.4f}{off ['very']:>10.4f}{off ['weighted']:>12.4f}")
    print (f"{'direction OPEN':<14}{on ['low']:>10.4f}{on ['very']:>10.4f}{on ['weighted']:>12.4f}")
    delta =on ["weighted"]-off ["weighted"]
    print (f"\nFARK: {delta :+.4f}")
    verdict ="GUVENLI"if delta >=-0.005 else "GERI AL"
    print (f"KAPI (loss <= 0.005): {verdict }")
    json .dump ({"off":off ,"ten":on ,"delta":delta ,"verdict":verdict },
    open ("results/ab_channel_axis.json","w"),indent =1 )
    print ("receipt -> results/ab_channel_axis.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
