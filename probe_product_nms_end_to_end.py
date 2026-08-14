# -*- coding: utf-8 -*-
"""URUNUN TAM ZINCIRINDE NMS: D7 brand-disi, oncesi/sonrasi. MIKRO.

Onceki tarama gate maskesi + ham `v_o` uzerindeydi. Bu, poz kafasi dahil TAM
urun zinciriyle (`canonical_d7.urun_poz`, tam_zincir=True) olcer. Kol `WG_NMS_MM`
with acilip kapanir; same kosuda two times modul yeniden yuklenir.
"""
import collections ,importlib ,json ,os ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
from sina_cluster import match_hungarian 

d7p =json .load (open ("results/d7_exam_set.json"))["pidler"]
out ={}
RS =sys .argv [1 :]or ["0","6.0"]
for ad ,r_mm in [(f"NMS r={x }",x )for x in RS ]:
    os .environ ["WG_NMS_MM"]=r_mm 
    import wire_gate ;importlib .reload (wire_gate )
    import canonical_d7 as K ;importlib .reload (K )
    assert wire_gate .NMS_MM ==float (r_mm ),wire_gate .NMS_MM 
    gate =K .gate_yukle ();S =K .step_map ();kay =K .yukle (d7p )
    rob ,tes =collections .defaultdict (lambda :[0 ,0 ,0 ]),[]
    for pid ,r in kay .items ():
        G =np .asarray (r .get ("G",[]),float )
        if K .x58 (r )is None or not len (G ):
            continue 
        P ,D =K .urun_poz (r ,gate ,S ,tam_zincir =True )
        Gd =np .asarray (r ["Gd"],float );dg =r ["diag"]
        tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        a =rob [r ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,dg ,max (3.0 ,0.06 *dg ),
        180.0 ,True )[:3 ])
    pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
    mi =float (2 *sum (a [0 ]for a in rob .values ())/
    max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
    out [ad ]={"robot_mikro":mi ,"tespit_mikro":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,"n":len (tes )}
    c =out [ad ]
    print (f"{ad :<20} robot {mi :.4f} | detection {c ['tespit_mikro']:.4f} | "
    f"makro {c ['makro']:.4f} | en kotu {c ['en_kotu']:.4f}",flush =True )
a ,b =out [f"NMS r={RS [0 ]}"],out [f"NMS r={RS [-1 ]}"]
art =sum (1 for m in b ["brand"]if b ["brand"][m ]>a ["brand"][m ]+1e-9 )
print (f"\nFARK robot {b ['robot_mikro']-a ['robot_mikro']:+.4f} | "
f"detection {b ['tespit_mikro']-a ['tespit_mikro']:+.4f} | "
f"artan brand {art }/{len (b ['brand'])}")
json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,"artan_marka":art ,
"not":"TAM urun zinciri (poz kafasi dahil), D7=DEV, MIKRO."},
open (f"results/urun_nms_uctan_uca_{'_'.join (RS )}.json","w"),indent =1 )
