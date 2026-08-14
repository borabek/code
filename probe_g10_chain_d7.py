# -*- coding: utf-8 -*-
"""D7'de YIGIN KIYASI: kanonik (G7 + gate v6) vs A3-A4 (g10 + gate v7).

A3-A4 zinciri D6'da +0.0203 tespit / +0.0245 robot olculmustu but MANSET yoluna
(D7, brand-disi) HIC sokulmadi. Temsil kolu ([[temsil-kolu-calisti-ara-sonuc]])
gorulmemis ureticide +0.2396 iddia ediyor; this, that iddianin kanonik olcekte sinavi.

SIZINTI DENETIMI YAPILDI and TEMIZ: g10'un training dizini `_label_auto_obj_TR`
(2108 part) with D7 kesisimi PARCA duzeyinde 0, MARKA duzeyinde 0 (12 D7
markasinin none of them egitimde absent). Yani D7 g10 for de brand-disi a exam.

Ayni harness, only yigin degisir (KD7_* cevre degiskenleri). MIKRO. Tam zincir.
"""
import collections ,importlib ,json ,os ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
from sina_cluster import match_hungarian 

YIGINLAR =[
("KANONIK G7+v6",{"KD7_KAYIT":"results/_der_yeni_G7BIRLESIK.pkl",
"KD7_GATE":"results/wire_gate_v6.pkl",
"KD7_OB":"results/_p1_olasilik_d7"}),
("A3-A4 g10+v7",{"KD7_KAYIT":"results/_der_yeni_g10.pkl",
"KD7_GATE":"results/wire_gate_v7.pkl",
"KD7_OB":"results/_p1_olasilik_d7g10"}),
]
d7p =json .load (open ("results/d7_sinav_kumesi.json"))["pidler"]
out ={}
for ad ,env in YIGINLAR :
    os .environ .update (env )
    import canonical_d7 as K ;importlib .reload (K )
    assert K .KAYIT ==env ["KD7_KAYIT"]and K .GATE ==env ["KD7_GATE"]
    gate =K .gate_yukle ();S =K .step_map ();kay =K .yukle (d7p )
    rob ,tes ,oracle_ =collections .defaultdict (lambda :[0 ,0 ,0 ]),[],[]
    for pid ,r in kay .items ():
        G =np .asarray (r .get ("G",[]),float )
        if K .x58 (r )is None or not len (G ):
            continue 
        Gd =np .asarray (r ["Gd"],float );dg =r ["diag"]
        P ,D =K .urun_poz (r ,gate ,S ,tam_zincir =True )
        tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        a =rob [r ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,dg ,max (3.0 ,0.06 *dg ),
        180.0 ,True )[:3 ])
        # ADAY KAHINI: gate'ten ONCE pool GT'yi ne up to kapsiyor (temsil olcusu)
        Ph =np .asarray (r ["P"],float );Dh =np .asarray (r ["Pd"],float )
        oracle_ .append ((len (G ),)+match_hungarian (Ph ,Dh ,G ,Gd ,dg ,
        max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ])
    pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
    mi =float (2 *sum (a [0 ]for a in rob .values ())/
    max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
    out [ad ]={"robot":mi ,"tespit":K .mikro (tes ),"aday_kahini":K .mikro (oracle_ ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,"n":len (tes )}
    c =out [ad ]
    print (f"{ad :<16} robot {c ['robot']:.4f} | tespit {c ['tespit']:.4f} | "
    f"candidate kahini {c ['aday_kahini']:.4f} | makro {c ['makro']:.4f} | "
    f"en kotu {c ['en_kotu']:.4f} | n={c ['n']}",flush =True )

a ,b =out ["KANONIK G7+v6"],out ["A3-A4 g10+v7"]
art =sum (1 for m in b ["brand"]if b ["brand"][m ]>a ["brand"].get (m ,0 )+1e-9 )
print (f"\nFARK robot {b ['robot']-a ['robot']:+.4f} | tespit {b ['tespit']-a ['tespit']:+.4f} "
f"| kahin {b ['aday_kahini']-a ['aday_kahini']:+.4f} | artan brand {art }/{len (b ['brand'])}")
for m in sorted (a ["brand"],key =lambda k :-a ["brand"][k ]):
    print (f"  {m :<7} {a ['brand'][m ]:.4f} -> {b ['brand'].get (m ,float ('nan')):.4f}  "
    f"{b ['brand'].get (m ,0 )-a ['brand'][m ]:+.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,"artan_marka":art ,
"sizinti_denetimi":"g10 training dizini _label_auto_obj_TR with D7 "
"kesisimi: part 0, brand 0 (12/12 brand temiz)",
"not":"AYNI harness (canonical_d7), only yigin degisti. Tam zincir, "
"MIKRO, D7=DEV brand-disi."},
open ("results/g10_zinciri_d7.json","w"),indent =1 )
