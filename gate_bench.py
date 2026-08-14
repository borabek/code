# -*- coding: utf-8 -*-
"""GATE TEZGAHI -- genellesme kollarinin ORTAK and TEK hakemi.

WHY: last five arm gate'e BILGI eklemeye calisti and besi de sifir verdi. Olculen sey this:
havuzlanmis 0.7584 but gorulmemis ureticide 0.5968/0.6680. Aradaki 0.13 bilgi eksikligi
not, EZBER. Bu tezgah ezbere saldiran kollari same olcutle yargilar.

TEK YARGIC KURALI: each arm
  (1) URUNUN karar yolundan gecer (wire_gate.decision_score + decision_mask) -- taklit YOK,
  (2) measurement kumesinin geometri gruplari egitimden CIKARILIR,
  (3) three bolmede birden raporlanir: havuzlanmis / WEI-disi / PXC-disi,
  (4) GRUP bootstrap with confidence araligi takes.

Kollar `kollar.py` not, cagiran betikten fonksiyon as gelir:
    arm(Xham, y, pid, mfg, keep) -> (Z_egitim, y_egitim, donustur_fn, ek_bilgi)
"""
import collections 
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

W ={"dusuk":0.895 ,"cok":0.105 }


def yukle ():
    """Olcum kumesi + training ozellikleri + manufacturer kodlari. TEK source."""
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    NPZ =cfg ["current_product"]["wire_gate"]["egitim_verisi"]
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    zen =np .load (NPZ ,allow_pickle =True )
    X =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    y =np .asarray (zen ["y"])
    pid =np .array ([str (x )for x in zen ["pids"]])
    mk =np .array ([str (x )for x in zen ["mfg"]])
    keep =~np .isin (np .array ([gk .get (p ,"yok:"+p )for p in pid ]),list (tg ))
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in pid [mk ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (mk )}
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    return {"cfg":cfg ,"npz":NPZ ,"DER":DER ,"rap":rap ,"gk":gk ,
    "X":X ,"y":y ,"pid":pid ,"mfg":mk ,"keep":keep ,"kod":kod ,
    "donusum":dag .get ("donusum"),"ad":wire_gate .FEAT_NAMES }


def puanla (model ,alt ,cfg ):
    """URUNUN karar yolu + correction zinciri. headline.py with AYNI order."""
    import wire_gate 
    from sina_cluster import esle 
    det ,rob =[],[]
    for r in alt :
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            Xr =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (model ,Xr ))
            if k .any ():
                P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                if cfg .get ("robot_pose_head"):
                    c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    c =wire_gate .pose_correct (Xr [k ],c )
                    if cfg .get ("robot_aci_secici"):
                        c =wire_gate .angle_correct (Xr [k ],c )
                    if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                        c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                    P =np .array ([x ["point"]for x in c ],float )
                    Pd =np .array ([x ["direction"]for x in c ],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
    return det ,rob 


def f1w (det ):
    A ={}
    for rj ,tp ,fp ,fn in det :
        a =A .setdefault (rj ,[0 ,0 ,0 ])
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
    s =0.0 
    for k ,(tp ,fp ,fn )in A .items ():
        if k not in W :
            continue 
        p =tp /max (tp +fp ,1e-9 );rr =tp /max (tp +fn ,1e-9 )
        s +=W [k ]*(2 *p *rr /max (p +rr ,1e-9 ))
    return s 


def calistir (D ,kol_fn ,ad ,tohumlar =(0 ,),ayrinti =True ):
    """Bir kolu UC bolmede olc. kol_fn(X, y, pid, mfg, keep, seed) -> model dict."""
    import measure_set 
    SON ,PARCA ={},{}
    bolmeler =[("havuzlanmis",None ,D ["DER"])]
    for k ,mad in D ["kod"].items ():
        alt =[x for x in D ["DER"]if x ["mfg"]==mad ]
        if len (alt )>=10 :
            bolmeler .append ((mad +"-disi",k ,alt ))
    for b ,mk ,alt in bolmeler :
        kp =D ["keep"].copy ()
        if mk is not None :
            kp =kp &(D ["mfg"]!=mk )
        v =[]
        for th in tohumlar :
            m =kol_fn (D ["X"],D ["y"],D ["pid"],D ["mfg"],kp ,th )
            det ,rob =puanla (m ,alt ,D ["cfg"])
            v .append ((f1w (det ),f1w (rob )))
            if th ==tohumlar [0 ]:
                PARCA [b ]=(det ,rob ,[x ["geo"]for x in alt ])
        SON [b ]={"tespit":float (np .mean ([x [0 ]for x in v ])),
        "robot":float (np .mean ([x [1 ]for x in v ])),
        "tohumlar":[float (x [0 ])for x in v ]}
    ud =[b for b in SON if b .endswith ("-disi")]
    SON ["_URETICI_DISI_ORT"]=float (np .mean ([SON [b ]["tespit"]for b in ud ]))if ud else 0.0 
    if ayrinti :
        print (f"{ad :<26}"+"".join (f"{SON [b ]['tespit']:>13.4f}"for b ,_ ,_ in bolmeler )
        +f"{SON ['_URETICI_DISI_ORT']:>10.4f}{SON ['havuzlanmis']['robot']:>9.4f}")
    return SON ,PARCA 


def ga (PARCA_A ,PARCA_B ,split ,n =2000 ):
    """Iki arm arasindaki farkin GRUP bootstrap confidence araligi."""
    import measure_set 
    da ,_ ,g =PARCA_A [split ];db ,_ ,_ =PARCA_B [split ]
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (da ,db )),g ,fn ,n =n )
    return lo ,hi 


def baslik (D ):
    b =["havuzlanmis"]+[m +"-disi"for k ,m in sorted (D ["kod"].items ())
    if len ([x for x in D ["DER"]if x ["mfg"]==m ])>=10 ]
    print (f"\n{'arm':<26}"+"".join (f"{x :>13}"for x in b )
    +f"{'URET-ORT':>10}{'robot':>9}")
    return b 
