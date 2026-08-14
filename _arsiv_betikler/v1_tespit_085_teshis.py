# -*- coding: utf-8 -*-
"""V1: TESPIT 0.85 OPERASYONU -- acilis teshisi (nerede duruyoruz, ceiling nerede).

Hedef 0.85. Su an 0.7538. Fark +0.096.

Bu betik operasyonun ILK adimi: hedefin hangi rejimden gelmesi GEREKTIGINI and that rejimlerin
TAVANLARININ nerede oldugunu olcer. Plan yapmadan before aritmetigi masaya koymak, this
arastirmada defalarca zaman kazandirdi.

OLCULEN (regime basina: low-CP %89.5 / very-CP %10.5):
    1. SU ANKI F1
    2. KAHIN GATE F1        -- mevcut adaylardan mukemmel secim
    3. ADAY RECALL          -- GT'nin yuzde kaci a adayla ortuluyor
    4. ADAY F1 TAVANI       -- 2R/(1+R), i.e. FP=0 varsayimiyla ulasilabilecek most high

and 0.85'in HANGI regime kombinasyonlariyla mumkun oldugunu cozer.
"""
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


def esle_ayrinti (P ,Pd ,G ,Gd ,diag ):
    """tespit modu eslesmesi -> (tp, fp, fn, secilen_aday_indisleri)."""
    hit =np .zeros (len (G ),bool );used =set ();sec =[]
    if len (P )and len (G ):
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *diag )
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >tt or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ );sec .append (a_ )
    tp =int (hit .sum ())
    return tp ,len (P )-tp ,len (G )-tp ,sec 


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import W ,esle ,f1_rejim ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"yok:"+p )for p in tpid ]),list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":DON }
    print ("gate hazir\n",flush =True )

    su ,kahin ,ceiling =[],[],[]
    rec ={"dusuk":[0 ,0 ],"cok":[0 ,0 ]}
    for r in DER :
        rj ="cok"if r ["n"]>=8 else "dusuk"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        Pt =r ["P"]if r ["X"]is not None else np .zeros ((0 ,3 ))
        Pdt =r ["Pd"]if r ["X"]is not None else np .zeros ((0 ,3 ))
        # SU AN
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            X58 =np .hstack ([r ["X"],r ["XR"]])
            s =wire_gate .decision_score (gate ,X58 );k =wire_gate .decision_mask (s )
            if k .any ():
                P =r ["P"][k ];Pd =r ["Pd"][k ]
        su .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        # KAHIN GATE: GT with eslesen adaylarin TAMAMI, baskasi absent
        tp ,fp ,fn ,sec =esle_ayrinti (Pt ,Pdt ,G ,Gd ,float (r ["diag"]))
        m =np .zeros (len (Pt ),bool )
        for a_ in sec :
            m [a_ ]=True 
        kahin .append ((rj ,)+esle (Pt [m ],Pdt [m ],G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        # ADAY TAVANI: each eslesen GT TP, kalani FN, FP=0
        ceiling .append ((rj ,tp ,0 ,len (G )-tp ))
        rec [rj ][0 ]+=tp ;rec [rj ][1 ]+=len (G )

    print (f"{'olcu':<24}{'dusuk-CP':>10}{'cok-CP':>10}{'AGIRLIKLI':>12}")
    S ={}
    for ad ,rows in (("1 SU AN",su ),("2 KAHIN GATE",kahin ),("4 ADAY TAVANI",ceiling )):
        rr =f1_rejim (rows )
        S [ad ]=rr 
        print (f"{ad :<24}{rr ['F1']['dusuk']:>10.4f}{rr ['F1']['cok']:>10.4f}"
        f"{rr ['agirlikli_F1']:>12.4f}")
    R ={k :rec [k ][0 ]/max (rec [k ][1 ],1 )for k in rec }
    print (f"{'3 ADAY RECALL':<24}{R ['dusuk']:>10.4f}{R ['cok']:>10.4f}"
    f"{sum (W [k ]*R [k ]for k in W ):>12.4f}")

    print (f"\n=== 0.85 NEREDEN GELMELI ===")
    d0 ,c0 =S ["1 SU AN"]["F1"]["dusuk"],S ["1 SU AN"]["F1"]["cok"]
    dT ,cT =S ["4 ADAY TAVANI"]["F1"]["dusuk"],S ["4 ADAY TAVANI"]["F1"]["cok"]
    print (f"  su an   : dusuk {d0 :.4f} | cok {c0 :.4f} -> agirlikli {S ['1 SU AN']['agirlikli_F1']:.4f}")
    print (f"  TAVAN   : dusuk {dT :.4f} | cok {cT :.4f} -> agirlikli "
    f"{W ['dusuk']*dT +W ['cok']*cT :.4f}")
    # very-CP tavanda olsa, low-CP ne must be?
    ger_d =(0.85 -W ["cok"]*cT )/W ["dusuk"]
    print (f"\n  cok-CP TAVANINDA ({cT :.4f}) olsaydi, dusuk-CP {ger_d :.4f} olmali")
    print (f"    -> dusuk-CP tavani {dT :.4f} | {'MUMKUN'if ger_d <=dT else 'IMKANSIZ'}")
    ger_c =(0.85 -W ["dusuk"]*dT )/W ["cok"]
    print (f"  dusuk-CP TAVANINDA ({dT :.4f}) olsaydi, cok-CP {ger_c :.4f} olmali")
    print (f"    -> cok-CP tavani {cT :.4f} | {'MUMKUN'if ger_c <=cT else 'IMKANSIZ'}")
    ust =W ["dusuk"]*dT +W ["cok"]*cT 
    print (f"\n  MUTLAK UST SINIR (iki regime de kendi ADAY tavaninda): {ust :.4f}")
    print (f"  0.85 {'BU SINIRIN ALTINDA -> yapisal olarak mumkun'if ust >=0.85 else 'BU SINIRIN USTUNDE -> ADAY URETIMI degismeden IMKANSIZ'}")
    print (f"\n  KAHIN GATE (mevcut adaylardan mukemmel secim): "
    f"{S ['2 KAHIN GATE']['agirlikli_F1']:.4f}")
    print (f"    -> 0.85 icin kahin gate'in {0.85 /S ['2 KAHIN GATE']['agirlikli_F1']:.0%}'i gerekli")
    with io .open ("results/v1_tespit_085.json","w",encoding ="utf-8")as f :
        json .dump ({"su_an":{"dusuk":d0 ,"cok":c0 ,
        "agirlikli":S ["1 SU AN"]["agirlikli_F1"]},
        "kahin_gate":S ["2 KAHIN GATE"]["agirlikli_F1"],
        "aday_recall":R ,"aday_tavani":{"dusuk":dT ,"cok":cT ,"agirlikli":ust },
        "085_mumkun":bool (ust >=0.85 )},f ,indent =1 )
    print ("receipt -> results/v1_tespit_085.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
