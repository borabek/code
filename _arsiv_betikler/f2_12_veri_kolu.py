# -*- coding: utf-8 -*-
"""F2-12: VERI-ONLY CHECK -- corpus buyumesi and manufacturer cesitliligi F1'i oynatiyor mu?

ASIL KOL. Olculmus kazanci which is single mekanizma data
([[ogrenme-egrisi-fiyat-etiketi]] +0.033/ln(grup), [[cesitlilik-and-fn-profili]] +0.0443).

IKI GATE, TEK FARK VERI:
    TABAN : results/zengin_parite_v3_taban.npz   (only ESKI parts)
    v3    : results/zengin_parite_v3.npz         (old + YENI parts)
Ikisi de AYNI protocol filtresinden gecti, AYNI siniflandirici, AYNI hiperparametre,
AYNI seed. Bu sart: [[gate-refit-minv4]] dersi "two gate AYNI dagilimda egitilmeli".

URUNUN DECISION YOLU KULLANILIR: skor `wire_gate.decision_score`, maske `decision_mask`.
Olcum betiklerinin gate'i ELDE yeniden kurmasi 2026-08-01'de yakalanmis a hataydi --
urunun `apply()` yolu arada yonlendirme does and two path ayrisirsa olculen sey urunun
YAPTIGI sey olmaz. Bu yuzden model sozlugu dagitilanla AYNI alanlarla kurulur.

TEZ DEGISMEZ: network egitilmez, remesh does not change, `v_o` does not change, candidate ureticisi does not change.
"""
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import wire_gate 

TABAN ="results/zengin_parite_v3_taban.npz"
V3 ="results/zengin_parite_v3.npz"
MAKBUZ ="results/f2_12_veri_kolu.json"


def egit (npz ,seed =0 ):
    """Dagitilan gate with AYNI recete: RF 400 / leaf 3, 58 ham -> part-ici z-skor -> 116."""
    from sklearn .ensemble import RandomForestClassifier 
    d =np .load (npz ,allow_pickle =True )
    X =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
    pid =np .asarray (d ["pids"],str )
    # PARCA-ICI z-skor PARCA BAZINDA uygulanir (calisma aninda da oyle)
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ),float )
    for p in np .unique (pid ):
        m =pid ==p 
        Z [m ]=wire_gate .within_part (X [m ],"zskor")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =seed )
    clf .fit (Z ,d ["y"])
    model ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":"zskor","cols":None ,
    "feat_names":None }
    return model ,{"candidate":int (len (d ["y"])),"part":int (len (np .unique (pid ))),
    "manufacturer":int (len (set (map (str ,d ["mfg"])))),
    "pozitif":float (np .mean (d ["y"]))}


def olc (model ,DER ):
    """Tespit F1 -- urunun karar yolu + metrigin esleme kurali."""
    tp =fp =fn =0 
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]]).astype (float )
        if M .shape [1 ]*2 !=model ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (model ,M ))
        P =np .asarray (r ["P"],float )[k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        e =0 
        if len (P )and len (G ):
            d =P [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            up ,ug =set (),set ()
            for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
            for a in range (len (P ))for b in range (len (G ))):
                if dd >tol or a_ in up or b_ in ug :
                    continue 
                up .add (a_ );ug .add (b_ );e +=1 
        tp +=e ;fp +=len (P )-e ;fn +=len (G )-e 
        s =per [r ["mfg"]];s [0 ]+=e ;s [1 ]+=len (P )-e ;s [2 ]+=len (G )-e 
    f1 =2 *tp /max (2 *tp +fp +fn ,1 )
    pm ={m :2 *s [0 ]/max (2 *s [0 ]+s [1 ]+s [2 ],1 )for m ,s in per .items ()}
    return {"F1":f1 ,"TP":tp ,"FP":fp ,"FN":fn ,"manufacturer":pm }


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import measure_set as OK 
    DER ,_ =OK .cluster ("results/_der_tam.pkl")
    print (f"OLCUM: {len (DER )} part\n")

    S ={}
    for ad ,npz in (("TABAN (yalniz eski)",TABAN ),("v3 (eski + YENI)",V3 )):
        model ,bilgi =egit (npz )
        r =olc (model ,DER )
        S [ad ]={**r ,"corpus":bilgi }
        print (f"{ad :<22} corpus {bilgi ['part']:>5} part / {bilgi ['manufacturer']:>2} manufacturer "
        f"| TESPIT F1 {r ['F1']:.4f}  (TP {r ['TP']} FP {r ['FP']} FN {r ['FN']})")

    a ,b =S ["TABAN (yalniz eski)"],S ["v3 (eski + YENI)"]
    d =b ["F1"]-a ["F1"]
    print (f"\nFARK (v3 - baseline): {d :+.4f}")
    ort =set (a ["manufacturer"])&set (b ["manufacturer"])
    # URETICI KAPISI ICIN EN AZ 5 PARCA SARTI (2026-08-04'te ogrenildi):
    # first kosuda WAGO 0.80 -> 0.50 dustu and kapiyi DUSURDU. Olcum kumesinde WAGO'dan
    # **TEK part** present; single parcalik a kova noise produces, gerileme not. Kapiyi
    # tetikleyebilmek for ureticinin at least 5 parcasi must be; digerleri RAPORLANIR but
    # karar vermez.
    say =collections .Counter (r ["mfg"]for r in DER )
    print (f"\n{'manufacturer':<8}{'part':>7}{'baseline':>9}{'v3':>9}{'fark':>9}  {'gate':>6}")
    kotu =[]
    for m in sorted (ort ):
        f =b ["manufacturer"][m ]-a ["manufacturer"][m ]
        n =say .get (m ,0 )
        gate =n >=5 
        print (f"{m :<8}{n :>7}{a ['manufacturer'][m ]:>9.4f}{b ['manufacturer'][m ]:>9.4f}{f :>+9.4f}"
        f"  {'EVET'if gate else 'yok(n<5)':>6}")
        if f <-0.005 and gate :
            kotu .append (m )
    print (f"\nGO: havuzlanmis >= +0.015  -> {'GECTI'if d >=0.015 else 'gecmedi'}")
    print (f"    hicbir manufacturer < -0.005 -> {'GECTI'if not kotu else 'GECMEDI '+str (kotu )}")
    print ("\nNOT: bu ARA olcumdur -- turetme %62'de durdu, corpus henuz tam degil.")
    print ("     Ayrica measurement kumesi PXC+WEI agirlikli; gorulmemis manufacturer sorusu D5-4 ile yanitlanir.")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"baseline":a ,"v3":b ,"fark":d ,"kotulesen_uretici":kotu ,
        "ara_olcum":True },f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
