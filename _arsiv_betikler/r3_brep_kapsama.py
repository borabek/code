# -*- coding: utf-8 -*-
"""R3: robot-hazir kaybinin ORTAK KOKU B-rep eslesmesi mi?

HIPOTEZ (R0+R1'den): CP'nin ekseni three kaynaktan gelebiliyor --
    (a) B-rep silindir ekseni  : ANALITIK, egimi full bilir      -> angle ~0
    (b) normal-kovaryans        : mesh'ten, egimi gorebilir
    (c) koordinat yuvarlamasi   : (a) and (b) susarsa kalan TOHUM -> egik agizda 90 dereceye up to error
R1'de tahminler full [1,0,0]/[0,1,0] output = (c). Ayni CP'lerde point da centroid'de kaliyor,
i.e. YANAL error da orada.

Yani single a kok neden two metrigi birden dusuruyor may be: B-REP ESLESMESININ TUTMAMASI.

Bu onemli, because kodda ASILI a is present (cp_openings.py icindeki not): B-rep kapilari
(BREP_MAX_OFF, BREP_R_MAX) radius YAY HATASI varken ayarlanmisti -- yaricaplar 3.5 fold small
cikiyordu. Hata 2026-07-31'de duzeltildi but kapilar YENIDEN TARANMADI ("P2 taramasi").

OLCULEN: eslesen each GT noktasi for adayin `brep_r` degeri (gate sutunu 13; 0 = silindir
eslesmedi) with lateral/angle basarisi arasindaki iliski.
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
BREP_R_SUT =13 # FEAT_NAMES: 0..12 baseline, 13 = brep_r


def main ():
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf (Xtr ),"clf_z":rf (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))

    SAT =[]
    for r in DER :
        if r ["X"]is None :
            continue 
        s =wire_gate .decision_score (m ,r ["X"])
        k =wire_gate .decision_mask (s )
        if not k .any ():
            continue 
        idx =np .where (k )[0 ]
        P =r ["P"][k ];Pd =r ["Pd"][k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
        pe_t =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *float (r ["diag"]))
        used ,hit =set (),set ()
        for d_ ,a_ ,b_ in sorted ((pe_t [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >tt or a_ in used or b_ in hit :
                continue 
            used .add (a_ );hit .add (b_ )
            SAT .append ({"brep_r":float (r ["X"][idx [a_ ],BREP_R_SUT ]),
            "lateral":float (pe [a_ ,b_ ]),"aci":float (an [a_ ,b_ ]),
            "pid":r ["pid"],"regime":"cok"if r ["n"]>=8 else "dusuk"})

    br =np .array ([x ["brep_r"]for x in SAT ])
    ya =np .array ([x ["lateral"]for x in SAT ])
    ac =np .array ([x ["aci"]for x in SAT ])
    var =br >0 
    print (f"{len (SAT )} eslesen GT noktasi | B-rep silindiri ESLESEN: {var .sum ()} "
    f"({var .mean ():.1%}) | eslesmeYEN: {(~var ).sum ()} ({(~var ).mean ():.1%})")

    print (f"\n{'grup':<22}{'n':>6}{'lateral med':>11}{'aci med':>10}{'lateral<=2':>10}"
    f"{'aci<=10':>9}{'IKISI':>8}")
    for ad ,i in (("B-rep ESLESTI",var ),("B-rep YOK",~var )):
        if not i .any ():
            continue 
        print (f"{ad :<22}{int (i .sum ()):>6}{np .median (ya [i ]):>11.2f}{np .median (ac [i ]):>10.2f}"
        f"{(ya [i ]<=2 ).mean ():>10.1%}{(ac [i ]<=10 ).mean ():>9.1%}"
        f"{((ya [i ]<=2 )&(ac [i ]<=10 )).mean ():>8.1%}")

    print ("\n90-DERECELIK POPULASYON B-rep'siz mi?")
    dik =ac >45 
    if dik .any ():
        print (f"  aci>45 olanlarin B-rep eslesme orani : {var [dik ].mean ():.1%}")
        print (f"  aci<=45 olanlarin                    : {var [~dik ].mean ():.1%}")

    print ("\nEGER B-rep kapsamasi %100 OLSAYDI (eslesen grubun oranlariyla):")
    if var .any ():
        y2 =(ya [var ]<=2 ).mean ();a2 =(ac [var ]<=10 ).mean ()
        i2 =((ya [var ]<=2 )&(ac [var ]<=10 )).mean ()
        su =((ya <=2 )&(ac <=10 )).mean ()
        print (f"  ikisi birden: {su :.1%} -> {i2 :.1%}  (+{(i2 -su )*100 :.1f} puan)")
        print (f"  robot-hazir 0.65 icin gereken: %87.4")

    print (f"\n{'regime':<10}{'n':>6}{'B-rep var':>11}{'IKISI OK':>10}")
    for rj in ("dusuk","cok"):
        i =np .array ([x ["regime"]==rj for x in SAT ])
        if i .any ():
            print (f"{rj :<10}{int (i .sum ()):>6}{var [i ].mean ():>11.1%}"
            f"{((ya [i ]<=2 )&(ac [i ]<=10 )).mean ():>10.1%}")

    with open ("results/r3_brep_kapsama.json","w",encoding ="utf-8")as f :
        json .dump ({"n":len (SAT ),"brep_kapsama":float (var .mean ()),
        "eslesen_ikisi_ok":float (((ya [var ]<=2 )&(ac [var ]<=10 )).mean ())if var .any ()else None ,
        "eslesmeyen_ikisi_ok":float (((ya [~var ]<=2 )&(ac [~var ]<=10 )).mean ())if (~var ).any ()else None ,
        "dik_brep_orani":float (var [dik ].mean ())if dik .any ()else None },f ,indent =1 )
    with open ("results/r3_satir.pkl","wb")as f :
        pickle .dump (SAT ,f )
    print ("\nmakbuz -> results/r3_brep_kapsama.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
