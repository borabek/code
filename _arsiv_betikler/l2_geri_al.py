# -*- coding: utf-8 -*-
"""L2'yi GERI AL: gate yeniden 13 ozellikle egitilir.

WHY: L2 (at most ezberleyen 3 ozelligi atmak) DEV kumesinde +0.0066/+0.0068 kazanmisti; but
that cluster L2'nin SECILDIGI kumeydi. Geometri as AYRIK VAL kumesinde same arm -0.0111/-0.0104
KAYBETTI (results/sinav_val.json). Onceden yazilan rule: "DEV'de kazanip VAL'de kaybeden arm
asiri-uydurmadir, geri alinir."

Ders: a feature "ezberliyor" damgasi yiyor diye atilamaz -- AUC dususu urun metrigi not.
13 ozellikli gate hem VAL tespitinde (0.6530 vs 0.6418) hem kesinlikte (0.695 vs 0.640) onde.

Eski model results/wire_gate.pkl.l2_2026_07_31 as saklanir (single adimda geri donulebilir).
"""
import os ,sys ,json ,pickle ,shutil 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
PKL ="results/wire_gate.pkl"
BAK ="results/wire_gate.pkl.l2_2026_07_31"


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    import wire_gate 

    old_ =pickle .load (open (PKL ,"rb"))
    if old_ .get ("cols")is None :
        print ("gate already 13 ozellikli, yapacak is absent");return 
    if not os .path .exists (BAK ):
        shutil .copy (PKL ,BAK )
        print (f"backup -> {BAK }")

    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"]
    print (f"training: {X .shape [0 ]} candidate x {X .shape [1 ]} ozellik "
    f"(eski gate {len (old_ ['cols'])} sutun kullaniyordu)")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X ,y )
    pickle .dump ({
    "clf":clf ,
    "feat_names":list (wire_gate .FEAT_NAMES_13 ),
    "cols":None ,
    "note":("L2 GERI ALINDI 2026-07-31: 13 ozellik. L2 (depth/size/aspect atma) DEV'de "
    "+0.0066 kazanmisti ama SECILDIGI kumeydi; ayrik VAL kumesinde -0.0111 tespit "
    "/ -0.0104 robot KAYBETTI (results/sinav_val.json). Onceden yazili kill kurali "
    "uygulandi. Eski model: results/wire_gate.pkl.l2_2026_07_31"),
    },open (PKL ,"wb"))
    print ("gate 13 ozellikle yeniden egitildi and yazildi")

    m =pickle .load (open (PKL ,"rb"))
    assert m ["cols"]is None and len (m ["feat_names"])==13 
    Xs =X [:5 ]
    assert m ["clf"].predict_proba (Xs ).shape ==(5 ,2 ),"13 sutunlu tahmin calismiyor"
    print ("dogrulama: cols=None, 13 name, prediction 13 sutunla calisiyor")


if __name__ =="__main__":
    main ()
