# -*- coding: utf-8 -*-
"""22-sutunlu gate'i (13 + 5 B-rep fiziksel + 4 ICBUKEY TOPOLOJI) URUNE AL.

RATIONALE -- UC EKSENDE BIRDEN gecti (results/t15_topo_end_to_end.json):
    measurement                 18 column   22 column    difference
    familiar (DEV+VAL)      0.7287     0.7410    +0.0123
    PXC disarida (91)      0.7048     0.7203    +0.0155
    WEI disarida (108)     0.4736     0.4832    +0.0096
Onceden yazili kill: familiar >= -0.01 VE unseen manufacturer EN KOTU artmali -> GECTI.

Bu night a arm ILK KEZ hem familiar hem HER IKI manufacturer-disi bolmede kazandi. Beklenendi:
topoloji SAF GEOMETRI -- a hole each ureticide deliktir, istatistik not.

Aday duzeyi (t13): kon_cevre AUC 0.709, TP medyan 1.000 = TAM TUR icbukey halka (FP 0.833).
Kapsama: sutunlarin %93.7'si full (renk blogunun %33'une karsilik) -- MESH tabanli oldugu for
B-rep/renk yollarindaki kapsama kaybi YOK.

GERI ALMA TEK ADIM: results/wire_gate.pkl.pre_topo -> results/wire_gate.pkl and
cp_config.gate_topo_feats = false.
"""
import os ,sys ,json ,shutil ,pickle 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
PKL ="results/wire_gate.pkl"
BAK ="results/wire_gate.pkl.pre_topo"
NPZ ="results/gate_regrow_data_topo.npz"


def main ():
    os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1"
    from sklearn .ensemble import RandomForestClassifier 
    import wire_gate 

    assert len (wire_gate .FEAT_NAMES )==22 ,len (wire_gate .FEAT_NAMES )
    if not os .path .exists (BAK ):
        shutil .copy (PKL ,BAK )
        print (f"backup -> {BAK }")
    d =np .load (NPZ ,allow_pickle =True )
    X ,y =d ["X"],d ["y"]
    assert X .shape [1 ]==22 ,X .shape 
    print (f"training: {X .shape [0 ]} candidate x 22 sutun | pozitif {y .mean ():.4f}")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X ,y )
    pickle .dump ({
    "clf":clf ,"feat_names":list (wire_gate .FEAT_NAMES ),"cols":None ,"n_feat":22 ,
    "topo_r":float (wire_gate .TOPO_R ),# see. wire_gate._dogrula_uyum

    "note":("TOPO 2026-08-01: 13 + 5 B-rep fiziksel + 4 icbukey topoloji. Uctan uca UC "
    "eksende birden kazandi: familiar +0.0123, PXC-disi +0.0155, WEI-disi +0.0096. "
    "kon_cevre = icbukey kenarlarin axis etrafindaki acisal kapsamasi; real "
    "acikliklarda TAM TUR (medyan 1.000), yanlislarda 0.833. Onceki model: "
    "results/wire_gate.pkl.pre_topo"),
    },open (PKL ,"wb"))
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    cfg ["gate_topo_feats"]=True 
    cfg ["gate_topo_not"]=("wire_gate.USE_TOPO_FEATS this bayrakla OPEN must be; kapatirsan "
    "results/wire_gate.pkl.pre_topo'yu geri koy (18 column).")
    json .dump (cfg ,open ("cp_config.json","w",encoding ="ascii"),indent =1 ,ensure_ascii =True )
    print ("cp_config.gate_topo_feats = true")

    import importlib 
    importlib .reload (wire_gate )
    m =pickle .load (open (PKL ,"rb"))
    assert m ["n_feat"]==22 and len (wire_gate .FEAT_NAMES )==22 ,"uretim/gate uyusmuyor"
    assert m ["clf"].predict_proba (X [:5 ]).shape ==(5 ,2 )
    print (f"dogrulama: uretim {len (wire_gate .FEAT_NAMES )} sutun = gate {m ['n_feat']} sutun, tahmin calisiyor")


if __name__ =="__main__":
    main ()
