# -*- coding: utf-8 -*-
"""18-sutunlu (B-rep FIZIKSEL ozellikli) gate'i URUNE AL.

RATIONALE -- uclu split kuralini gecti (`results/q6_fiz_uctan_uca_{dev,val}.json`):

    arm                     DEV tespit   VAL tespit
    rt2 13 (dagitilan)        0.7089       0.6417
    guncel hat, 13 column      0.6938       0.6477
    guncel hat, 18 column      0.7567       0.7326

    OZELLIK etkisi  tespit  DEV +0.0628 [+0.0228,+0.1061]  VAL +0.0854 [+0.0406,+0.1306]  IKISI DE BELIRGIN
    TOPLAM          tespit  DEV +0.0481 [+0.0083,+0.0915]  VAL +0.0911 [+0.0523,+0.1324]  IKISI DE BELIRGIN
    TOPLAM          robot   DEV +0.0501 [+0.0085,+0.0974]  VAL +0.0437 [+0.0088,+0.0815]  IKISI DE BELIRGIN

VERI etkisi (fiz13-rt2) DEV -0.0147 / VAL +0.0057 -- ZIT ISARET, two GA da sifiri iceriyor
-> "guncel hat kaybettiriyor" hipotezi COKTU, kazanc ozelliklerden geliyor.

Karar kumesinde kazandi, AYRIK exam kumesinde dogrulandi, four measurement de belirgin. Bu gece olen
five kolun (J, L2, gate v6, 5. uye, oy olcegi) none of them this esikten gecmemisti.

GERI ALMA TEK ADIM: results/wire_gate.pkl.pre_fiz -> results/wire_gate.pkl and
cp_config.gate_fiz_feats = false.
"""
import os ,sys ,json ,shutil ,pickle 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
PKL ="results/wire_gate.pkl"
BAK ="results/wire_gate.pkl.pre_fiz"
NPZ ="results/gate_regrow_data_fiz.npz"


def main ():
    os .environ ["WG_FIZ_FEATS"]="1"
    from sklearn .ensemble import RandomForestClassifier 
    import wire_gate 

    assert len (wire_gate .FEAT_NAMES )==18 ,len (wire_gate .FEAT_NAMES )
    if not os .path .exists (BAK ):
        shutil .copy (PKL ,BAK )
        print (f"backup -> {BAK }")

    d =np .load (NPZ ,allow_pickle =True )
    X ,y =d ["X"],d ["y"]
    assert X .shape [1 ]==18 ,X .shape 
    print (f"training: {X .shape [0 ]} candidate x 18 sutun | pozitif {y .mean ():.4f}")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X ,y )
    pickle .dump ({
    "clf":clf ,
    "feat_names":list (wire_gate .FEAT_NAMES ),
    "cols":None ,
    "n_feat":18 ,
    "note":("FIZ 2026-07-31: 13 + 5 B-rep FIZIKSEL feature (brep_r, esesenli, r_orani, "
    "bos_derinlik, gecen). Uctan uca DEV +0.0481 / VAL +0.0911 tespit, ikisi de "
    "BELIRGIN; feature etkisi single basina DEV +0.0628 / VAL +0.0854. brep_r with "
    "mevcut size korelasyonu -0.009 = kopya not, new bilgi. Bu ozellikler "
    "brep_axes._fit_circle duzeltmesine BAGIMLI (radius oncesinde 3.5 fold "
    "kucuktu). Onceki model: results/wire_gate.pkl.pre_fiz"),
    },open (PKL ,"wb"))
    print ("gate 18 sutunla egitildi and yazildi")

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    cfg ["gate_fiz_feats"]=True 
    cfg ["gate_fiz_feats_not"]=("wire_gate.USE_FIZ_FEATS this bayrakla OPEN must be; kapatirsan "
    "results/wire_gate.pkl.pre_fiz'i geri koy (13 column).")
    json .dump (cfg ,open ("cp_config.json","w",encoding ="ascii"),indent =1 ,ensure_ascii =True )
    print ("cp_config.gate_fiz_feats = true")

    m =pickle .load (open (PKL ,"rb"))
    assert m ["n_feat"]==18 and len (m ["feat_names"])==18 
    assert m ["clf"].predict_proba (X [:5 ]).shape ==(5 ,2 )
    print ("dogrulama: 18 column, prediction calisiyor")


if __name__ =="__main__":
    main ()
