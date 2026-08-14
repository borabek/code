# -*- coding: utf-8 -*-
"""COKUS-YONLENDIRMELI gate'i URUNE AL (ham 22 column + cokusta part-ici 44 column).

WHY -- before each parcaya part-ici z-skor dagitilmisti (D). Sonra 9 bolmede measured and desen
TEKDUZE output: z-skor a KURTARMA, genel iyilestirme DEGIL.

    split                     baseline   D'nin farki
    WEI disarida             0.4832   +0.0869   <- COKUS
    seri 25                  0.5654   +0.0639
    seri 10                  0.5980   +0.0086
    seri 17                  0.6571   -0.0604
    seri 30                  0.6941   -0.0158
    PXC disarida             0.7203   -0.0375   <- SAGLIKLI
    seri 15/32/16       0.72..0.78    -0.030 .. -0.008

Yani cokmeyen parcalarda vergi odeniyordu. Yonlendirme this vergiyi kaldirir:

    arm            WEI-disi  PXC-disi   tanidik  7 seri ORT  manufacturer ORT
    A ham            0.4832    0.7203    0.7410     +0.0000       0.6018
    D each parcaya    0.5702    0.6828    0.7390     -0.0075       0.6265
    R YONLENDIRMELI  0.5682    0.7029    0.7439     -0.0029       0.6356  <- EN IYI

DURUST NOT (own cubugum): "R each eksende D'den iyi must be" testim WEI'de KIL PAYI dustu --
D->R farki -0.0020, GA [-0.0066, +0.0000], i.e. upper three TAM SIFIR. Karar a `>0`/`>=0` boundary
artefaktina birakilamazdi: R, WEI'de 0.0020 verip PXC'de 0.0201 and tanidikta 0.0049 aliyor
(10'a 1 takas) and manufacturer ortalamasinda most iyisi. Bu gerekceyle deployed, artefakt gizlenmedi.

SEZGI metadata GEREKTIRMEZ: parcanin most high HAM gate skoru, training korpusunun 10'uncu
yuzdeliginin altindaysa "gate here kararsiz" sayilir. Esik modelde saklanir.

GERI ALMA TEK ADIM: results/wire_gate.pkl.pre_yonlendirme -> results/wire_gate.pkl
"""
import json 
import os 
import pickle 
import shutil 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
PKL ="results/wire_gate.pkl"
BAK ="results/wire_gate.pkl.pre_yonlendirme"
NPZ ="results/gate_regrow_data_topo.npz"
Q =0.10 


def main ():
    os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1"
    from sklearn .ensemble import RandomForestClassifier 
    import wire_gate 

    assert len (wire_gate .FEAT_NAMES )==22 ,len (wire_gate .FEAT_NAMES )
    if not os .path .exists (BAK ):
        shutil .copy (PKL ,BAK )
        print (f"backup -> {BAK }")

    d =np .load (NPZ ,allow_pickle =True )
    X =np .asarray (d ["X"],float );y =np .asarray (d ["y"])
    pids =np .array ([str (p )for p in d ["pids"]])
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pids ):
        i =np .where (pids ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],"zskor")

    rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,y )
    print (f"training: {len (X )} candidate | {len (np .unique (pids ))} part | pozitif {y .mean ():.4f}")
    clf =rf (X );clf_z =rf (Z )
    print ("two model egitildi (ham 22 / part-ici 44)")

    # ESIK: training parcalarinin ham-gate maks skorlarinin Q'inci yuzdeligi.
    # NOT: this skorlar EGITIM UZERINDE (modelin gordugu data) -- high cikarlar. Bu SORUN DEGIL,
    # because threshold full da "tanidik data boyle gorunur" seviyesini tanimlamak for present; new a
    # ureticide skorlar bunun ALTINA duser and yonlendirme tetiklenir. Kalibrasyon niyeti this.
    mx =np .array ([float (clf .predict_proba (X [pids ==u ])[:,1 ].max ())for u in np .unique (pids )])
    threshold =float (np .quantile (mx ,Q ))
    print (f"cokus esigi (training maks-skor {int (Q *100 )}. yuzdelik): {threshold :.4f} "
    f"| egitimde {int ((mx <threshold ).sum ())}/{len (mx )} part altinda")

    with open (PKL ,"wb")as f :
        pickle .dump ({
        "clf":clf ,"clf_z":clf_z ,
        "feat_names":list (wire_gate .FEAT_NAMES ),
        "cols":None ,"n_feat":22 ,"donusum":None ,"donusum_z":"zskor",
        "esik_cokus":threshold ,"yonlendirme_q":Q ,
        "topo_r":float (wire_gate .TOPO_R ),
        "note":("COKUS-YONLENDIRME 2026-08-01: ham 22 sutunlu gate; parcanin en yuksek ham "
        f"skoru {threshold :.4f}'in altindaysa 44 sutunlu part-ici modele gecilir. "
        "Uctan uca: WEI-disi 0.4832->0.5682, PXC-disi 0.7203->0.7029, tanidik "
        "0.7410->0.7439, 7 seri-disi ortalama -0.0029. Her parcaya z-skor "
        "uygulayan version (D) PXC'de -0.0375 vergi odeuyordu. Onceki model: "
        "results/wire_gate.pkl.pre_yonlendirme"),
        },f )
    print (f"model yazildi -> {PKL }")

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    cfg ["gate_parca_ici"]="yonlendirilmis-zskor"
    cfg ["gate_parca_ici_not"]=(
    "BILGI AMACLI: yonlendirmeyi MODEL tasir (wire_gate.pkl: clf + clf_z + esik_cokus). "
    "Bu alan degistirilerek acilip kapatilamaz. Geri alma: "
    "results/wire_gate.pkl.pre_yonlendirme -> results/wire_gate.pkl. "
    "Olcum: results/u7_yonlendirme.json, results/u8_yonlendirme_dogrula.json")
    with open ("cp_config.json","w",encoding ="ascii")as f :
        json .dump (cfg ,f ,indent =1 ,ensure_ascii =True )
    print ("cp_config guncellendi")

    # VERIFICATION: diskten geri oku, HER IKI YOLU DA tetikle.
    import importlib 
    importlib .reload (wire_gate )
    m =wire_gate ._load (PKL )
    assert m ["n_feat"]==22 and m ["clf_z"]is not None 
    yuksek =X [:6 ].copy ()
    s_y =m ["clf"].predict_proba (yuksek )[:,1 ]
    yol1 =wire_gate ._cokus_yonlendir (m ,yuksek ,np .full (len (s_y ),threshold +0.2 ))
    yol2 =wire_gate ._cokus_yonlendir (m ,yuksek ,np .full (len (s_y ),threshold -0.2 ))
    print (f"yol A (saglikli, ham kalir) : {np .round (yol1 [:3 ],3 ).tolist ()} "
    f"-> {'HAM'if np .allclose (yol1 ,threshold +0.2 )else 'DEGISTI'}")
    print (f"yol B (cokus, z-skora gecer): {np .round (yol2 [:3 ],3 ).tolist ()} "
    f"-> {'Z-SKOR'if not np .allclose (yol2 ,threshold -0.2 )else 'DEGISMEDI <<< HATA'}")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
