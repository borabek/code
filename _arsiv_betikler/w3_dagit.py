# -*- coding: utf-8 -*-
"""W3: YENI WEI VERISINI URUNE DAGIT (w2'de OLCULEN kolun aynisi).

W2 SONUCU: prospektif (109 gorulmemis part) 0.6510 -> 0.6901, +0.0391,
GA [+0.0203, +0.0676] = GERCEK, 3 tohumun ucunde de same direction. PXC -0.0037 = noise.

NE DAGITILIYOR: corpus + new WEI'nin YALNIZ EGITIM YARISI (110 part / 1343 candidate).
    Prospektif test yarisi (109 part) EGITIME GIRMEZ -- harcanmamis durust a test
    kumesi as kalir. LOCKED only 95 part; ikinci a temiz test degerlidir.
    Dagitilan sey OLCULEN seyin AYNISI becomes; this rule bugun three times ihlal edilip duzeltildi.

DEGISEN TEK SEY: training korpusunun buyuklugu. Ozellikler, column count (116), donusum
(part-ici z-skor), RF ayarlari, candidate ureticisi, threshold -- all of them AYNI.
"""
import hashlib 
import io 
import json 
import os 
import pickle 
import shutil 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ESKI ="results/zengin_parite.npz"
YENI ="results/yeni_wei.npz"
BIRLESIK ="results/zengin_parite_w2.npz"
GATE ="results/wire_gate.pkl"
EK =(" | W2 2026-08-02: +110 yeni WEI parcasi (1343 candidate); "
"prospektif 109 part EGITIME GIRMEDI")


def egit_ve_yaz (npz_yolu ,not_ek ):
    """Verilen training npz'sinden URUN GATE'INI kur. Ileri and GERI path AYNI koddur."""
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    d =np .load (npz_yolu ,allow_pickle =True )
    XF =np .hstack ([np .asarray (d ["X22"],float ),np .asarray (d ["XR"],float )])
    y =np .asarray (d ["y"]);pid =np .array ([str (x )for x in d ["pids"]])
    eski =pickle .load (open (GATE ,"rb"))
    Z =np .zeros ((len (XF ),XF .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (XF [i ],eski .get ("donusum"))
    assert Z .shape [1 ]==eski ["n_feat"],(Z .shape ,eski ["n_feat"])
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z ,y )
    yd =dict (eski );yd ["clf"]=clf 
    n =eski .get ("note","")
    while EK in n :
        n =n .replace (EK ,"")
    yd ["note"]=n +not_ek 
    pickle .dump (yd ,open (GATE ,"wb"))
    return hashlib .md5 (open (GATE ,"rb").read ()).hexdigest (),len (y ),len (np .unique (pid ))


def geri ():
    """GERI DONUS: W2 oncesi gate'i ESKI npz'den yeniden kur.

    WHY BYTE KOPYASI DEGIL: first kosuda betik tekrar-calistirmaya dayanikli degildi and
    yedegin uzerine ZATEN-DAGITILMIS gate'i yazdi; W2 oncesi file (MD5 8e2931f0) kayboldu.
    Yeniden uretim byte-birebir CIKMADI (RF pickle'i full yinelenmiyor), that is why geri donus
    residual DOSYA not TARIF: same data (zengin_parite.npz), same sutunlar, same RF ayarlari.
    Islevsel as W2 oncesi gate budur; makbuzdaki numbers this tarifle uretilmisti.
    """
    md5 ,na ,np_ =egit_ve_yaz (ESKI ,"")
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    wg =cfg .setdefault ("current_product",{}).setdefault ("wire_gate",{})
    wg ["egitim_verisi"]=ESKI ;wg ["md5"]=md5 ;wg ["yol"]=GATE 
    wg .pop ("gate_kimlik",None )
    with io .open ("cp_config.json","w",encoding ="utf-8")as f :
        json .dump (cfg ,f ,indent =1 ,ensure_ascii =False )
    print (f"GERI ALINDI -> {ESKI } | {na } candidate / {np_ } part | MD5 {md5 [:12 ]}")
    print ("  headline.py --yaz ile mansetı yeniden uretin.")


def main ():
    if "--geri"in sys .argv :
        return geri ()
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 

    zen =np .load (ESKI ,allow_pickle =True )
    X0 =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    yw =np .load (YENI ,allow_pickle =True )
    X1 =np .asarray (yw ["X22"],float )
    if X1 .shape [1 ]==22 :
        X1 =np .hstack ([X1 ,np .asarray (yw ["XR"],float )])
    assert X1 .shape [1 ]==X0 .shape [1 ]==58 ,(X0 .shape ,X1 .shape )
    eg =np .array ([str (x )for x in yw ["split"]])=="training"

    # --- BIRLESIK EGITIM VERISI (old npz'nin BICIMINI birebir korur: X22=22, XR=36)
    X22 =np .vstack ([np .asarray (zen ["X22"],float ),X1 [eg ][:,:22 ]])
    XR =np .vstack ([np .asarray (zen ["XR"],float ),X1 [eg ][:,22 :]])
    y =np .concatenate ([np .asarray (zen ["y"]),np .asarray (yw ["y"])[eg ]])
    pids =np .concatenate ([np .array ([str (x )for x in zen ["pids"]]),
    np .array ([str (x )for x in yw ["pids"]])[eg ]])
    mfgk =np .unique ([str (x )for x in zen ["mfg"]])
    # new parcalarin all of them WEI: korpusta WEI'yi temsil eden kodu bul (cogunluk esleme)
    import collections 
    from big_arbiter import eligible 
    mo ={p :m for m ,p ,jf ,s in eligible ()}
    m0 =np .array ([str (x )for x in zen ["mfg"]])
    kod ={k :collections .Counter (mo .get (p ,"?")for p in np .array (
    [str (x )for x in zen ["pids"]])[m0 ==k ]).most_common (1 )[0 ][0 ]for k in mfgk }
    wei_kod =[k for k ,v in kod .items ()if v =="WEI"]
    assert len (wei_kod )==1 ,f"WEI kodu belirsiz: {kod }"
    mfg =np .concatenate ([m0 ,np .array ([wei_kod [0 ]]*int (eg .sum ()))])
    # `votes` and `zengin_ad` da TASINMALI: parite testi training npz'sindeki votes tavanini
    # denetliyor and first yazimda this dizileri dusurdugum for test KIRMIZI yandi (correct davranis).
    # votes, X22'nin 11. sutunudur (FEAT_NAMES.index('votes')==11, old npz with birebir dogrulandi).
    import wire_gate as _wg 
    vi =_wg .FEAT_NAMES .index ("votes")
    v_eski =np .asarray (zen ["votes"])if "votes"in zen .files else X22 [:len (zen ["y"]),vi ]
    assert np .allclose (X22 [:len (v_eski ),vi ],v_eski ),"votes sutunu kaymis"
    votes =np .concatenate ([v_eski ,X1 [eg ][:,vi ]])
    ek ={}
    if "zengin_ad"in zen .files :
        ek ["zengin_ad"]=zen ["zengin_ad"]
    np .savez (BIRLESIK ,X22 =X22 ,XR =XR ,y =y ,pids =pids ,mfg =mfg ,votes =votes ,**ek )
    print (f"  votes tavani {votes .max ():.0f} (urun tavani = model sayisi 4)")
    assert votes .max ()<=4 ,f"votes tavani asildi: {votes .max ()}"
    print (f"BIRLESIK EGITIM VERISI -> {BIRLESIK }")
    print (f"  {len (y )} candidate / {len (np .unique (pids ))} part "
    f"(oncesi {len (zen ['y'])} / {len (np .unique (zen ['pids']))}, "
    f"+{int (eg .sum ())} candidate / +{len (np .unique (np .array ([str (x )for x in yw ['pids']])[eg ]))} part)")
    print (f"  pozitif orani {y .mean ():.3%} (oncesi {np .asarray (zen ['y']).mean ():.3%})")

    # --- URUN GATE'INI YENIDEN EGIT (yapisi birebir korunur)
    eski =pickle .load (open (GATE ,"rb"))
    DON =eski .get ("donusum")
    XF =np .hstack ([X22 ,XR ])
    Z =np .zeros ((len (XF ),XF .shape [1 ]*2 ))
    for u in np .unique (pids ):
        i =np .where (pids ==u )[0 ]
        Z [i ]=wire_gate .within_part (XF [i ],DON )
    assert Z .shape [1 ]==eski ["n_feat"]==116 ,(Z .shape ,eski ["n_feat"])
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z ,y )
    # NOT'U BIR KEZ EKLE. Betigi ikinci times calistirinca `old` already W2 gate'i becomes and
    # duz birlestirme notu IKI KEZ writes (first kosuda became). Ek already varsa yeniden eklenmez.
    yeni_d =dict (eski );yeni_d ["clf"]=clf 
    n =eski .get ("note","")
    while EK in n :
        n =n .replace (EK ,"")
    yeni_d ["note"]=n +EK 
    pickle .dump (yeni_d ,open (GATE ,"wb"))
    md5 =hashlib .md5 (open (GATE ,"rb").read ()).hexdigest ()
    print (f"\nURUN GATE yeniden egitildi -> {GATE }")
    print (f"  MD5 {md5 [:12 ]} | sutun {Z .shape [1 ]} | agac 400")
    print (f"  GERI DONUS: `python w3_dagit.py --geri` (dosya kopyasi DEGIL, tarif)")

    # --- cp_config: training verisi yolu + gate kimligi
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    cfg .setdefault ("current_product",{}).setdefault ("wire_gate",{})
    cfg ["current_product"]["wire_gate"]["egitim_verisi"]=BIRLESIK 
    # RECEIPT DAMGASI: test_artifact `current_product.wire_gate.md5` + `path` alanlarini reads.
    # Ilk yazimda only ic ice `gate_kimlik` blogunu guncelledigim for damga bayat kaldi and
    # test kirmizi yandi (correct davranis) -- damga residual DOGRUDAN buraya da yaziliyor.
    cfg ["current_product"]["wire_gate"]["md5"]=md5 
    cfg ["current_product"]["wire_gate"]["yol"]=GATE 
    cfg ["current_product"]["wire_gate"]["gate_kimlik"]={
    "dosya":GATE ,"md5":md5 ,"sutun":int (Z .shape [1 ]),
    "egitim_verisi":BIRLESIK ,"candidate":int (len (y )),"part":int (len (np .unique (pids ))),
    "donusum":DON ,"router":False ,"tarih":"2026-08-02",
    "evidence":"results/w2_veri_etkisi.json",
    "not":("W2: +110 yeni WEI parcasi. Prospektif 109 part (results/yeni_wei.npz "
    "split=='test') EGITIME GIRMEDI, harcanmamis test kumesidir."),
    }
    with io .open ("cp_config.json","w",encoding ="utf-8")as f :
        json .dump (cfg ,f ,indent =1 ,ensure_ascii =False )
    print ("cp_config.json: egitim_verisi + gate_kimlik guncellendi")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
