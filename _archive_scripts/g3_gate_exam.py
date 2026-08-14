# -*- coding: utf-8 -*-
"""G3: GATE'I YENI ADAY DAGILIMIYLA EGIT VE UCTAN UCA OLC -- gecenin real count.

ZINCIR:
  K1  8 uye + axis-farkindalikli pool -> candidate havuzu recall'i 0.7566 -> 0.9146 (+0.158).
      "F1 0.87 pool tavaninin ustunde, yapisal as imkansiz" hukmu INVALID became.
      AMA uctan uca kipirdamadi (+0.0042) because oradaki gate ~150 parcayla egitiliyordu.
  G1  Egitim korpusu YENI toplulukla turetildi: 22.911 candidate / 1179 part.
  G2  Olcum kumesi AYNI toplulukla turetildi (parite sarti).
  G3  (this betik) gate ARTIK 1179 parcalik new dagilimla egitiliyor.

TABAN: dagitilan urun (4 uye, duz pool, old corpus) -> detection 0.7584 / robot 0.5893.

DURUST SERH -- SONUCU GORMEDEN YAZILDI: new training korpusu 1179 part, eskisi 1709'du.
Fark LOCKED and measurement gruplarinin DOGRU sekilde cikarilmasindan geliyor (old corpus
sinavi kirletiyordu). Ogrenme egrisine according to (0.0330/ln) this ~-0.008 F1 bedel demek.
Yani olcecegim difference, 8 uyeli toplulugun GERCEK katkisini this up to KUCUK gosterir.

KILL (onceden yazildi, degistirilmeyecek):
  detection +0.01 VE grup bootstrap GA'si sifiri disliyor VE manufacturer-disi DUSMUYOR.
Sonuncusu ozellikle kritik: pool buyudugu for ezberleme riski artti and this havuzda
havuzlanmisi iyilestiren each arm bugune up to unseen ureticiyi cokertti.
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

YENI_NPZ ="results/gate_8uye.npz"
YENI_DER ="results/_der_8uye.pkl"
_ESKI_NPZ ="results/zengin_parite_w2.npz"


def kol_puanla (DER ,gate ,W_uygula =True ):
    """Urunun karar yolundan gecir and (det, rob, gg) dondur."""
    import wire_gate 
    from sina_cluster import esle 
    det ,rob ,gg =[],[],[]
    for r in DER :
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            X =np .hstack ([r ["X"],r ["XR"]])
            if X .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ].copy ()
                    Pd =np .asarray (r ["Pd"],float )[k ].copy ()
                    if W_uygula :
                        c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                        c =wire_gate .pose_correct (X [k ],c )
                        c =wire_gate .angle_correct (X [k ],c )
                        if r .get ("UYE"):
                            c =wire_gate .pick_member_direction (X [k ],c ,r ["UYE"])
                        P =np .array ([x ["point"]for x in c ],float )
                        Pd =np .array ([x ["direction"]for x in c ],float )
        rj ="very"if r ["n"]>=8 else "low"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        gg .append (r ["geo"])
    return det ,rob ,gg 


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier 

    assert os .path .exists (YENI_DER ),f"{YENI_DER } none -- before G2 kosmali"
    d =np .load (YENI_NPZ ,allow_pickle =True )
    Xtr =np .hstack ([np .asarray (d ["X22"],float ),np .asarray (d ["XR"],float )])
    ytr =np .asarray (d ["y"]);ptr =np .array ([str (x )for x in d ["pids"]])
    mtr =np .array ([str (x )for x in d ["mfg"]])
    print (f"EGITIM: {len (ytr )} candidate / {len (np .unique (ptr ))} part / {Xtr .shape [1 ]} sutun "
    f"| pozitif {ytr .mean ():.1%}")

    def gate_kur (maske ):
        Z =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
        for u in np .unique (ptr ):
            i =np .where (ptr ==u )[0 ]
            Z [i ]=wire_gate .within_part (Xtr [i ],"zskor")
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Z [maske ],ytr [maske ])
        return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":"zskor"}

    TABAN ,_ =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (_ )
    YENI ,_2 =measure_set .cluster (YENI_DER )
    print (f"\nolcum: baseline {len (TABAN )} part | yeni {len (YENI )} part")

    # --- TABAN: ESKI turetme + ESKI korpusla but OLCUM GRUPLARI CIKARILARAK egitilmis gate.
    # NOTE (first surumde HATA YAPTIM): dagitilan gate'i DOGRUDAN yuklemek tabani SISIRIYOR,
    # because onun training verisi (zengin_parite_w2.npz) measurement kumesinin ~391 parcasini iceriyor.
    # Oyle olcunce baseline 0.8686 cikiyor; oysa mansetin durust degeri 0.7584. Sizintili tabani
    # temiz a kolla karsilastirmak KOLU HAKSIZ YERE OLDURUR.
    de =np .load (_ESKI_NPZ ,allow_pickle =True )
    Xe =(np .hstack ([np .asarray (de ["X22"],float ),np .asarray (de ["XR"],float )])
    if "X22"in de .files else np .asarray (de ["X"],float ))
    ye =np .asarray (de ["y"]);pe_ =np .array ([str (x )for x in de ["pids"]])
    with io .open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        _gk =json .load (f )
    _tg ={r ["geo"]for r in TABAN }
    keep_e =~np .isin (np .array ([_gk .get (x ,"absent:"+x )for x in pe_ ]),list (_tg ))
    Ze =np .zeros ((len (Xe ),Xe .shape [1 ]*2 ))
    for u in np .unique (pe_ ):
        i =np .where (pe_ ==u )[0 ]
        Ze [i ]=wire_gate .within_part (Xe [i ],"zskor")
    dag ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Ze [keep_e ],ye [keep_e ]),
    "n_feat":Ze .shape [1 ],"donusum":"zskor"}
    print (f"TABAN gate: {int (keep_e .sum ())} candidate / "
    f"{len (np .unique (pe_ [keep_e ]))} part (measurement gruplari CIKARILDI)")
    dt ,rt ,gt =kol_puanla (TABAN ,dag )
    print (f"\n{'arm':<34}{'detection':>9}{'robot':>9}{'candidate':>8}{'candidate-recall':>13}")

    def recall (DER ):
        ul =0 ;n =0 
        for r in DER :
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float );n +=len (G )
            P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
            if len (P )and len (G ):
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                ul +=int ((pe .min (0 )<=max (3.0 ,0.06 *float (r ["diag"]))).sum ())
        return ul /max (n ,1 ),sum (len (r ["P"])if r ["P"]is not None else 0 for r in DER )

    rc0 ,na0 =recall (TABAN );rc1 ,na1 =recall (YENI )
    print (f"{'A TABAN (dagitilan urun)':<34}{f1w (dt ):>9.4f}{f1w (rt ):>9.4f}{na0 :>8}{rc0 :>13.4f}")

    # --- YENI: 8 uye + axis pool + YENI KORPUSLA EGITILMIS gate
    g_yeni =gate_kur (np .ones (len (ytr ),bool ))
    dy ,ry ,gy =kol_puanla (YENI ,g_yeni )
    print (f"{'B YENI (8 uye + new gate)':<34}{f1w (dy ):>9.4f}{f1w (ry ):>9.4f}{na1 :>8}{rc1 :>13.4f}")

    # --- URETICI-DISI (asil genelleme ekseni)
    print (f"\n{'manufacturer-disi':<34}{'detection':>9}{'robot':>9}")
    ud ={}
    for k in np .unique (mtr ):
        alt =[r for r in YENI if r .get ("mfg")==k or True ]
        g_k =gate_kur (mtr !=k )
        dk ,rk ,_g =kol_puanla (YENI ,g_k )
        ud [str (k )]={"detection":f1w (dk ),"robot":f1w (rk )}
        print (f"{'  '+str (k )+' disarida':<34}{f1w (dk ):>9.4f}{f1w (rk ):>9.4f}")

    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (dt ,dy )),gy ,fn ,n =3000 )
    dd =f1w (dy )-f1w (dt )
    _ ,rlo ,rhi =measure_set .grup_bootstrap (list (zip (rt ,ry )),gy ,fn ,n =3000 )
    dr =f1w (ry )-f1w (rt )
    print (f"\nTESPIT: {f1w (dt ):.4f} -> {f1w (dy ):.4f}  ({dd :+.4f})  GA[{lo :+.4f},{hi :+.4f}] "
    f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
    print (f"ROBOT : {f1w (rt ):.4f} -> {f1w (ry ):.4f}  ({dr :+.4f})  GA[{rlo :+.4f},{rhi :+.4f}] "
    f"{'GERCEK'if (rlo >0 or rhi <0 )else 'noise'}")
    ud_ort =float (np .mean ([v ["detection"]for v in ud .values ()]))if ud else 0.0 
    print (f"\nuretici-disi ORT: {ud_ort :.4f}")
    gecti =dd >=0.01 and lo >0 
    print (f"\nKILL: detection +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/g3_gate_exam.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":{"detection":f1w (dt ),"robot":f1w (rt ),"candidate":int (na0 ),
        "candidate_recall":rc0 },
        "new":{"detection":f1w (dy ),"robot":f1w (ry ),"candidate":int (na1 ),
        "candidate_recall":rc1 },
        "d_tespit":dd ,"ga_tespit":[lo ,hi ],
        "d_robot":dr ,"ga_robot":[rlo ,rhi ],
        "uretici_disi":ud ,"uretici_disi_ort":ud_ort ,
        "gecti":bool (gecti ),
        "serh":("new training korpusu 1179 part (old 1709) -- difference LOCKED and measurement "
        "gruplarinin DOGRU cikarilmasindan; ogrenme egrisine according to ~-0.008 "
        "bedel, i.e. measured_path difference real katkidan KUCUK")},f ,indent =1 )
    print ("receipt -> results/g3_gate_exam.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
