# -*- coding: utf-8 -*-
"""U10: KAHIN ORANINI DURUST OLC -- baseline, arm and kahin AYNI KOSUDAN.

DUZELTILEN HATA (2026-08-02): T8'de "kahinin %66'si yakalandi" yazdim; kahin payini SABIT
0.0536'ya boldum. O sabit BASKA BIR ZINCIRDE olculmustu (uye selector dagitilmadan before,
baseline 0.5523). T8'in own tabani whereas 0.5280 idi. Taban degisince hem pay hem payda kayar and
ratio ANLAMSIZLASIR. Nitekim single degiskenle olcunce "pool bilesimi" farki absent became.

Bu, `f1w` tuzagiyla same sinifta: FARKLI KOSULARDAN gelen sayilari same formulde kullanmak.

BU BETIK ucunu de TEK KOSUDA olcer:
    T  baseline        : uye selector KAPALI (pose + angle)
    K  arm          : uye selector OPEN (dagitilan)
    O  kahin        : GT'ye bakip havuzdaki EN IYI yonu sec (ulasilabilir DEGIL, upper boundary)
and oranI (K-T)/(O-T) as gives -- payda AYNI kosudan.

Ayrica GUNCEL bosslugu gives: dagitilan zincir this an nerede, kahin nerede.
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
YAKIN =5.0 


def kahin_orani (baseline ,arm ,oracle_ ):
    """(arm - baseline) / (kahin - baseline). UCU DE AYNI KOSUDAN gelmeli.

    Sabit a kahin degerine bolmek YANLISTIR: baseline degisince ratio anlamini yitirir
    (2026-08-02'de full this error yapildi). Bu fonksiyon ucunu birden ister ki ayrisamasinlar.
    """
    pay =arm -baseline 
    payda =oracle_ -baseline 
    if payda <=1e-9 :
        return float ("nan")
    return pay /payda 


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }
    g =[r ["geo"]for r in DER ]

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in tpid ]),list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":DON }
    uye_m =wire_gate ._load (wire_gate .UYE_PATH )
    print (f"gate + uye selector hazir",flush =True )

    def havuz_kur (P ,Pd ,i ,uyeler ):
        hav =[(Pd [i ],1.0 ,0.0 )]
        for lst in uyeler :
            for m in lst :
                q =np .asarray (m ["point"],float )
                dd =float (np .linalg .norm (q -P [i ]))
                if dd <=YAKIN :
                    hav .append ((np .asarray (m ["direction"],float ),
                    float (m .get ("confidence",1.0 )),dd ))
        return hav 

        # SECIM URUNUN KENDI FONKSIYONUNDAN. Ilk surumde here UCUNCU times elle yeniden
        # yazilmisti and 0.5704 veriyordu; dagitilan `wire_gate.pick_member_direction` 0.5801 veriyor
        # (two bagimsiz betikte dogrulandi). Yani deviation URUNDE not OLCUMDE idi -- this gece
        # konulan 'measurement urunu TAKLIT ETMEZ, AYNI KODU cagirir' kuralinin ihlaliydi.

    def puanla (mod ):
        """mod: 'baseline' (selector absent) | 'arm' (dagitilan selector) | 'kahin' (GT'ye bakar)."""
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 );k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    cps =wire_gate .pose_correct (X58 [k ],cps )
                    cps =wire_gate .angle_correct (X58 [k ],cps )
                    P =np .array ([c ["point"]for c in cps ],float )
                    Pd =np .array ([c ["direction"]for c in cps ],float )
                    if mod =="arm"and r .get ("UYE"):
                        cps =wire_gate .pick_member_direction (X58 [k ],cps ,r ["UYE"])
                        Pd =np .array ([c ["direction"]for c in cps ],float )
                    elif mod =="kahin"and r .get ("UYE"):
                        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
                        for i in range (len (P )):
                            hav =havuz_kur (P ,Pd ,i ,r ["UYE"])
                            if len (hav )<2 or not len (G ):
                                continue 
                            DIR =np .array ([h [0 ]for h in hav ])
                            DIR =DIR /(np .linalg .norm (DIR ,axis =1 ,keepdims =True )+1e-9 )
                            b =int (np .argmin (np .linalg .norm (G -P [i ],axis =1 )))
                            a =np .degrees (np .arccos (np .clip (np .abs (DIR @Gd [b ]),0 ,1 )))
                            Pd [i ]=DIR [int (np .argmin (a ))]
            rj ="very"if r ["n"]>=8 else "low"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    R ={}
    for mod ,ad in (("baseline","T baseline (selector KAPALI)"),("arm","K dagitilan selector"),
    ("kahin","O KAHIN (upper boundary)")):
        d ,rr =puanla (mod )
        R [mod ]=(float (f1w (d )),float (f1w (rr )),rr )
        print (f"{ad :<26}tespit {R [mod ][0 ]:.4f} | ROBOT {R [mod ][1 ]:.4f}",flush =True )

    T ,K ,O =R ["baseline"][1 ],R ["arm"][1 ],R ["kahin"][1 ]
    ratio =kahin_orani (T ,K ,O )
    print (f"\n=== AYNI KOSUDAN ===")
    print (f"  baseline {T :.4f} -> arm {K :.4f} -> kahin {O :.4f}")
    print (f"  kolun kazanci {K -T :+.4f} | kahinin toplami {O -T :+.4f} | YAKALAMA {ratio :.0%}")
    print (f"  KALAN BOSLUK: {O -K :+.4f}")
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (R ["baseline"][2 ],R ["arm"][2 ])),g ,fn ,n =2000 )
    print (f"  kolun GA'si [{lo :+.4f}, {hi :+.4f}]")
    print (f"\nESKI (YANLIS) HESAP: kahin payini SABIT 0.0536'ya bolmustum; o sabit")
    print (f"BASKA bir zincirden geliyordu (baseline 0.5523). Dogru payda: {O -T :.4f}")
    with io .open ("results/u10_kahin_orani.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":T ,"arm":K ,"kahin":O ,"yakalama":float (ratio ),
        "kalan_bosluk":float (O -K ),"ga":[float (lo ),float (hi )]},f ,indent =1 )
    print ("receipt -> results/u10_kahin_orani.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
