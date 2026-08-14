# -*- coding: utf-8 -*-
"""W2: SALT VERI ETKISI -- same gate, +85 grup new WEI parcasi.

Denetimin P2.3 maddesi: "Ayni RF gate'i no baska degisiklik olmadan yeniden egit.
Boylece only data etkisi olculsun."

VERI (w1): korpusta HIC OLMAYAN 219 WEI parcasi islendi (2649 candidate).
    training yarisi        1343 candidate  -> training korpusuna EKLENIR
    PROSPEKTIF TEST      1306 candidate  -> HIC EGITILMEZ, ayri a manufacturer-ici genelleme testi

Iki measurement:
    1. MEVCUT gelistirme kumesi (194 part) -- havuzlanmis + manufacturer kirilimi
    2. PROSPEKTIF WEI TESTI  -- gate'in HIC gormedigi 84 grup

TEZ CIZGISI: same network, same remesh, same candidate ureticisi, same RF ayarlari, same ozellikler.
Degisen TEK sey training korpusunun buyuklugu.

KILL (denetimin yazdigi P2): prospektif WEI artisi < +0.02 VE havuzlanmis artis < +0.005
whereas, ya da PXC kaybi > 0.005 whereas salt-data kolu KAPANIR.
"""
import collections 
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


def main ():
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1_rejim ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    X0 =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    y0 =np .asarray (zen ["y"]);p0 =np .array ([str (x )for x in zen ["pids"]])
    m0 =np .array ([str (x )for x in zen ["mfg"]])

    yw =np .load ("results/yeni_wei.npz",allow_pickle =True )
    # NOTE: w1'de `feats_for` WG_ZENGIN=1 with cagrildigi for zengin blogu ZATEN iceriyor
    # (X22 58 column) and also kaydedilen XR onun last 36 sutununun BIREBIR KOPYASI (dogrulandi).
    # Korpus tarafi whereas 22+36 as AYRI kaydedilmisti; sıralama same (22 temel, after 36 zengin).
    X1 =np .asarray (yw ["X22"],float )
    if X1 .shape [1 ]==22 :# old form: XR ayri kaydedilmis
        X1 =np .hstack ([X1 ,np .asarray (yw ["XR"],float )])
    assert X1 .shape [1 ]==X0 .shape [1 ],f"sutun uyusmazligi {X1 .shape [1 ]} vs {X0 .shape [1 ]}"
    y1 =np .asarray (yw ["y"]);p1 =np .array ([str (x )for x in yw ["pids"]])
    b1 =np .array ([str (x )for x in yw ["split"]])
    P1 =np .asarray (yw ["pts"],float );D1 =np .asarray (yw ["dirs"],float )
    N1 =np .asarray (yw ["ngt"])
    eg =b1 =="training"
    print (f"\nmevcut corpus {len (y0 )} candidate / {len (np .unique (p0 ))} part")
    print (f"yeni WEI: training {int (eg .sum ())} candidate / {len (np .unique (p1 [eg ]))} part | "
    f"PROSPEKTIF TEST {int ((~eg ).sum ())} candidate / {len (np .unique (p1 [~eg ]))} part",flush =True )

    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")

    def donustur (X ,pidler ):
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pidler ):
            i =np .where (pidler ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],DON )
        return Z 

    Z0 =donustur (X0 ,p0 )
    Z1 =donustur (X1 ,p1 )
    grp0 =np .array ([gk .get (p ,"absent:"+p )for p in p0 ])
    keep0 =~np .isin (grp0 ,list (tg ))

    # --- PROSPEKTIF TEST puanlama (part bazinda, GT'siz candidate absent)
    def prospektif (m ):
        det =[]
        for u in np .unique (p1 [~eg ]):
            i =np .where ((p1 ==u )&(~eg ))[0 ]
            s =wire_gate .decision_score (m ,X1 [i ])
            k =wire_gate .decision_mask (s )
            tp =int (y1 [i ][k ].sum ());fp =int (k .sum ()-tp )
            fn =int (N1 [i ][0 ]-tp )
            det .append (("very"if N1 [i ][0 ]>=8 else "low",tp ,fp ,max (fn ,0 )))
        return det 

    def gelistirme (m ,alt ):
        det =[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                Xr =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (m ,Xr )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ];Pd =r ["Pd"][k ]
            det .append (("very"if r ["n"]>=8 else "low",)
            +esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        return det 

    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in p0 [m0 ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (m0 )}
    BOLME =[("havuzlanmis",None ,DER )]
    for k ,mad in kod .items ():
        alt =[x for x in DER if x ["mfg"]==mad ]
        if len (alt )>=10 :
            BOLME .append ((mad ,k ,alt ))

            # YENI VERININ URETICISI: all of them WEI. Uretici-disi a bolmede that ureticinin YENI verisi de
            # egitimden CIKARILMALI -- otherwise "WEI-disi" olcumu WEI verisiyle egitilmis becomes and
            # protokolun kendisi bozulur. (Ilk kosuda this hatayi yaptim: WEI satiri +0.0747 showed,
            # because B kolunda WEI cikarilip new WEI geri konuyordu. O number GECERSIZDI.)
    WEI_KOD =[k for k ,mad in kod .items ()if mad =="WEI"]
    yeni_mfg ="WEI"

    def egit (kp ,ekle ,mfg_disi ,seed ):
        ek =ekle and (mfg_disi !=yeni_mfg )# own ureticisi disarida whereas new data de absent
        Xtr =np .vstack ([Z0 [kp ],Z1 [eg ]])if ek else Z0 [kp ]
        ytr =np .concatenate ([y0 [kp ],y1 [eg ]])if ek else y0 [kp ]
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =seed ).fit (Xtr ,ytr )
        return {"clf":clf ,"n_feat":Xtr .shape [1 ],"donusum":DON },ek 

    SON ,PARCA ,TOHUM ={},{},{}
    print (f"\n{'arm':<20}"+"".join (f"{b :>13}"for b ,_ ,_ in BOLME )+f"{'PROSPEKTIF':>12}")
    for ad ,ekle in (("A mevcut corpus",False ),("B +new WEI",True )):
        SON [ad ]={};TOHUM [ad ]={};sat =f"{ad :<20}"
        for b ,mk ,alt in BOLME :
            kp =keep0 .copy ()
            if mk is not None :
                kp =kp &(m0 !=mk )
            deg =[]
            for th in (0 ,1 ,2 ):
                m ,ek =egit (kp ,ekle ,b if mk is not None else None ,th )
                det =gelistirme (m ,alt )
                deg .append (float (f1w (det )))
                if th ==0 :
                    PARCA [(ad ,b )]=(det ,[x ["geo"]for x in alt ])
            SON [ad ][b ]=float (np .mean (deg ));TOHUM [ad ][b ]=deg 
            im =""if (ekle and b ==yeni_mfg )else ""
            sat +=f"{np .mean (deg ):>13.4f}"
        deg =[]
        for th in (0 ,1 ,2 ):
            m ,_ =egit (keep0 ,ekle ,None ,th )
            dp_ =prospektif (m )
            deg .append (float (f1w (dp_ )))
            if th ==0 :
                PARCA [(ad ,"PROSPEKTIF")]=(dp_ ,[gk .get (u ,"absent:"+u )
                for u in np .unique (p1 [~eg ])])
        SON [ad ]["PROSPEKTIF"]=float (np .mean (deg ));TOHUM [ad ]["PROSPEKTIF"]=deg 
        sat +=f"{np .mean (deg ):>12.4f}"
        print (sat ,flush =True )
    print (f"  (WEI satiri: yeni data WEI oldugu for WEI-disi bolmede EKLENMEZ -> "
    f"A=B beklenir, protocol korunur)")
    print ("\ntohum yayilimi (3 seed):")
    for ad in SON :
        print (f"  {ad :<18}"+"  ".join (f"{b }={'/'.join (f'{v :.4f}'for v in TOHUM [ad ][b ])}"
        for b in SON [ad ]))

    A ,B =SON ["A mevcut corpus"],SON ["B +new WEI"]
    print (f"\n{'split':<16}{'A':>10}{'B':>10}{'difference':>10}{'%95 GA':>22}")
    GA ={}
    for b in list (A ):
        da ,g =PARCA [("A mevcut corpus",b )];db ,_ =PARCA [("B +new WEI",b )]
        fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
        _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (da ,db )),g ,fn ,n =2000 )
        GA [b ]=(lo ,hi )
        print (f"{b :<16}{A [b ]:>10.4f}{B [b ]:>10.4f}{B [b ]-A [b ]:>+10.4f}"
        f"   [{lo :+.4f}, {hi :+.4f}] {'GERCEK'if (lo >0 or hi <0 )else 'noise'}")

    dp =B ["PROSPEKTIF"]-A ["PROSPEKTIF"]
    dh =B ["havuzlanmis"]-A ["havuzlanmis"]
    dpxc =B .get ("PXC",0 )-A .get ("PXC",0 )
    # OKUMA KURALI (sonuc gorulmeden sabitlendi, first kosunun ardindan):
    #   KAZANC  : prospektif >= +0.02 VE GA'si sifiri disliyor  (ana criterion)
    #   ZARAR   : PXC kaybi > 0.005 ISE, that loss however GA'si sifiri disliyorsa GERCEK sayilir.
    #             Nokta tahmini gurultunun icindeyse koruma ATESLENMEMISTIR.
    # Gerekce: koruma maddesi "a ureticiye zarar verme" for kondu; ayirt edilemeyen a
    # difference zarar KANITI degildir. Ama gevsetme not, SIMETRI: kazanci da GA with ariyorum.
    pl ,ph =GA ["PROSPEKTIF"];xl ,xh =GA .get ("PXC",(0.0 ,0.0 ))
    kazanc =(dp >=0.02 )and (pl >0 )
    zarar =(dpxc <-0.005 )and (xh <0 )
    gecti =kazanc and not zarar 
    print (f"\nKILL: kazanc = prospektif >= +0.02 VE GA>0 | zarar = PXC < -0.005 VE GA<0")
    print (f"  prospektif {dp :+.4f} GA[{pl :+.4f},{ph :+.4f}] -> kazanc={kazanc }")
    print (f"  PXC        {dpxc :+.4f} GA[{xl :+.4f},{xh :+.4f}] -> zarar ={zarar }")
    print (f"  havuzlanmis{dh :+.4f} (bilgi)")
    print (f"  -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/w2_data_etkisi.json","w",encoding ="utf-8")as f :
        json .dump ({"A":A ,"B":B ,"seed":TOHUM ,"ga":{k :list (v )for k ,v in GA .items ()},
        "kazanc":bool (kazanc ),"zarar":bool (zarar ),"gecti":bool (gecti ),
        "not":"WEI-disi bolmede new WEI verisi EKLENMEZ (protocol)"},
        f ,indent =1 )
    print ("receipt -> results/w2_data_etkisi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
