# -*- coding: utf-8 -*-
"""U6: part-ici z-skor kararini YEDI bolmede sina (single bolmeye mi dayaniyordu?).

TEREDDUT: dagitim karari, gorulmemis-manufacturer ekseninde TEK a sayiya dayaniyor (WEI disarida
0.4832 -> 0.5702). Elimizde only IKI manufacturer oldugu for that eksende TOPLAM two split present; "most
kotu state" single a olcumdur and gurultulu may be. Karar that single bolmeye fit olduysa real a
ucuncu ureticide caresiz kaliriz.

BU DENEY: part numarasinin first 2 hanesi = URUN SERISI. Seri-disi split, manufacturer-disindan
DAHA KOLAY a eksendir (same ureticinin istatistigi egitimde kalir) -- instead of gecmez, but
YEDI bagimsiz split gives. Sorulan sey sudur:

    z-skor GENEL a saglamlik kazanci mi, otherwise only WEI bolmesinde mi calisiyordu?

Bir arm seven seride de same yone gidiyorsa karar saglamdir; only birinde kazanip otekilerde
kaybediyorsa single bolmeye fit olmusuz demektir.

TURETME onbellekten (results/_u4_der.pkl) -- network cikarimi absent.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
ONEK =2 
MIN_TEST ,MIN_EGITIM =10 ,30 


def main ():
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}

    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    tr_onek =np .array ([p [:ONEK ]for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }

    # HANGI SERILER: hem testte hem egitimde yeterli kutle must be
    te_say =collections .Counter (r ["pid"][:ONEK ]for r in DER )
    tr_say =collections .Counter (p [:ONEK ]for p in np .unique (tr_pid ))
    seriler =sorted (k for k in te_say if te_say [k ]>=MIN_TEST and tr_say [k ]>=MIN_EGITIM )
    print (f"{len (seriler )} seri | DER {len (DER )} part")

    Z =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xtr [i ],"zskor")
    MAT ={"A ham":Xtr ,"D zskor":Z }

    def puanla (clf ,alt ,don ):
        det =[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                s =clf .predict_proba (wire_gate .within_part (r ["X"],don ))[:,1 ]
                m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            det .append (("cok"if r ["n"]>=8 else "dusuk",)
            +esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        return det 

    print (f"\n{'seri':<8}{'manufacturer':>9}{'n':>5}{'A ham':>9}{'D zskor':>10}{'fark':>9}")
    S ,farklar ={},[]
    for k in seriler :
        alt =[r for r in DER if r ["pid"][:ONEK ]==k ]
        keep =(tr_onek !=k )&~np .isin (tr_grp ,list (tg ))
        v ={}
        for ad ,M in MAT .items ():
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [keep ],ytr [keep ])
            v [ad ]=f1w (puanla (clf ,alt ,None if ad =="A ham"else "zskor"))
        fark =v ["D zskor"]-v ["A ham"]
        farklar .append (fark )
        mf =collections .Counter (mfg_of .get (r ["pid"],"?")for r in alt ).most_common (1 )[0 ][0 ]
        print (f"{k :<8}{mf :>9}{len (alt ):>5}{v ['A ham']:>9.4f}{v ['D zskor']:>10.4f}{fark :>+9.4f}",
        flush =True )
        S [k ]={"manufacturer":mf ,"n":len (alt ),"A":float (v ["A ham"]),"D":float (v ["D zskor"]),
        "fark":float (fark )}

    f =np .array (farklar )
    kazanan =int ((f >0 ).sum ());kaybeden =int ((f <0 ).sum ())
    print (f"\n{kazanan }/{len (f )} seride D kazandi | ortalama {f .mean ():+.4f} | "
    f"medyan {np .median (f ):+.4f} | en kotu {f .min ():+.4f} | en iyi {f .max ():+.4f}")

    # ISARET TESTI (dagilimsiz): D gercekten genel mi, otherwise single bolmeye mi fit?
    from math import comb 
    n =int ((f !=0 ).sum ())
    p_iki_yonlu =min (1.0 ,2 *sum (comb (n ,i )for i in range (kazanan ,n +1 ))/2 **n )
    print (f"sign testi: {kazanan }/{n } pozitif -> p = {p_iki_yonlu :.4f}")
    yorum =("GENEL saglamlik kazanci"if (kazanan >=len (f )-1 and f .mean ()>0 )
    else "TEK BOLMEYE FIT -- karar kirilgan"if kazanan <=len (f )//2 
    else "KARISIK: cogunlukta kazaniyor ama tekduze degil")
    print (f"YORUM: {yorum }")
    print ("\nNOT: seri-disi, manufacturer-disindan DAHA KOLAY bir eksendir (ayni ureticinin\n"
    "istatistigi egitimde kalir). Yerine gecmez; kararin TEK bolmeye fit olup olmadigini\n"
    "sinamak icin kullanilir.")
    with open ("results/u6_seri_disi.json","w",encoding ="utf-8")as fh :
        json .dump ({"seriler":S ,"kazanan_seri":kazanan ,"n_seri":len (f ),
        "ortalama_fark":float (f .mean ()),"medyan_fark":float (np .median (f )),
        "isaret_testi_p":float (p_iki_yonlu ),"yorum":yorum },fh ,indent =1 )
    print ("receipt -> results/u6_seri_disi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
