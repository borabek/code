# -*- coding: utf-8 -*-
"""T3: BAYES TAVANI -- YAPISAL/ILISKISEL bilgi ayrilamaz payi dusuruyor mu?

T1 (2026-08-04) FIZIKSEL olcumlerin tavani HIC oynatmadigini showed (%12.8 -> %12.8).
[[fizik-bayes-tavanini-oynatmiyor]]: F3 kollari YAPISAL/TOPOLOJIK bilgi getirmek zorunda.

BU BETIK O SORUYU SORAR. Kod tabaninda measured: `wire_gate.py`'de **komsu ya da
periyodiklik ozniteligi SIFIR** -- each candidate single basina karar goruyor. Ama klemens bloklari
DUZENLI DIZIDIR: ten kutbun sekizi ateslediyse missing ikisinin nerede oldugu komsularindan
bellidir. Bu, mevcut uzayda HIC bulunmayan a bilgi ekseni.

BEDAVA OLCUM: yapisal features tamamen ADAY NOKTA KUMESINDEN is computed; `_der_tam.pkl`
icindeki `P`/`Pd` yeterli, network cikarimi GEREKMEZ. (T2 -- hidden feature sondasi -- forward
pass ister and turetme bitene up to beklemek zorunda.)

TEZ DEGISMEZ: no sey egitilmez, `v_o` turetmesi and candidate ureticisi aynen kalir. Bu a
OLCUMDUR; gecmezse B kolu KURULMAZ.

GO (B kolunun sarti): ayrilamaz pay >= %20 GORELI azalmali.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import wire_gate 
from t1_bayes_ceiling import ayrilamaz_pay 

AD =["adim_hata","kafes_uyum","esdogrusal_n","sira_rank","sira_boy",
"komsu_mesafe","komsu_mesafe_z","ayna_esi","yon_uyum","yogunluk"]


def _baskin_adim (t ):
    """1B izdusumde BASKIN ADIM (pitch): ardisik farklarin most sik degeri.

    Klemens bloklarinda kutuplar sabit araliklidir. Farklarin MEDYANI not MODU alinir --
    medyan, missing kutuplarin actigi double araliklardan etkilenir.
    """
    if len (t )<3 :
        return 0.0 
    d =np .diff (np .sort (t ))
    d =d [d >1e-6 ]
    if len (d )<2 :
        return 0.0 
        # 0.5mm kovalarda mod (remesh gurultusu ~0.4mm)
    kova =np .round (d /0.5 ).astype (int )
    v ,c =np .unique (kova ,return_counts =True )
    return float (v [np .argmax (c )]*0.5 )


def yapisal (P ,Pd ):
    """Aday basina ILISKISEL features. Tek adayin own geometrisinden DEGIL,
    parcadaki DIGER adaylara according to konumundan uretilir."""
    n =len (P )
    F =np .zeros ((n ,len (AD )),float )
    if n ==0 :
        return F 
    C =P -P .mean (0 )
    # ana eksenler: adaylarin yayildigi direction (dizinin uzun ekseni)
    if n >=3 :
        _ ,_ ,Vt =np .linalg .svd (C ,full_matrices =False )
        e0 =Vt [0 ]
    else :
        e0 =np .array ([1.0 ,0 ,0 ])
    t =C @e0 
    step_ =_baskin_adim (t )
    for i in range (n ):
        d =np .linalg .norm (P -P [i ],axis =1 )
        d [i ]=np .inf 
        yakin =np .sort (d )[:min (3 ,n -1 )]if n >1 else np .array ([np .inf ])
        km =float (yakin [0 ])if np .isfinite (yakin [0 ])else 0.0 
        if step_ >1e-6 :
        # lattice dugune uzaklik: t[i] adimin full kati mi?
            kalan =abs (t [i ]/step_ -round (t [i ]/step_ ))*step_ 
            F [i ,0 ]=kalan 
            F [i ,1 ]=1.0 if kalan <0.25 *step_ else 0.0 
            # same sirada esdogrusal komsu count (adimin katlarinda duranlar)
            dt =np .abs (t -t [i ])/step_ 
            dik =np .linalg .norm ((P -P [i ])-np .outer (t -t [i ],e0 ),axis =1 )
            F [i ,2 ]=float (np .sum ((np .abs (dt -np .round (dt ))<0.25 )&(dik <2.0 )&
            (np .arange (n )!=i )))
        F [i ,3 ]=float (np .sum (t <t [i ]))
        F [i ,4 ]=float (n )
        F [i ,5 ]=km 
        F [i ,6 ]=(km -np .median ([np .sort (np .linalg .norm (P -P [j ],axis =1 ))[1 ]
        for j in range (n )]))if n >1 else 0.0 
        # AYNA SIMETRISI: parcanin ortasina according to yansimasinda candidate present mi?
        ayna =P [i ]-2 *(C [i ]@e0 )*e0 -P .mean (0 )+P .mean (0 )
        F [i ,7 ]=float (np .min (np .linalg .norm (P -ayna ,axis =1 )))if n >1 else 0.0 
        # YON UYUMU: komsularin ekleme yonuyle same mi (dizide yonler paraleldir)
        if n >1 :
            j =int (np .argmin (d ))
            F [i ,8 ]=float (abs (Pd [i ]@Pd [j ]))
        F [i ,9 ]=float (np .sum (d <10.0 ))
    return F 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()

    X0 ,YP ,Y ,GR ,NA =[],[],[],[],[]# NA = parcadaki KABUL EDILEN candidate count
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]])
        if M .shape [1 ]*2 !=gate ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
        if not k .any ():
            continue 
        P =np .asarray (r ["P"],float )[k ]
        Pd =np .asarray (r ["Pd"],float )[k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if len (G ):
            d =P [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            iyi =(pe .min (1 )<=max (3.0 ,0.06 *float (r ["diag"]))).astype (int )
        else :
            iyi =np .zeros (len (P ),int )
        YPk =yapisal (P ,Pd )
        Mk =M [k ]
        for i in range (len (P )):
            X0 .append (Mk [i ]);YP .append (YPk [i ]);Y .append (iyi [i ]);GR .append (r ["geo"])
            NA .append (len (P ))
    X0 =np .array (X0 ,float );YP =np .array (YP ,float )
    Y =np .array (Y );GR =np .array (GR );NA =np .array (NA )
    X1 =np .hstack ([X0 ,YP ])
    print (f"candidate {len (Y )} | iyi %{100 *Y .mean ():.1f} | sutun {X0 .shape [1 ]} -> {X1 .shape [1 ]}")

    # ------------------------------------------------------------------ REJIM AYRIMI
    # ILK SURUM HAVUZLANMIS OLCTU VE BU BIR OLCUM HATASIYDI (2026-08-04, own hatam):
    # `_baskin_adim` at least 3 point ister, otherwise 0.0 returns. Korpusun %89.5'i DUSUK-CP
    # (1-3 CP) oldugu for that satirlarda TUM lattice sutunlari SABIT SIFIR. Havuzlanmis
    # measurement, yapisal bilginin TANIMSIZ oldugu satirlarla full -> null sonuc garantiydi.
    # Dogru soru "yapi whereas yariyor mu" not, "yapinin VAR OLDUGU places whereas yariyor mu";
    # already hedeflenen 366 FN'nin tamami orada.
    REJIM =[("HAVUZLANMIS",np .ones (len (Y ),bool )),
    ("YAPI TANIMLI (>=4 candidate)",NA >=4 ),
    ("yapi tanimsiz (<4)",NA <4 )]
    S ,gecti_her ={},{}
    for rad ,msk in REJIM :
        if msk .sum ()<60 :
            print (f"\n{rad }: {int (msk .sum ())} candidate -- measurement icin AZ, atlandi");continue 
        print (f"\n=== {rad } ({int (msk .sum ())} candidate, iyi %{100 *Y [msk ].mean ():.1f}) ===")
        print (f"{'oznitelik uzayi':<26}{'kNN uyusmazlik':>16}{'AYRILAMAZ pay':>16}{'ceiling F1~':>11}")
        alt ={}
        for ad ,M in (("MEVCUT (gate ozn.)",X0 ),("+ YAPISAL/ILISKISEL",X1 ),
        ("YALNIZ yapisal",YP )):
            u ,a =ayrilamaz_pay (M [msk ],Y [msk ],GR [msk ])
            alt [ad ]={"uyusmazlik":u ,"ayrilamaz":a ,"ceiling":1 -u /2 }
            print (f"{ad :<26}{100 *u :>15.1f}%{100 *a :>15.1f}%{1 -u /2 :>11.4f}")
        a0 =alt ["MEVCUT (gate ozn.)"]["ayrilamaz"];a1 =alt ["+ YAPISAL/ILISKISEL"]["ayrilamaz"]
        gor =(a0 -a1 )/max (a0 ,1e-9 )
        g =gor >=0.20 
        gecti_her [rad ]={"goreli_azalma":gor ,"gecti":bool (g )}
        S [rad ]=alt 
        print (f"AYRILAMAZ: %{100 *a0 :.1f} -> %{100 *a1 :.1f}  (GORELI %{100 *gor :+.1f})  "
        f"-> {'GECTI'if g else 'gecmedi'}")
        if rad .startswith ("YAPI TANIMLI"):
            print ("  single single katki:")
            for j ,ad in enumerate (AD ):
                _ ,a =ayrilamaz_pay (np .hstack ([X0 [msk ],YP [msk ][:,[j ]]]),Y [msk ],GR [msk ])
                print (f"    {ad :<16} ayrilamaz %{100 *a :.1f}  "
                f"({100 *(a0 -a )/max (a0 ,1e-9 ):+.1f}% goreli)")

    kilit =gecti_her .get ("YAPI TANIMLI (>=4 candidate)",{})
    print (f"\nB KOLUNUN KAPISI = YAPI TANIMLI rejimi -> "
    f"{'GECTI, B KURULUR (only o rejimde)'if kilit .get ('gecti')else 'GECMEDI'}")
    with io .open ("results/t3_structural_ceiling.json","w",encoding ="utf-8")as f :
        json .dump ({"n":int (len (Y )),"regime":S ,"karar":gecti_her ,
        "gate":kilit ,"sutunlar":AD },f ,indent =1 )
    print ("receipt -> results/t3_structural_ceiling.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
