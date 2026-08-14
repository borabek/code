# -*- coding: utf-8 -*-
"""R7b: KOSULLU 2. ASAMA -- R7'nin tasarim kusurunu duzelt, after verdict ver.

R7 SONUCU: sifir ates. Sebep model not TASARIM:
  * 2. stage TUM adaylarin satirlarinda egitildi. O dagilimda `mevcut` girisi %83.6
    correct; digerleri ~%45. Model correct sekilde HEP `mevcut`i secti and argmax never
    degismedi. Yani 2. stage, 1. asamanin verdigi bilgiyi (this point KOTU) never kullanmadi.
  * Bu, direction kolundaki hatanin same ailesi: sorun modelde not, EGITIM DAGILIMINDA.
    Orada label isaretsizdi; here condition absent sayildi.

DUZELTME (two degisiklik):
  1. 2. stage YALNIZ `mevcut`in KOTU oldugu adaylarin satirlariyla egitilir.
     Ogrenilen soru residual correct soru: "mevcut already kotuyken hangi giris <=2mm?"
  2. `mevcut` 2. asamanin secenekleri arasindan CIKARILIR. 1. stage already onu reddetti;
     secenekler between birakmak prior'u geri sokar. Karar kurali: 1. stage esigi
     GECERSE, alternatiflerin argmax'i alinir and MUTLAK confidence esigi aranir.

Bu, hasar asimetrisine (1:20) uygun single mesru mimari: YUKSEK KESINLIKLI gate + kosullu secim.

Hala DAGITIM DEGIL: grup-capraz fizibilite. Gecerse R8 (training korpusu) kosulur.
"""
import io ,json ,os ,pickle ,sys 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set 
import wire_gate 
from sina_cluster import esle ,f1w 
from r5_konum_teshis import yon_uygula 
from r7_konum_iki_asamali import dagilim_oz ,GIRIS 

ONB ="results/_r4_sozluk.pkl"


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .metrics import roc_auc_score 
    DER ,gate ,ek =T2 .yukle ()
    PARCA =pickle .load (open (ONB ,"rb"))
    YS =wire_gate ._load ("results/yon_secici.pkl")

    H ={}
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        H [r ["pid"]]=None if d_ is None else (d_ ["P"].copy (),yon_uygula (d_ ,YS ),
        d_ ["SK"],d_ ["X"])

    AX ,AY ,AG ,AK =[],[],[],[]
    BX ,BY ,BG ,BK =[],[],[],[]
    NOK ={}
    for r in DER :
        h =H [r ["pid"]]
        if h is None :
            continue 
        P ,Pd ,SK ,X =h 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (P )or not len (G ):
            continue 
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        for i in range (len (P )):
            if not np .isfinite (pe [i ]).any ():
                continue 
            b =int (np .argmin (pe [i ]))
            if pe [i ,b ]>tol :
                continue 
            S =SK [i ]if i <len (SK )else {}
            oz =dagilim_oz (S ,P [i ],Pd [i ])
            kotu =int (pe [i ,b ]>2.0 )
            AX .append (np .concatenate ([X [i ],oz ]));AY .append (kotu )
            AG .append (r ["geo"]);AK .append ((r ["pid"],i ))
            NOK [(r ["pid"],i )]={}
            for ad in GIRIS :
                if ad not in S :
                    continue 
                q =np .asarray (S [ad ],float )
                NOK [(r ["pid"],i )][ad ]=q 
                w =q -G [b ]
                yy =float (np .linalg .norm (w -float (w @Gd [b ])*Gd [b ]))
                e =[1.0 if ad ==t2 else 0.0 for t2 in GIRIS ]
                e +=[float (np .linalg .norm (q -P [i ])),
                float (np .linalg .norm ((q -P [i ])-float ((q -P [i ])@Pd [i ])*Pd [i ]))]
                BX .append (np .concatenate ([X [i ],oz ,e ]));BY .append (int (yy <=2.0 ))
                BG .append (r ["geo"]);BK .append ((r ["pid"],i ,ad ,kotu ))
    AX =np .array (AX ,float );AY =np .array (AY );AG =np .array (AG )
    BX =np .array (BX ,float );BY =np .array (BY );BG =np .array (BG )
    Bk =np .array ([k [3 ]for k in BK ])# this satirin adayinda mevcut KOTU mu
    print (f"asama1 {len (AY )} candidate (kotu {AY .mean ():.1%})")
    print (f"asama2 TUM {len (BY )} satir (iyi {BY .mean ():.1%}) | "
    f"KOSULLU alt cluster {int (Bk .sum ())} satir (iyi {BY [Bk ==1 ].mean ():.1%})",flush =True )

    ug =np .unique (np .concatenate ([AG ,BG ]));rng =np .random .RandomState (0 );rng .shuffle (ug )
    fold ={g :j %5 for j ,g in enumerate (ug )}
    ka =np .array ([fold [g ]for g in AG ]);kb =np .array ([fold [g ]for g in BG ])
    oofA =np .zeros (len (AY ));oofB =np .zeros (len (BY ))
    for f_ in range (5 ):
        ta =ka !=f_ ;tb =(kb !=f_ )&(Bk ==1 )# KOSULLU training
        cA =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (AX [ta ],AY [ta ])
        oofA [~ta ]=cA .predict_proba (AX [~ta ])[:,1 ]
        if len (np .unique (BY [tb ]))<2 :
            continue 
        cB =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (BX [tb ],BY [tb ])
        oofB [kb ==f_ ]=cB .predict_proba (BX [kb ==f_ ])[:,1 ]
    aucA =float (roc_auc_score (AY ,oofA ))
    m =Bk ==1 
    aucB =float (roc_auc_score (BY [m ],oofB [m ]))if len (np .unique (BY [m ]))>1 else 0.5 
    print (f"AUC asama1 {aucA :.4f} | AUC asama2 (KOTU adaylarda) {aucB :.4f}",flush =True )

    SA ={k :oofA [j ]for j ,k in enumerate (AK )}
    SB ={}
    for j ,(pid ,i ,ad ,_k )in enumerate (BK ):
        SB .setdefault ((pid ,i ),{})[ad ]=oofB [j ]

    def kos (threshold ,confidence ):
        rob ,det ,gg =[],[],[]
        na =0 ;iyi_vur =0 ;kotu_vur =0 
        for r in DER :
            h =H [r ["pid"]]
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            if h is None :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            else :
                P0 ,Pd ,SK ,_X =h 
                P =P0 .copy ()
                for i in range (len (P )):
                    key =(r ["pid"],i )
                    if SA .get (key ,0.0 )<threshold :
                        continue 
                    sc =SB .get (key )
                    if not sc :
                        continue 
                    en =max (sc ,key =sc .get )
                    if sc [en ]<confidence :
                        continue 
                    q =NOK [key ].get (en )
                    if q is None :
                        continue 
                    P [i ]=q ;na +=1 
                    j =AK .index (key )if key in AK else -1 
                    if j >=0 :
                        kotu_vur +=int (AY [j ]==1 );iyi_vur +=int (AY [j ]==0 )
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg ,na ,kotu_vur ,iyi_vur 

    AKi ={k :j for j ,k in enumerate (AK )}
    AK =list (AK )
    r0 ,d0 ,gg ,*_ =kos (2.0 ,2.0 )
    b_rob ,b_det =f1w (r0 ),f1w (d0 )
    print (f"\ntaban: robot(FIZ) {b_rob :.4f} | tespit {b_det :.4f}")
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    print (f"\n{'esikQ':>7}{'confidence':>7}{'ates':>6}{'kotuya':>8}{'iyiye':>7}"
    f"{'robot':>10}{'d':>9}{'GA':>22}{'tespit':>10}{'d':>9}")
    SON ={}
    for q in (0.70 ,0.80 ,0.85 ,0.90 ):
        threshold =float (np .quantile (oofA ,q ))
        # GUVEN ESIGI DUSURULDU: kosullu lower kumede pozitif ratio %13.3 oldugu for
        # skorlar 0.5'e never ulasmiyordu. 0.00 = "1. stage atesleyince KOSULSUZ argmax"
        # ki this mimarinin MUTLAK TAVANIDIR. Bunda bile kaybediyorsa arm kapanir.
        for confidence in (0.0 ,0.15 ,0.25 ,0.35 ):
            ra ,da ,_ ,na ,kv ,iv =kos (threshold ,confidence )
            _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (r0 ,ra )),gg ,fn ,n =1500 )
            dr =f1w (ra )-b_rob 
            print (f"{q :>7.2f}{confidence :>7.2f}{na :>6}{kv :>8}{iv :>7}{f1w (ra ):>10.4f}{dr :>+9.4f}"
            f"   [{lo :+.4f},{hi :+.4f}]{f1w (da ):>10.4f}{f1w (da )-b_det :>+9.4f}")
            SON [f"{q }_{confidence }"]={"esikQ":q ,"confidence":confidence ,"ates":na ,
            "kotuya":kv ,"iyiye":iv ,"robot":f1w (ra ),
            "d_robot":dr ,"ga":[lo ,hi ],"tespit":f1w (da ),
            "d_tespit":f1w (da )-b_det }
    eniyi =max (SON .values (),key =lambda v :v ["d_robot"])
    print (f"\nTARAMA en iyisi (UST SINIR, secim yanlisi var): robot {eniyi ['d_robot']:+.4f} "
    f"(threshold %{100 *eniyi ['esikQ']:.0f}, confidence {eniyi ['confidence']}) "
    f"GA[{eniyi ['ga'][0 ]:+.4f},{eniyi ['ga'][1 ]:+.4f}] | tespit {eniyi ['d_tespit']:+.4f}")
    val_ =eniyi ["d_robot"]>=0.01 and eniyi ["ga"][0 ]>0 and eniyi ["d_tespit"]>=-0.005 
    print (f"\nHUKUM: {'UST SINIR bile gecerse training-korpusu turetmesi DEGER'if val_ else 'UST SINIR bile GECMIYOR -- KONUM KOLU KAPANIR'}")
    with io .open ("results/r7b_konum_kosullu.json","w",encoding ="utf-8")as f :
        json .dump ({"taban_robot":b_rob ,"taban_tespit":b_det ,"auc_asama1":aucA ,
        "auc_asama2_kosullu":aucB ,"tarama":SON ,"en_iyi":eniyi ,
        "deger":bool (val_ )},f ,indent =1 )
    print ("receipt -> results/r7b_konum_kosullu.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
