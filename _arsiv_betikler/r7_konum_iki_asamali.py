# -*- coding: utf-8 -*-
"""R7: IKI ASAMALI KONUM SECICISI -- FIZIBILITE (dagitim DEGIL).

R5: kurtarma 42 / bozma riski 857 = 1:20 asimetri.
R6-A: global tanim degisikligi YOK -- tezin v_o'this three rakibi de yendi (cember -0.0875,
      mouth -0.1001, open -0.5432). Yani arm however SECICI with yasar.
R6-B: "mevcut kotu mu" OGRENILEBILIR (grup-capraz AUC 0.8223; most high %10 skorda
      kotu orani %16.4 -> %59.6).

GERIYE KALAN BILINMEYEN: iyi a noktaya YANLISLIKLA ates ettigimizde secilen giris de
<=2mm cikiyor mu? Cikiyorsa hasar absent and arm yasar; cikmiyorsa 1:20 asimetri kolu oldurur.
Bu SADECE UCTAN UCA olculebilir.

BU BETIK DAGITMAZ. Olcum kumesinde GRUP-CAPRAZ (5 fold, geometri grubuyla) works and
uctan uca robot(FIZIKSEL)+tespit produces. Amaci TEK a soruyu yanitlamak:
"45 dakikalik training-korpusu turetmesini harcamaya value mi?"

TARAMA UYARISI: birden very (threshold, marj) ciftini AYNI measurement kumesinde deniyorum. En iyisi
a UST SINIRDIR, dagitilabilir prediction not. O yuzden ONCEDEN KAYITLI a ayar da
also raporlanir: threshold=%85 yuzdelik, marj=0.15 (direction kolunun 0.05'inden BUYUK secildi
because buradaki hasar asimetrisi 1:20).
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

ONB ="results/_r4_sozluk.pkl"
KAYITLI_ESIK_Q =0.85 # ONCEDEN secildi
KAYITLI_MARJ =0.15 # ONCEDEN secildi (direction kolu 0.05; here asimetri 1:20)
GIRIS =("mouth","cember","open","kirpik","axis",
"kesit1","kesit3","kesit5","kesit7","kesit10","acik3","acik5")


def dagilim_oz (S ,p ,d ):
    Q =np .array ([np .asarray (v ,float )for v in S .values ()],float )if S else np .zeros ((0 ,3 ))
    if len (Q ):
        off =Q -p 
        dp =np .linalg .norm (off -(off @d )[:,None ]*d ,axis =1 )
        oz =[len (Q ),float (dp .mean ()),float (dp .max ()),float (np .median (dp )),
        float (np .linalg .norm (Q .mean (0 )-p ))]
    else :
        oz =[0 ,0.0 ,0.0 ,0.0 ,0.0 ]
    return oz +[1.0 if a in S else 0.0 for a in GIRIS [:8 ]]


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    DER ,gate ,ek =T2 .yukle ()
    PARCA =pickle .load (open (ONB ,"rb"))
    YS =wire_gate ._load ("results/yon_secici.pkl")

    H ={}
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        H [r ["pid"]]=None if d_ is None else (d_ ["P"].copy (),yon_uygula (d_ ,YS ),
        d_ ["SK"],d_ ["X"])

        # ---------- satirlari kur (two stage single gecisde)
    AX ,AY ,AG ,AK =[],[],[],[]# stage 1: candidate duzeyi ("mevcut kotu mu")
    BX ,BY ,BG ,BK =[],[],[],[]# stage 2: (candidate, giris) duzeyi ("this giris <=2mm mi")
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
            AX .append (np .concatenate ([X [i ],oz ]));AY .append (int (pe [i ,b ]>2.0 ))
            AG .append (r ["geo"]);AK .append ((r ["pid"],i ))
            # giris 0 = mevcut, after dictionary
            lst_ =[("mevcut",P [i ])]+[(a ,np .asarray (S [a ],float ))for a in GIRIS if a in S ]
            for gi ,(ad ,q )in enumerate (lst_ ):
                w =q -G [b ]
                yy =float (np .linalg .norm (w -float (w @Gd [b ])*Gd [b ]))
                e =[1.0 if ad ==t2 else 0.0 for t2 in ("mevcut",)+GIRIS ]
                e +=[float (np .linalg .norm (q -P [i ])),
                float (np .linalg .norm ((q -P [i ])-float ((q -P [i ])@Pd [i ])*Pd [i ]))]
                BX .append (np .concatenate ([X [i ],oz ,e ]));BY .append (int (yy <=2.0 ))
                BG .append (r ["geo"]);BK .append ((r ["pid"],i ,ad ))
    AX =np .array (AX ,float );AY =np .array (AY );AG =np .array (AG )
    BX =np .array (BX ,float );BY =np .array (BY );BG =np .array (BG )
    print (f"asama1 {len (AY )} satir (kotu {AY .mean ():.1%}) | "
    f"asama2 {len (BY )} satir (iyi giris {BY .mean ():.1%})",flush =True )

    # ---------- grup-capraz OOF (AYNI katlar, geometri grubu)
    ug =np .unique (np .concatenate ([AG ,BG ]));rng =np .random .RandomState (0 );rng .shuffle (ug )
    fold ={g :j %5 for j ,g in enumerate (ug )}
    ka =np .array ([fold [g ]for g in AG ]);kb =np .array ([fold [g ]for g in BG ])
    oofA =np .zeros (len (AY ));oofB =np .zeros (len (BY ))
    for f_ in range (5 ):
        ta =ka !=f_ ;tb =kb !=f_ 
        cA =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (AX [ta ],AY [ta ])
        cB =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (BX [tb ],BY [tb ])
        oofA [~ta ]=cA .predict_proba (AX [~ta ])[:,1 ]
        oofB [~tb ]=cB .predict_proba (BX [~tb ])[:,1 ]
    print ("OOF hazir",flush =True )

    SA ={k :oofA [j ]for j ,k in enumerate (AK )}
    SB ={}
    for j ,(pid ,i ,ad )in enumerate (BK ):
        SB .setdefault ((pid ,i ),{})[ad ]=oofB [j ]
    NOK ={}
    for r in DER :
        h =H [r ["pid"]]
        if h is None :
            continue 
        P ,Pd ,SK ,X =h 
        for i in range (len (P )):
            S =SK [i ]if i <len (SK )else {}
            NOK [(r ["pid"],i )]={"mevcut":P [i ]}|{a :np .asarray (S [a ],float )
            for a in GIRIS if a in S }

    def kos (threshold ,marj ):
        rob ,det ,gg =[],[],[]
        n_ates =0 
        for r in DER :
            h =H [r ["pid"]]
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
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
                    if en =="mevcut"or sc [en ]-sc .get ("mevcut",0.0 )<marj :
                        continue 
                    q =NOK [key ].get (en )
                    if q is not None :
                        P [i ]=q ;n_ates +=1 
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg ,n_ates 

    r0 ,d0 ,gg ,_ =kos (2.0 ,9.0 )# no zaman atesleme = baseline
    b_rob ,b_det =f1w (r0 ),f1w (d0 )
    print (f"\ntaban: robot(FIZ) {b_rob :.4f} | tespit {b_det :.4f}")
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    print (f"\n{'esikQ':>7}{'marj':>7}{'ates':>7}{'robot':>10}{'d':>9}{'GA':>22}{'tespit':>10}{'d':>9}")
    SON ={}
    for q in (0.70 ,0.80 ,0.85 ,0.90 ,0.95 ):
        threshold =float (np .quantile (oofA ,q ))
        for marj in (0.05 ,0.15 ,0.25 ):
            ra ,da ,_ ,na =kos (threshold ,marj )
            _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (r0 ,ra )),gg ,fn ,n =1500 )
            dr =f1w (ra )-b_rob 
            im =" <-KAYITLI"if (abs (q -KAYITLI_ESIK_Q )<1e-9 and abs (marj -KAYITLI_MARJ )<1e-9 )else ""
            print (f"{q :>7.2f}{marj :>7.2f}{na :>7}{f1w (ra ):>10.4f}{dr :>+9.4f}"
            f"   [{lo :+.4f},{hi :+.4f}]{f1w (da ):>10.4f}{f1w (da )-b_det :>+9.4f}{im }")
            SON [f"{q }_{marj }"]={"esikQ":q ,"marj":marj ,"ates":na ,
            "robot":f1w (ra ),"d_robot":dr ,"ga":[lo ,hi ],
            "tespit":f1w (da ),"d_tespit":f1w (da )-b_det }
    kay =SON [f"{KAYITLI_ESIK_Q }_{KAYITLI_MARJ }"]
    eniyi =max (SON .values (),key =lambda v :v ["d_robot"])
    print (f"\nKAYITLI ayar (threshold %{100 *KAYITLI_ESIK_Q :.0f}, marj {KAYITLI_MARJ }): "
    f"robot {kay ['d_robot']:+.4f} GA[{kay ['ga'][0 ]:+.4f},{kay ['ga'][1 ]:+.4f}] | "
    f"tespit {kay ['d_tespit']:+.4f}")
    print (f"TARAMA en iyisi (UST SINIR): robot {eniyi ['d_robot']:+.4f} "
    f"(threshold %{100 *eniyi ['esikQ']:.0f}, marj {eniyi ['marj']})")
    val_ =kay ["d_robot"]>=0.01 and kay ["ga"][0 ]>0 and kay ["d_tespit"]>=-0.005 
    print (f"\nHUKUM: 45 dakikalik training-korpusu turetmesi "
    f"{'DEGER -- R8 kosulur'if val_ else 'DEGMEZ -- KOL KAPANIR'}")
    with io .open ("results/r7_konum_iki_asamali.json","w",encoding ="utf-8")as f :
        json .dump ({"taban_robot":b_rob ,"taban_tespit":b_det ,"tarama":SON ,
        "kayitli":kay ,"tarama_en_iyi":eniyi ,"value":bool (val_ )},f ,indent =1 )
    print ("receipt -> results/r7_konum_iki_asamali.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
