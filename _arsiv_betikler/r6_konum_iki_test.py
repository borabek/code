# -*- coding: utf-8 -*-
"""R6: KONUM KOLUNUN IKI AYRI SORUSU -- before UCUZ olani.

R5 TESHISI: kurtarma adayi 42, bozma riski tasiyan gecen point 857. 1:20 ASIMETRI.
Yanlis atesleyen a selector kurtardiginin yirmi katini kirar. O yuzden this kolun sorusu
"hangi giris more iyi" DEGIL.

IKI AYRI SORU VAR and BIRINCISI SECICI GEREKTIRMIYOR:

  A) TANIM SORUSU (ucuz, ogrenme absent, tez-sadik):
     Tez v_o'yu "mouth boundary noktalarinin ortalamasi" diye tanimlar. [[brep-radius-arc-bug]]
     kaydi same ailedeki yanliligi olcmustu: kismi ornekleme varsa ORTALAMA full tarafa
     kayar, cember oturtma sart. `cember` (most small kareler centre) and `open` (mouth
     duzleminin Chebyshev merkezi = telin gecebilecegi most genis yer) AYNI BUYUKLUGUN
     more iyi kestiricileri -- new a kavram not. Eger biri GLOBAL as more iyiyse
     arm here biter: selector absent, kumar absent, single satirlik tanim degisikligi.
     R5'te kurtaranlarin basinda `open` (13/42) and `kesit*` (21/42) vardi -- but bunlar
     KURTARDIKLARI places sayildi; BOZDUKLARI places sayilmadi. Net etki here olculur.

  B) ATESLEME SORUSU (pahali, ogrenme present):
     Global kazanan otherwise selector sart; and seciciyi mesru kilan TEK sey "mevcut point
     kotu mu"yu bilebilmektir. Onu bilmiyorsak argmax 1:20 kumarina girer. GRUP-CAPRAZ
     AUC with olculur. AUC<0.65 whereas arm KAPANIR and 45 dakikalik turetme YAPILMAZ.

Bu betik HICBIR SEY DAGITMAZ.
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


def main ():
    DER ,gate ,ek =T2 .yukle ()
    PARCA =pickle .load (open (ONB ,"rb"))
    YS =wire_gate ._load ("results/yon_secici.pkl")

    # ---- hazirlik: each part for P, Pd(direction secicili), SK
    H ={}
    adlar ={}
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        if d_ is None :
            H [r ["pid"]]=None 
            continue 
        H [r ["pid"]]=(d_ ["P"].copy (),yon_uygula (d_ ,YS ),d_ ["SK"],d_ ["X"])
        for S in d_ ["SK"]:
            for a in S :
                adlar [a ]=adlar .get (a ,0 )+1 
    print (f"sozluk girisleri: {dict (sorted (adlar .items (),key =lambda x :-x [1 ]))}\n")

    def kos (sel_ ):
        """secim(i, S, p) -> point. Robot(FIZIKSEL) + tespit returns."""
        rob ,det ,gg =[],[],[]
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
                    q =sel_ (i ,SK [i ]if i <len (SK )else {},P0 [i ])
                    if q is not None :
                        P [i ]=q 
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg 

        # ================= A) TANIM SORUSU =================
    print ("="*74 )
    print ("A) TANIM: each girisi GLOBAL as v_o instead of koy (ogrenme YOK)")
    print ("="*74 )
    r0 ,d0 ,gg =kos (lambda i ,S ,p :None )
    b_rob ,b_det =f1w (r0 ),f1w (d0 )
    print (f"{'giris':<14}{'robot(FIZ)':>12}{'d':>9}{'GA':>22}{'tespit':>10}{'d':>9}{'kapsam':>9}")
    print (f"{'TABAN (v_o)':<14}{b_rob :>12.4f}{'':>9}{'':>22}{b_det :>10.4f}")
    A ={}
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    for a in sorted (adlar ,key =lambda x :-adlar [x ]):
        if a =="mevcut":
            continue 
        n_uyg =[0 ]

        def sec (i ,S ,p ,_a =a ,_n =n_uyg ):
            if _a in S :
                _n [0 ]+=1 
                return np .asarray (S [_a ],float )
            return None 
        ra ,da ,_ =kos (sec )
        _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (r0 ,ra )),gg ,fn ,n =1500 )
        gerc ="GERCEK"if (lo >0 or hi <0 )else ""
        A [a ]={"robot":f1w (ra ),"d_robot":f1w (ra )-b_rob ,"ga":[lo ,hi ],
        "tespit":f1w (da ),"d_tespit":f1w (da )-b_det ,"n":n_uyg [0 ]}
        print (f"{a :<14}{f1w (ra ):>12.4f}{f1w (ra )-b_rob :>+9.4f}"
        f"   [{lo :+.4f},{hi :+.4f}] {gerc :<6}"
        f"{f1w (da ):>10.4f}{f1w (da )-b_det :>+9.4f}{n_uyg [0 ]:>9}")

    kazanan =max (A ,key =lambda a :A [a ]["d_robot"])if A else None 
    A_gecti =bool (kazanan and A [kazanan ]["d_robot"]>=0.01 and A [kazanan ]["ga"][0 ]>0 
    and A [kazanan ]["d_tespit"]>=-0.005 )
    print (f"\nA HUKMU: en iyi giris '{kazanan }' ({A [kazanan ]['d_robot']:+.4f}) -> "
    f"{'GLOBAL TANIM DEGISIKLIGI GECERLI'if A_gecti else 'global kazanan YOK, B sart'}")

    # ================= B) ATESLEME SORUSU =================
    print ("\n"+"="*74 )
    print ("B) ATESLEME: 'mevcut point kotu mu' OGRENILEBILIR mi (grup-capraz AUC)")
    print ("="*74 )
    FX ,FY ,FG =[],[],[]
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
            # SOZLUK DAGILIMI = merkezleme belirsizliginin fiziksel vekili:
            # girisler each other yakinsa mouth iyi tanimli, saciliyorsa ambiguous.
            Q =np .array ([np .asarray (v ,float )for v in S .values ()],float )if S else np .zeros ((0 ,3 ))
            if len (Q ):
                off =Q -P [i ]
                dperp =np .linalg .norm (off -(off @Pd [i ])[:,None ]*Pd [i ],axis =1 )
                oz =[len (Q ),float (dperp .mean ()),float (dperp .max ()),float (np .median (dperp )),
                float (np .linalg .norm (Q .mean (0 )-P [i ]))]
            else :
                oz =[0 ,0.0 ,0.0 ,0.0 ,0.0 ]
            oz +=[1.0 if a in S else 0.0 for a in 
            ("mouth","cember","acik","kirpik","axis","kesit1","kesit3","kesit5")]
            FX .append (np .concatenate ([X [i ],oz ]))
            FY .append (int (pe [i ,b ]>2.0 ))
            FG .append (r ["geo"])
    FX =np .array (FX ,float );FY =np .array (FY );FG =np .array (FG )
    print (f"satir {len (FY )} | 'mevcut KOTU' orani {FY .mean ():.1%} | sutun {FX .shape [1 ]}")

    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .metrics import roc_auc_score 
    ug =np .unique (FG );rng =np .random .RandomState (0 );rng .shuffle (ug )
    fold ={g :j %5 for j ,g in enumerate (ug )}
    kk =np .array ([fold [g ]for g in FG ])
    oof =np .zeros (len (FY ))
    for f_ in range (5 ):
        tr =kk !=f_ 
        c =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (FX [tr ],FY [tr ])
        oof [~tr ]=c .predict_proba (FX [~tr ])[:,1 ]
    auc =float (roc_auc_score (FY ,oof ))if len (np .unique (FY ))>1 else 0.5 
    print (f"GRUP-CAPRAZ AUC ('mevcut kotu mu'): {auc :.4f}")
    # kaç noktayi guvenle ateslemeye ayirabiliyoruz?
    for q in (0.90 ,0.95 ,0.99 ):
        t =float (np .quantile (oof ,q ))
        sel =oof >=t 
        print (f"  en yuksek %{100 *(1 -q ):.0f} skor: {int (sel .sum ())} nokta, "
        f"gercekten kotu olan {FY [sel ].mean ():.1%} (baseline {FY .mean ():.1%})")
    B_gecti =auc >=0.65 
    print (f"\nB HUKMU: AUC {auc :.4f} -> {'ATESLEME OGRENILEBILIR'if B_gecti else 'ATESLEME OGRENILEMEZ -- KOL KAPANIR'}")

    json .dump ({"taban_robot":b_rob ,"taban_tespit":b_det ,"global":A ,
    "A_kazanan":kazanan ,"A_gecti":A_gecti ,
    "atesleme_auc":auc ,"kotu_orani":float (FY .mean ()),
    "B_gecti":bool (B_gecti ),"n_satir":int (len (FY ))},
    io .open ("results/r6_konum_iki_test.json","w"),indent =1 )
    print ("\nmakbuz -> results/r6_konum_iki_test.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
