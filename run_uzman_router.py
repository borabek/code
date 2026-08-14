# -*- coding: utf-8 -*-
"""Y2 — YOGUN-PARCA UZMANI + YONLENDIRICI (router)

WHY. Cok-CP'li parts TANIDIK markada bile zayif halka:
`cok_CP_F1 0.6583` vs `dusuk_CP_F1 0.7701` (cp_config). Gorulmemis markada
whereas ucurum: NIT uctan uca 0.0089.

BU KOL DAHA ONCE KAPANAN HIGH-CP KOLLARINDAN FARKLIDIR:
  * `high-cp-segmentation-closed`  -> SEGMENTASYON duzeyinde bolmeydi
  * `h3-highcp-finetune-dead`      -> same modelin FINETUNE'u
  * `cc-d-minv10-regime-split`     -> ESIK/parametre rejimi
  * `pitstop-highcp-gate`          -> GATE rejimi
Burada AYRI EGITILMIS BIR SECICI + OGRENILMIS YONLENDIRICI present. Uzman,
only dense parcalarda egitilir; so loss fonksiyonunu sparse
parcalarla paylasmaz.

REJIM TANIMI `headline.py` with AYNI: `n_gt >= 8` -> "very", degilse "low".
Ama YONLENDIRICI GT KULLANAMAZ; part duzeyi ozniteliklerden rejimi TAHMIN
eder and dogruluğu also raporlanir (wrong yonlendirme kolun tavanini
belirler).

KOLLAR:
  baseline    : single genel model (bugunku)
  uzman_gt : regime GT'den bilinseydi (UST SINIR -- dagitilamaz, ceiling olcer)
  uzman    : regime YONLENDIRICIDEN (dagitilabilir hali)

KAPI: uctan uca +0.01. D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KUME =os .environ .get ("UR_KUME","d6")
NEG_KAT =int (os .environ .get ("UR_NEG","12"))
COK_ESIK =int (os .environ .get ("UR_COK","8"))# headline with AYNI
ITER ,LR ,YAPRAK ,L2R =200 ,0.06 ,63 ,1.0 
NMS =5.0 
KURAL =("goreli",0.85 ,0.20 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def parca_oz (d ):
    """YONLENDIRICI for part duzeyi feature. GT KULLANMAZ."""
    idx =np .asarray (d ["idx"],int )
    P =np .asarray (d ["P"],float )
    n_kon =len (np .unique (idx ))
    kut =P .max (0 )-P .min (0 )if len (P )else np .zeros (3 )
    return np .array ([len (idx ),n_kon ,float (d ["diag"]),
    kut [0 ],kut [1 ],kut [2 ],
    float (np .prod (np .sort (kut )[-2 :])),
    n_kon /max (float (d ["diag"]),1e-6 )],float )


def yap ():
    return HistGradientBoostingClassifier (
    max_iter =ITER ,learning_rate =LR ,max_leaf_nodes =YAPRAK ,
    l2_regularization =L2R ,random_state =0 )


def egit (data_ ,ic ,cok_agirlik =1.0 ):
    """cok_agirlik > 1 whereas COK-CP parcalarindan gelen orneklerin agirligi
    artirilir. Y23: veriyi BOLMEK instead of AGIRLIKLANDIRMAK. Uzman kolu
    kahin rejimle bile dustu (-0.0418) and sebebi data parcalanmasiydi;
    agirliklandirma parcalamaz."""
    n_s =sum (len (data_ [i ]["y"])for i in ic )
    if not n_s :
        return None 
    M =np .empty ((n_s ,data_ [ic [0 ]]["_M"].shape [1 ]),np .float32 )
    o =0 
    for i in ic :
        m_ =data_ [i ]["_M"]
        M [o :o +len (m_ )]=m_ 
        o +=len (m_ )
    Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
    W =np .concatenate ([np .full (len (data_ [i ]["y"]),
    cok_agirlik if data_ [i ]["_cok"]else 1.0 ,
    np .float32 )for i in ic ])
    if Y .sum ()==0 or Y .sum ()==len (Y ):
        return None 
    rng =np .random .default_rng (0 )
    poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
    sec =np .concatenate ([poz ,rng .choice (
    neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
    m =yap ().fit (M [sec ],Y [sec ],sample_weight =W [sec ])
    del M 
    return m 


def ince_ayar (genel_veri ,ic_cok ,data_ ,ek_agac =60 ):
    """UZMANLASMA, VERI PARCALAMADAN.

    Uzman kolu kahin rejimle bile dustu (-0.0418) and reason teshis edildi:
    lower kumede SIFIRDAN egitmek each uzmani more few veriyle birakiyor.
    Duzeltme: genel modelden DEVAM ET (`warm_start`) and only dense
    parcalarda EK AGAC ekle. Boylece uzman TUM verinin bilgisiyle baslar,
    after dense rejime uyarlanir.

    NOT: kayittaki "high-CP finetune olu" SEGMENTASYON agi icindi; this
    SECICI tarafinda ayri a mekanizma.
    """
    if genel_veri is None or not ic_cok :
        return None 
    import copy 
    m =copy .deepcopy (genel_veri )
    n_s =sum (len (data_ [i ]["y"])for i in ic_cok )
    if not n_s :
        return None 
    M =np .empty ((n_s ,data_ [ic_cok [0 ]]["_M"].shape [1 ]),np .float32 )
    o =0 
    for i in ic_cok :
        mm =data_ [i ]["_M"]
        M [o :o +len (mm )]=mm 
        o +=len (mm )
    Y =np .concatenate ([data_ [i ]["y"]for i in ic_cok ])
    if Y .sum ()==0 or Y .sum ()==len (Y ):
        return None 
    rng =np .random .default_rng (0 )
    poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
    sec =np .concatenate ([poz ,rng .choice (
    neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
    # SESSIZ NO-OP DUZELTMESI (2026-08-13). Ilk yazim `m.max_iter_`
    # kullaniyordu -- boyle a feature YOK (dogrusu `n_iter_`).
    # AttributeError, genis a `except` by yutuluyor and fonksiyon
    # None donuyordu; two arm da sessizce TABANA dusup full +0.0000 veriyordu.
    # "Ince ayar whereas yaramiyor" diye rapor edilecekti. Artik error YUTULMAZ.
    m .set_params (warm_start =True ,max_iter =int (m .n_iter_ )+ek_agac )
    m .fit (M [sec ],Y [sec ])
    del M 
    assert int (m .n_iter_ )>ek_agac //2 ,"ince ayar agac EKLEMEDI"
    return m 


def f1 (c ):
    return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
        d ["_cok"]=int (len (d ["G"])>=COK_ESIK )
    n_cok =sum (d ["_cok"]for d in data_ )
    print (f"{len (data_ )} part | cok-CP {n_cok } ({n_cok /len (data_ ):.1%}) "
    f"| threshold n_gt>={COK_ESIK }",flush =True )

    # TANIDIK MARKA kosulu: rastgele 3 fold (brand-KARISIK)
    # KAT TOHUMU: very-CP agirligi 2x single tohumda +0.0065 verdi; fold
    # gurultusu +-0.008 oldugu for very tohumlu dogrulama SART.
    rng =np .random .default_rng (int (os .environ .get ("UR_TOHUM","1")))
    pay =rng .permutation (len (data_ ))%3 
    KOLLAR =("baseline","uzman_gt","uzman","agirlik2","agirlik4",
    "agirlik8","ince_ayar_gt","ince_ayar")
    agg ={k :collections .Counter ()for k in KOLLAR }
    yon_dogru =yon_top =0 
    for f_ in range (3 ):
        ic =[i for i in range (len (data_ ))if pay [i ]!=f_ ]
        dis =[i for i in range (len (data_ ))if pay [i ]==f_ ]
        genel =egit (data_ ,ic )
        agirlikli ={a :egit (data_ ,ic ,cok_agirlik =a )for a in (2.0 ,4.0 ,8.0 )}
        uz_cok =egit (data_ ,[i for i in ic if data_ [i ]["_cok"]])
        ia_cok =ince_ayar (genel ,[i for i in ic if data_ [i ]["_cok"]],data_ )
        uz_dus =egit (data_ ,[i for i in ic if not data_ [i ]["_cok"]])
        # YONLENDIRICI: part duzeyi, GT'siz
        XR =np .vstack ([parca_oz (data_ [i ])for i in ic ])
        YR =np .asarray ([data_ [i ]["_cok"]for i in ic ],int )
        direction =yap ().fit (XR ,YR )if 0 <YR .sum ()<len (YR )else None 
        for i in dis :
            d =data_ [i ]
            tah =(int (direction .predict (parca_oz (d )[None ])[0 ])if direction is not None 
            else 0 )
            yon_dogru +=int (tah ==d ["_cok"])
            yon_top +=1 
            for arm in KOLLAR :
                if arm =="baseline":
                    m =genel 
                elif arm .startswith ("agirlik"):
                    m =agirlikli [float (arm [7 :])]
                elif arm .startswith ("ince_ayar"):
                    rej =(d ["_cok"]if arm .endswith ("_gt")else tah )
                    m =ia_cok if rej else genel 
                else :
                    rej =d ["_cok"]if arm =="uzman_gt"else tah 
                    m =uz_cok if rej else uz_dus 
                if m is None :
                    m =genel 
                s =m .predict_proba (d ["_M"])[:,1 ]
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,
                nms_mm =NMS )
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
                K .YANAL ,K .ACI ,False ,
                signed =True )[:3 ]
                c =agg [arm ]
                c ["tp"]+=tp ;c ["fp"]+=fp ;c ["fn"]+=fn 
        print (f"  fold{f_ } bitti ({time .time ()-t0 :.0f} s)",flush =True )

    last_ ={k :f1 (agg [k ])for k in KOLLAR }
    print (f"\nyonlendirici dogrulugu: {yon_dogru }/{yon_top } "
    f"({yon_dogru /max (yon_top ,1 ):.3f})")
    print (f"\n=== TABAN {last_ ['baseline']:.4f} (TANIDIK brand kosulu) ===")
    for k in KOLLAR [1 :]:
        fark =last_ [k ]-last_ ["baseline"]
        et ="  <- KAPI GECTI"if fark >=0.01 else ""
        ust ="  (UST SINIR, dagitilamaz)"if k =="uzman_gt"else ""
        print (f"  {k :<10}{last_ [k ]:.4f}   {fark :+.4f}{et }{ust }")
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,
    "cok_esik":COK_ESIK ,"n_cok":n_cok ,
    "yonlendirici_dogruluk":yon_dogru /max (yon_top ,1 ),
    "toplam":last_ ,
    "not":"Yogun-part UZMANI + ogrenilmis router. TANIDIK "
    "brand kosulu (rastgele katlar). uzman_gt UST SINIRDIR. "
    "D7'ye BAKILMADI."},
    open (f"results/uzman_router_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/uzman_router_{KUME }.json")


if __name__ =="__main__":
    main ()
