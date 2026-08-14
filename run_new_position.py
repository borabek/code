# -*- coding: utf-8 -*-
"""L3/L4/L5 SURUCUSU — new konum bilgisi, UCTAN UCA

Taban = temel + kanonik + neg12 (bugun dogrulanan yigin, d6 0.3135).
Kollar: ayna / vida / temas / all of them
KAPI: uctan uca mikro robot F1'de +0.01. Gecen arm `full`'da dogrulanir.
D7'ye BAKILMAZ.
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
import connector3d # noqa: E402
import canonical_d7 as K # noqa: E402
import kanonik_hizalama as KH # noqa: E402
import p6_decision # noqa: E402
import new_position_oz as YKO # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KUME =os .environ .get ("YK_KUME","d6")
KAT_MIN =int (os .environ .get ("YK_KAT_MIN","40"))
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
NEG_KAT =int (os .environ .get ("YK_NEG","12"))
ITER ,LR ,YAPRAK ,L2R =200 ,0.06 ,63 ,1.0 
NMS =5.0 
KURAL =("goreli",0.85 ,0.20 )
CT =int (connector3d .CONTACT )
if os .environ .get ("YK_MOD")=="2":
# IKINCI TUR. Birinci turda ayna esi +0.0075 with most guclu new sinyal
# became but +0.01 kapisini GECEMEDI. Kapiyi indirmek sisirmedir; mesru
# which is, kapiyi degistirmeden BIRLESIM denemektir. Ayna (feature) with
# yerel negatif (ornekleme) FARKLI mekanizmalar -- toplamlari kapiyi
# gecebilir. `temas` de eklenir (+0.0043, ucuncu bagimsiz mekanizma).
    KOLLAR =("baseline","ayna","ayna_yerelneg","ayna_temas",
    "ayna_temas_yerelneg")
else :
    KOLLAR =("baseline","ayna","vida","temas","hepsi")
DILIM ={"ayna":(0 ,5 ),"vida":(5 ,9 ),"temas":(9 ,13 ),
"ayna_yerelneg":(0 ,5 )}


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (
    np .float32 )


def f1 (c ):
    return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )


def yap ():
    return HistGradientBoostingClassifier (
    max_iter =ITER ,learning_rate =LR ,max_leaf_nodes =YAPRAK ,
    l2_regularization =L2R ,random_state =0 )


def main ():
    t0 =time .time ()
    import trimesh 
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    n_ok =0 
    for _i ,d in enumerate (data_ ):
        d ["y"]=np .asarray (d ["y"],int )
        idx =np .asarray (d ["idx"],int )
        P =np .asarray (d ["P"],float )
        YD =np .asarray (d ["YD"],float )
        tek ,ters =np .unique (idx ,return_inverse =True )
        Pk =P [tek ]
        Dk =YKO .konum_yonu (idx ,YD ,tek )
        mf =f"{MESH }/{d ['pid']}.npz"
        if os .path .exists (mf ):
            z =np .load (mf )
            V =np .asarray (z ["V"],float )
            F =np .asarray (z ["F"],int )
            pb =np .mean ([np .asarray (q ,float )for q in z ["pbs"]],axis =0 )
            ag =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
            isinci =trimesh .ray .ray_triangle .RayMeshIntersector (ag )
            tm =YKO .isin_temas (Pk ,Dk ,isinci ,V ,pb [:,CT ])
            n_ok +=1 
        else :
            tm =np .zeros ((len (Pk ),4 ),np .float32 )
        blok =np .hstack ([YKO .ayna_esi (Pk ,Dk ),YKO .vida_cifti (Pk ,Dk ),tm ])
        d ["_M"]=temel (d )
        d ["_K"]=KH .oznitelik (P [idx ],YD ,
        V if os .path .exists (mf )else P ).astype (
        np .float32 )
        d ["_B"]=blok [ters ].astype (np .float32 )
        # `data.index(d)` KULLANILMAZ: O(n^2) and dictionary inside numpy dizisi
        # oldugu for karsilastirmasi da pahalidir.
        if (_i +1 )%100 ==0 :
            print (f"  oznitelik {_i +1 }/{len (data_ )} "
            f"({time .time ()-t0 :.0f} s)",flush =True )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    # KAT TURU (2026-08-13). Varsayilan MARKA-DISI katlar = GORULMEMIS
    # brand kosulu. `YK_RASTGELE_KAT=1` with RASTGELE 3 fold is used =
    # brand-KARISIK, i.e. TANIDIK brand kosulu.
    # WHY: this kollar (ayna esi, yerel negatif, array uyeligi, isin-temas)
    # gorulmemis markada measured and kapiyi gecemedi. Tanidik markada temsil
    # problemi very more small oldugu for AYNI kollar tutabilir; bunu
    # olcmeden "olu" saymak, kapatma hukmunu sondanin kosuluna kurban
    # etmek becomes.
    if os .environ .get ("YK_RASTGELE_KAT")=="1":
        _rng =np .random .default_rng (1 )
        _pay =_rng .permutation (len (data_ ))%3 
        for _i ,_d in enumerate (data_ ):
            _d ["mfg"]=f"fold{_pay [_i ]}"
        brand =collections .Counter (d ["mfg"]for d in data_ )
        katlar =[f"fold{i }"for i in range (3 )]
        print ("  KATLAR RASTGELE (brand-karisik = TANIDIK brand kosulu)",
        flush =True )
    else :
        katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | mesh {n_ok } | katlar {katlar } | "
    f"neg={NEG_KAT } ({time .time ()-t0 :.0f} s)",flush =True )

    def mat (d ,arm ):
        if arm =="baseline":
            return np .hstack ([d ["_M"],d ["_K"]])
        if arm =="hepsi":
            return np .hstack ([d ["_M"],d ["_K"],d ["_B"]])
        if arm in ("ayna_temas","ayna_temas_yerelneg"):
            return np .hstack ([d ["_M"],d ["_K"],d ["_B"][:,0 :5 ],
            d ["_B"][:,9 :13 ]])
        a ,b =DILIM [arm ]
        return np .hstack ([d ["_M"],d ["_K"],d ["_B"][:,a :b ]])

    agg ={k :collections .Counter ()for k in KOLLAR }
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        for arm in KOLLAR :
            n_s =sum (len (data_ [i ]["y"])for i in ic )
            M =np .empty ((n_s ,mat (data_ [ic [0 ]],arm ).shape [1 ]),np .float32 )
            PA =np .empty (n_s ,np .int32 )
            o =0 
            for pi ,i in enumerate (ic ):
                m_ =mat (data_ [i ],arm )
                M [o :o +len (m_ )]=m_ 
                PA [o :o +len (m_ )]=pi 
                o +=len (m_ )
            Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
            rng =np .random .default_rng (0 )
            poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
            if arm .endswith ("yerelneg"):
            # PARCA-YEREL NEGATIF (L1a, single basina +0.0034): negatifler
            # pozitifle AYNI parcadan. Toplam size kuresel ornekleme
            # with AYNI kalir, i.e. kiyas single degiskenli.
                pl =[poz ]
                for pj in np .unique (PA [poz ]):
                    yer =np .where (PA ==pj )[0 ]
                    ng =yer [Y [yer ]==0 ]
                    n_al =min (len (ng ),NEG_KAT *int ((Y [yer ]==1 ).sum ()))
                    if n_al :
                        pl .append (rng .choice (ng ,n_al ,replace =False ))
                sec =np .concatenate (pl )
            else :
                sec =np .concatenate ([poz ,rng .choice (
                neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),
                replace =False )])
            m =yap ().fit (M [sec ],Y [sec ])
            del M ,PA 
            for i in dis :
                d =data_ [i ]
                s =m .predict_proba (mat (d ,arm ))[:,1 ]
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,
                nms_mm =NMS )
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
                K .YANAL ,K .ACI ,False ,
                signed =True )[:3 ]
                c =agg [arm ]
                c ["tp"]+=tp ;c ["fp"]+=fp ;c ["fn"]+=fn 
        print (f"  {b } bitti ({time .time ()-t0 :.0f} s)",flush =True )

    last_ ={k :f1 (agg [k ])for k in KOLLAR }
    print (f"\n=== TABAN {last_ ['baseline']:.4f} ===")
    for k in KOLLAR [1 :]:
        fark =last_ [k ]-last_ ["baseline"]
        print (f"  {k :<8}{last_ [k ]:.4f}   {fark :+.4f}"
        +("  <- KAPI GECTI"if fark >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,"neg":NEG_KAT ,
    "toplam":last_ ,
    "not":"L3 ayna esi + L4 vida cifti + L5 isin-temas. Taban = "
    "temel + kanonik + neg12. Hepsi GT'siz. "
    "D7'ye BAKILMADI."},
    open (f"results/yeni_konum_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/yeni_konum_{KUME }.json")


if __name__ =="__main__":
    main ()
