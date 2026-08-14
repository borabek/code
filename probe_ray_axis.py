# -*- coding: utf-8 -*-
"""A1 SONDASI -- ISIN ATMA with YON, GERCEK VERIDE

SENTETIK YETENEK GECTI (`ray_axis.self_check`): ekseni known three
delikte deviation 0.0 derece, tup skoru 8.45 / zemin 1.00.

DIAGNOSIS (docs/autopsy_dense_part.md). NIT'te 6322 option = ~260 konum x 24
direction. Konum/GT orani 11:1 (iyi); fazlaligin TAMAMI direction coklugundan.
Yon konum basina TEK olsaydi gereken AUC 0.9968 -> 0.91.

ESLESME KUTUSU -- BUGUN IKI KEZ DUSTUGUM TRAP. Aday-GT eslesmesi OKLID
mesafesiyle YAPILMAZ. Kabul kutusu carpimdir: GT yonune according to YANAL <= 2mm
and EKSENEL <= 40mm. Oklid 2mm kullanmak `oracle`i 0.593'ten 0.0172'ye
dusuruyordu -- measured_path sey mekanizma not kusurdu.

KOLLAR (GT with matched adaylarda, direction YALITILMIS):
  bugunku    : model skoru most high option
  tup        : tup skoru most high option (own isaretiyle)
  tup_disari : tup most high EKSEN + sign "govdeden disari"
  tup_x_skor : tup x model skoru
  oracle      : correct direction candidates between VAR mi (upper boundary)

KAPI: NIT'te `tup` or `tup_disari`, `bugunku`yu >= 0.05 asacak.
D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import ray_axis as IE # noqa: E402
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402

KUME =os .environ .get ("IS_KUME","d6")
KAT_MIN =int (os .environ .get ("IS_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
MAKS_KONUM =int (os .environ .get ("IS_KONUM","3"))# GT basina konum tavani
MAKS_YON =int (os .environ .get ("IS_YON","24"))# konum basina direction tavani
KOLLAR =("bugunku","tup","tup_eksen","tup_hava","tup_disari",
"tup_x_skor","oracle")


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def kutu_maskesi (Pk ,g ,gn ,lateral =None ,axial =None ):
    """CARPIM kabul kutusu: GT yonune according to lateral and axial AYRI."""
    lateral =K .YANAL if lateral is None else lateral 
    axial =40.0 if axial is None else axial 
    v =Pk -g [None ,:]
    al =v @gn 
    yan =np .linalg .norm (v -al [:,None ]*gn [None ,:],axis =1 )
    return (yan <=lateral )&(np .abs (al )<=axial )


def main ():
    t0 =time .time ()
    import trimesh 
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } | konum<= {MAKS_KONUM } "
    f"direction<= {MAKS_YON }",flush =True )

    oof =[None ]*len (data_ )
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        n_s =sum (len (data_ [i ]["y"])for i in ic )
        M =np .empty ((n_s ,data_ [0 ]["_M"].shape [1 ]),np .float32 )
        o =0 
        for i in ic :
            m_ =data_ [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        for i in dis :
            oof [i ]=m .predict_proba (data_ [i ]["_M"])[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    skipped =collections .Counter ()
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        mf =f"{MESH }/{d ['pid']}.npz"
        if not os .path .exists (mf ):
            skipped ["mesh_yok"]+=1 
            continue 
        z =np .load (mf )
        V ,F =np .asarray (z ["V"],float ),np .asarray (z ["F"],int )
        if len (F )<100 :
            skipped ["mesh_kucuk"]+=1 
            continue 
        ag =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
        isinci =trimesh .ray .ray_triangle .RayMeshIntersector (ag )
        center_ =V .mean (0 )

        G =np .asarray (d ["G"],float )
        Gn =_birim (np .asarray (d ["Gd"],float ))
        P =np .asarray (d ["P"],float )
        idx =np .asarray (d ["idx"],int )
        YD =_birim (np .asarray (d ["YD"],float ))
        s =np .asarray (s ,float )
        Pk =P [idx ]

        say ={k :0 for k in KOLLAR }
        for j in range (len (G )):
            m_ =kutu_maskesi (Pk ,G [j ],Gn [j ])
            if not m_ .any ():
                continue 
            se =np .where (m_ )[0 ]
            # GT basina at most MAKS_KONUM konum (lateral mesafeye according to)
            kon =np .unique (idx [se ])
            if len (kon )>MAKS_KONUM :
                dk =np .linalg .norm (P [kon ]-G [j ],axis =1 )
                kon =kon [np .argsort (dk )[:MAKS_KONUM ]]
            se =se [np .isin (idx [se ],kon )]
            if len (se )>MAKS_YON *MAKS_KONUM :
                se =se [np .argsort (-s [se ])[:MAKS_YON *MAKS_KONUM ]]

            aci =np .degrees (np .arccos (np .clip (YD [se ]@Gn [j ],-1 ,1 )))
            if (aci <=K .ACI ).any ():
                say ["oracle"]+=1 
            if aci [int (np .argmax (s [se ]))]<=K .ACI :
                say ["bugunku"]+=1 

                # --- TUP SKORU: each konum for own kokunden
            tup =np .zeros (len (se ))
            for ki in np .unique (idx [se ]):
                yer =np .where (idx [se ]==ki )[0 ]
                tup [yer ]=IE .tup_skoru (isinci ,P [ki ],YD [se ][yer ])
            if aci [int (np .argmax (tup ))]<=K .ACI :
                say ["tup"]+=1 
            karma =tup *s [se ]
            if aci [int (np .argmax (karma ))]<=K .ACI :
                say ["tup_x_skor"]+=1 

                # --- DIAGNOSIS: EKSEN mi wrong, ISARET mi?
                # `tup` rastgeleden (1/24 ~ 0.04) bile kotu ciktiysa this tesaduf
                # not SISTEMATIK TERS ISARETTIR: agizda dururken tup ICERI
                # uzanir, GT yonu whereas DISARI bakar. Asagidaki `tup_eksen`
                # ISARETSIZ acidir; yuksekse axis correct, sorun only isarettir.
            en =int (np .argmax (tup ))
            e =YD [se ][en ]
            eks_aci =np .degrees (np .arccos (
            np .clip (abs (float (e @Gn [j ])),-1 ,1 )))
            if eks_aci <=K .ACI :
                say ["tup_eksen"]+=1 

                # --- ISARET KURALI 1: HAVA TARAFI (isin verisinden dogrudan)
                # Disari which is taraf havaya produced taraftir: that yonde isin no
                # seye carpmaz (serbest path large), ic tarafta duvara carpar.
            kok =P [idx [se ][en ]]
            d_ci =IE .ilk_carpma (isinci ,np .vstack ([kok ,kok ]),
            np .vstack ([e ,-e ]))
            v =e if d_ci [0 ]>=d_ci [1 ]else -e 
            if np .degrees (np .arccos (
            np .clip (float (v @Gn [j ]),-1 ,1 )))<=K .ACI :
                say ["tup_hava"]+=1 

                # --- ISARET KURALI 2: kuresel weight merkezinden disari
            r =kok -center_ 
            v2 =e *(1.0 if (e @r )>=0 else -1.0 )
            if np .degrees (np .arccos (
            np .clip (float (v2 @Gn [j ]),-1 ,1 )))<=K .ACI :
                say ["tup_disari"]+=1 

        a =ist [d ["mfg"]]
        a ["gt"].append (len (G ))
        for k_ ,v_ in say .items ():
            a [k_ ].append (v_ )
        n +=1 
        if n %20 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{n } part | skipped {dict (skipped )}")
    print ("GT with ESLESEN adaylarda direction dogrulugu (CARPIM kutusu)")
    print (f"{'brand':<7}{'GT':>7}"+"".join (f"{k :>13}"for k in KOLLAR ))
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =max (sum (a ["gt"]),1 )
        r ={k :sum (a [k ])/g for k in KOLLAR }
        r ["gt"]=g 
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>7}"+"".join (f"{r [k ]:>13.4f}"for k in KOLLAR ))
    print ("\n=== BUGUNKUYE GORE (dense brand NIT) ===")
    if "NIT"in out :
        h =out ["NIT"]["bugunku"]
        print (f"  {'tup_eksen':<12}{out ['NIT']['tup_eksen']:.4f}   "
        f"(ISARETSIZ -- yuksekse axis DOGRU, sorun isarette)")
        for k_ in ("tup","tup_hava","tup_disari","tup_x_skor"):
            f =out ["NIT"][k_ ]-h 
            print (f"  {k_ :<12}{out ['NIT'][k_ ]:.4f}   {f :+.4f}"
            +("  <- KAPI GECTI"if f >=0.05 else ""))
        print (f"  {'oracle':<12}{out ['NIT']['oracle']:.4f}   (ust sinir)")
    json .dump ({"damga":receipt_hash .damga (),"cluster":KUME ,"brand":out ,
    "maks_konum":MAKS_KONUM ,"maks_yon":MAKS_YON ,
    "not":"Isin atma with delik ekseni. CARPIM kabul kutusu "
    "(lateral 2mm / axial 40mm) -- Oklid DEGIL. "
    "Sentetik yetenek testi GECTI. D7'ye BAKILMADI."},
    open (f"results/isin_ekseni_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/isin_ekseni_{KUME }.json")


if __name__ =="__main__":
    main ()
