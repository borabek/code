# -*- coding: utf-8 -*-
"""YENI-7 -- IKI YONLU (BIMODAL) YON + DISARI ISARETI

DIAGNOSIS. NIT'te direction KAHINI 0.593, gerceklesen 0.150. Bugune kadarki most large
single bosluk. Iki olculmus basarisizlik same sebebe sign ediyor:
  * K2.1 periyodiklik : tespit +0.0126 but robot -0.0100  (direction KOPYALAMA)
  * dik_kipsel        : 0.064, dik_suzgec 0.166'nin ALTINDA (modal oylama)
Ikisi de "parcadaki yonu kopyala" fikrini denedi, ikisi de coktu.

HIPOTEZ. Bir klemens blogunda girisler TEK yonlu degildir: giris and cikis
karsit yuzlerdedir. Isaretli angle metriginde 180 derece = TAM basarisizlik.
Modal oylama single kumeyi selects, parcanin diger yarisi ters isaretlenir.
Yani kopyalama fikri wrong degildi; ISARET ele alinmamisti.

COZUM. Yonu ISARETSIZ EKSEN as sec (+/- birlesir), isareti AYRICA
"govdeden disari" kosulundan koy. GT direction sozlesmesi denetimi disariligin
1.000 tutarli oldugunu olctu -- i.e. sign ogrenilecek not, TURETILECEK
a buyukluktur.

ONCE VERIFICATION: GT yonlerinin part ici kip count ISARETLI and ISARETSIZ
as ayri sayilir. Isaretsizde belirgin sekilde azaliyorsa hipotez ayakta.

KOLLAR (all of them GT KONUMUNDA, i.e. direction TEK BASINA yalitilir):
  bugunku       : most high skorlu adayin yonu
  eksen_skor    : yerel unsigned modal axis, sign skordan
  eksen_disari  : yerel unsigned modal axis, sign DISARIDAN (centre)
  eksen_dis_yer : same, but yerel (15mm) merkeze according to disari
  parca_eksen   : PARCA capinda single unsigned axis + disari isareti
  kahin         : correct direction candidates between VAR mi

KAPI: NIT'te herhangi a arm `bugunku`yu >= 0.05 asacak.
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
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402

KUME =os .environ .get ("BY_KUME","d6")
KAT_MIN =int (os .environ .get ("BY_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
YAKIN_R =float (os .environ .get ("BY_R","2.0"))
EKSENEL =float (os .environ .get ("BY_EKSENEL","40.0"))
YEREL_R =float (os .environ .get ("BY_YEREL","15.0"))
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
KOLLAR =("bugunku","eksen_skor","eksen_disari","eksen_dis_yer",
"parca_eksen","kahin")


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def eksen_modal (Y ,W =None ):
    """ISARETSIZ modal axis: +v and -v AYNI sayilir.

    Oy = |cos| esigi uzerindeki komsu count. Kazanani, kendisine hizali
    olanlarin sign DUZELTILMIS ortalamasiyla inceltiriz.
    """
    if len (Y )==1 :
        return Y [0 ]
    c =np .abs (np .clip (Y @Y .T ,-1 ,1 ))
    oy =(np .degrees (np .arccos (c ))<=K .ACI )
    a =oy .sum (1 )if W is None else (oy *W [None ,:]).sum (1 )
    j =int (np .argmax (a ))
    uy =Y [oy [j ]]
    sign =np .sign (uy @Y [j ])
    sign [sign ==0 ]=1.0 
    v =(uy *sign [:,None ]).mean (0 )
    n =np .linalg .norm (v )
    return Y [j ]if n <1e-9 else v /n 


def kip_sayisi (Y ,signed ):
    """acgozlu clustering with kip count (ACI toleransinda)."""
    kalan =list (range (len (Y )))
    n =0 
    while kalan :
        j =kalan [0 ]
        c =Y [kalan ]@Y [j ]
        c =np .clip (c if signed else np .abs (c ),-1 ,1 )
        yakin =np .degrees (np .arccos (c ))<=K .ACI 
        kalan =[k for k ,y in zip (kalan ,yakin )if not y ]
        n +=1 
    return n 


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar }",flush =True )

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
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        G =np .asarray (d ["G"],float )
        if len (G )<2 :
            continue 
        Gn =_birim (np .asarray (d ["Gd"],float ))
        P =np .asarray (d ["P"],float )
        idx =np .asarray (d ["idx"],int )
        YD =_birim (np .asarray (d ["YD"],float ))
        s =np .asarray (s ,float )

        # --- VERIFICATION: GT yonlerinin kip count
        a =ist [d ["mfg"]]
        a ["kip_imzali"].append (kip_sayisi (Gn ,True ))
        a ["kip_eksen"].append (kip_sayisi (Gn ,False ))

        mf =f"{MESH }/{d ['pid']}.npz"
        V =(np .asarray (np .load (mf )["V"],float )
        if os .path .exists (mf )else P )
        center_ =V .mean (0 )

        # PARCA capinda single unsigned axis (upper skorlu adaylardan)
        ust =np .argsort (-s )[:max (30 ,2 *len (G ))]
        p_eksen =eksen_modal (YD [ust ],s [ust ])

        say ={k :0 for k in KOLLAR }
        # ESLESME KUTUSU DUZELTMESI. Ilk kosuda adaylari GT'ye OKLID 2mm with
        # esledim; oysa kabul kutusu CARPIMDIR: GT yonune according to lateral <= 2mm
        # and axial <= 40mm. Bir candidate eksende 30mm uzakta olup still gecerli
        # eslesme may be. Oklid kullanmak `kahin`i 0.593'ten 0.0172'ye
        # dusuruyordu -- olculen sey mekanizma not KUSURDU.
        Pk =P [idx ]
        for j in range (len (G )):
            v =Pk -G [j ][None ,:]
            al =v @Gn [j ]
            yan =np .linalg .norm (v -al [:,None ]*Gn [j ][None ,:],axis =1 )
            m_ =(yan <=K .YANAL )&(np .abs (al )<=EKSENEL )
            if not m_ .any ():
                continue 
            Yo ,So =YD [m_ ],s [m_ ]
            aci =np .degrees (np .arccos (np .clip (Yo @Gn [j ],-1 ,1 )))
            if (aci <=K .ACI ).any ():
                say ["kahin"]+=1 
            if aci [int (np .argmax (So ))]<=K .ACI :
                say ["bugunku"]+=1 

            e =eksen_modal (Yo ,So )
            # KONUM SIZINTISI DUZELTMESI. Ilk kosuda `konum = G[j]`, i.e.
            # DISARI testi GT KONUMUNU kullaniyordu. Uruende elimizde only
            # ADAY konumu present and candidate eksende 40mm'ye up to kayabilir --
            # `e @ (konum - centre)` isareti that is why ters donebilir.
            # Dogrusu: that GT with eslesen adaylarin EN YUKSEK SKORLUSUNUN
            # own konumu (uruende de this bilinir).
            konum =Pk [m_ ][int (np .argmax (So ))]
            # sign: skor agirlikli izdusum
            pr =(So *(Yo @e )).sum ()
            v1 =e *(1.0 if pr >=0 else -1.0 )
            # sign: kuresel merkezden disari
            r =konum -center_ 
            v2 =e *(1.0 if (e @r )>=0 else -1.0 )
            # sign: YEREL merkezden disari
            yak =V [np .linalg .norm (V -konum ,axis =1 )<=YEREL_R ]
            ry =konum -(yak .mean (0 )if len (yak )>=10 else center_ )
            v3 =e *(1.0 if (e @ry )>=0 else -1.0 )
            # PARCA axis + kuresel disari
            v4 =p_eksen *(1.0 if (p_eksen @r )>=0 else -1.0 )
            for ad ,v in (("eksen_skor",v1 ),("eksen_disari",v2 ),
            ("eksen_dis_yer",v3 ),("parca_eksen",v4 )):
                if np .degrees (np .arccos (
                np .clip (float (v @Gn [j ]),-1 ,1 )))<=K .ACI :
                    say [ad ]+=1 
        a ["gt"].append (len (G ))
        for k_ ,v_ in say .items ():
            a [k_ ].append (v_ )
        n +=1 
        if n %80 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{n } part | GT KONUMUNDA direction dogrulugu (direction YALITILDI)")
    print (f"{'brand':<7}{'GT':>6}{'kip+-':>7}{'kipEks':>8}"
    +"".join (f"{k :>14}"for k in KOLLAR ))
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =max (sum (a ["gt"]),1 )
        r ={k :sum (a [k ])/g for k in KOLLAR }
        r ["gt"]=g 
        r ["kip_imzali"]=float (np .median (a ["kip_imzali"]))
        r ["kip_eksen"]=float (np .median (a ["kip_eksen"]))
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>6}{r ['kip_imzali']:>7.1f}{r ['kip_eksen']:>8.1f}"
        +"".join (f"{r [k ]:>14.4f}"for k in KOLLAR ))
    print ("\n=== BUGUNKUYE GORE (dense brand NIT) ===")
    if "NIT"in out :
        h =out ["NIT"]["bugunku"]
        for k_ in KOLLAR [1 :-1 ]:
            f =out ["NIT"][k_ ]-h 
            print (f"  {k_ :<14}{out ['NIT'][k_ ]:.4f}   {f :+.4f}"
            +("  <- KAPI GECTI"if f >=0.05 else ""))
        print (f"  {'kahin':<14}{out ['NIT']['kahin']:.4f}   (ust sinir)")
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,"brand":out ,
    "not":"Isaretsiz axis + disari isareti. GT KONUMUNDA, direction "
    "yalitildi. kip+- = GT yonlerinin signed kip sayisi, "
    "kipEks = unsigned. D7'ye BAKILMADI."},
    open (f"results/bimodal_yon_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/bimodal_yon_{KUME }.json")


if __name__ =="__main__":
    main ()
