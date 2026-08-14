# -*- coding: utf-8 -*-
"""Y16 -- REMESH-VARYANT TOPLULUGU. Uc remesh hedefinin ciktilarini merges.

RATIONALE. Zincir each parcayi 6000 tepeye remesh eder. Remesh KENDISI a
noise kaynagidir: different hedef cozunurluk different ucgenleme, different
ozdeger tabani, different prediction produces. Ayni model three different hedefte
kosulup ciktilar birlestirilirse this gurultunun a kismi stabilize
may be. **Yeni training YOK** -- that is why seed gurultusune (0.046) tabi
DEGIL; arm single kosuda verdict giyebilir.

IKI VARYANT olculur:
  * BIRLESIM (oy >= 1): recall'u acar, kesinligi dusurur.
  * OYLAMA  (oy >= 2): kesinligi acar, recall'u dusurur.

Kiyas tabani, same sondanin 6000 hedefli kosusudur (i.e. mevcut davranis).
Karar ESLI PARCA BOOTSTRAP with verilir -- kiyas same parcalarda.
"""
import json 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import match_hungarian # noqa: E402

KUME_MM =float (os .environ .get ("RT_KUME","5.0"))
YOL =os .environ .get ("RT_YOL","saha")


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def _yukle (fp ):
    """dokum -> {pid: kayit}, only istenen path."""
    out ={}
    for r in json .load (open (fp )):
        if r .get ("yol")!=YOL :
            continue 
        out [str (r ["pid"])]=r 
    return out 


def birlestir (listeler ,min_oy ):
    """Uc varyantin (P, D) listelerini KUME_MM'de merges.

    Ilk varyant cipa alinir; each cluster, kendisine KUME_MM inside most yakin
    adaylari toplar. Kume oyu = that kumeye katki veren VARYANT count
    (same varyanttan two candidate oyu 1 sayar -- otherwise single varyant single basina
    coklu oy uretirdi).
    """
    P =np .concatenate ([l [0 ]for l in listeler ])if listeler else np .zeros ((0 ,3 ))
    D =np .concatenate ([l [1 ]for l in listeler ])if listeler else np .zeros ((0 ,3 ))
    src_ =np .concatenate ([np .full (len (l [0 ]),i )for i ,l in 
    enumerate (listeler )])if listeler else np .zeros (0 ,int )
    if not len (P ):
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    kullanildi =np .zeros (len (P ),bool )
    oP ,oD =[],[]
    # cipa order: before first varyant (6000 = mevcut davranis), after digerleri
    rank_ =np .argsort (src_ ,kind ="stable")
    for i in rank_ :
        if kullanildi [i ]:
            continue 
        d =np .linalg .norm (P -P [i ][None ,:],axis =1 )
        uye =np .where ((d <=KUME_MM )&(~kullanildi ))[0 ]
        kullanildi [uye ]=True 
        oy =len (set (src_ [uye ].tolist ()))
        if oy <min_oy :
            continue 
            # konum: uyelerin ortalamasi. direction: cipanin yonuyle same yarikureye
            # hizalanmis uye yonlerinin ortalamasi (sign kacmasini onler).
        p =P [uye ].mean (0 )
        d0 =D [i ]
        Du =D [uye ]*np .sign (np .maximum (D [uye ]@d0 ,-1.0 ))[:,None ]
        Du =np .where (np .abs (D [uye ]@d0 )[:,None ]>0 ,Du ,D [uye ])
        v =Du .mean (0 )
        n =float (np .linalg .norm (v ))
        oP .append (p )
        oD .append (v /n if n >1e-9 else d0 )
    return np .asarray (oP ).reshape (-1 ,3 ),np .asarray (oD ).reshape (-1 ,3 )


def olc (kayitlar ,selector ):
    """selector(pid, r) -> (P, D). Doner: (tespit, rob_isaretsiz, rob_isaretli,
    parca_basina_liste)."""
    tot ={k :[0 ,0 ,0 ]for k in ("tespit","rob","rbi")}
    part =[]
    for pid ,r in sorted (kayitlar .items ()):
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        Gd =_birim (r ["Gd"])
        diag =float (r ["diag"])
        P ,D =selector (pid ,r )
        line_ ={"pid":pid }
        for ad ,kw in (("tespit",dict (tol =2.0 ,am =180.0 ,signed =False )),
        ("rob",dict (tol =2.0 ,am =10.0 ,signed =False )),
        ("rbi",dict (tol =2.0 ,am =10.0 ,signed =True ))):
            tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,diag ,kw ["tol"],
            kw ["am"],False ,
            signed =kw ["signed"])
            tot [ad ][0 ]+=tp 
            tot [ad ][1 ]+=fp 
            tot [ad ][2 ]+=fn 
            line_ [ad ]=(tp ,fp ,fn )
        part .append (line_ )
    def f1 (t ):
        tp ,fp ,fn =t 
        return 2 *tp /max (2 *tp +fp +fn ,1 )
    return {k :f1 (v )for k ,v in tot .items ()},part 


def esli_bootstrap (pa ,pb ,ad ,n =4000 ,seed =0 ):
    """part duzeyinde esli bootstrap; FARKIN dagilimi."""
    rng =np .random .default_rng (seed )
    A =np .asarray ([r [ad ]for r in pa ],float )
    B =np .asarray ([r [ad ]for r in pb ],float )
    m =len (A )
    fk =[]
    for _ in range (n ):
        i =rng .integers (0 ,m ,m )
        a ,b =A [i ].sum (0 ),B [i ].sum (0 )
        f =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )
        fk .append (f (b )-f (a ))
    fk =np .asarray (fk )
    return float (fk .mean ()),float (np .percentile (fk ,2.5 )),float (np .percentile (fk ,97.5 )),float ((fk >0 ).mean ())


def main ():
    hedefler =[6000 ,5000 ,7200 ]# 6000 ONCE: cipa = mevcut davranis
    file_ ={t :f"results/_dokum_remesh{t }.json"for t in hedefler }
    file_ [6000 ]=os .environ .get ("RT_TABAN","results/_dokum_taban.json")
    K ={}
    for t in hedefler :
        if not os .path .exists (file_ [t ]):
            print (f"EKSIK: {file_ [t ]} -- arm tamamlanmadi")
            return 1 
        K [t ]=_yukle (file_ [t ])
    ortak =sorted (set .intersection (*[set (K [t ])for t in hedefler ]))
    print (f"yol={YOL } | ortak part: {len (ortak )} "
    f"(tekil: {[len (K [t ])for t in hedefler ]})")
    if not ortak :
        print ("ORTAK PARCA YOK")
        return 1 
    K ={t :{p :K [t ][p ]for p in ortak }for t in hedefler }

    def tek (t ):
        return lambda pid ,r :(np .asarray (K [t ][pid ]["P"],float ).reshape (-1 ,3 ),
        _birim (K [t ][pid ]["D"])
        if len (K [t ][pid ]["P"])else np .zeros ((0 ,3 )))

    def top (min_oy ):
        def f (pid ,r ):
            L =[]
            for t in hedefler :
                p =np .asarray (K [t ][pid ]["P"],float ).reshape (-1 ,3 )
                d =_birim (K [t ][pid ]["D"])if len (p )else np .zeros ((0 ,3 ))
                L .append ((p ,d ))
            return birlestir (L ,min_oy )
        return f 

    res_ ={}
    taban_parca =None 
    print (f"\n{'varyant':28s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for ad ,sec in [("TABAN (6000)",tek (6000 )),
    ("tek 5000",tek (5000 )),
    ("tek 7200",tek (7200 )),
    (f"TOPLULUK birlesim(>=1)",top (1 )),
    (f"TOPLULUK oylama(>=2)",top (2 )),
    (f"TOPLULUK oybirligi(>=3)",top (3 ))]:
        m ,part =olc (K [6000 ],sec )
        res_ [ad ]={"metrik":m ,"part":part }
        if taban_parca is None :
            taban_parca =part 
        print (f"{ad :28s} {m ['tespit']:8.4f} {m ['rob']:8.4f} {m ['rbi']:8.4f}")

    print ("\n--- ESLI PARCA BOOTSTRAP (TABAN'a per fark) ---")
    print (f"{'varyant':28s} {'metrik':>8s} {'fark':>9s} "
    f"{'%95 GA':>22s} {'poz%':>6s}")
    for ad in res_ :
        if ad .startswith ("TABAN"):
            continue 
        for mad in ("tespit","rob","rbi"):
            f ,lo ,hi ,pz =esli_bootstrap (taban_parca ,res_ [ad ]["part"],mad )
            yildiz =" *"if (lo >0 or hi <0 )else ""
            print (f"{ad :28s} {mad :>8s} {f :+9.4f} "
            f"[{lo :+.4f},{hi :+.4f}]{yildiz :>3s} {100 *pz :5.1f}")

    with open ("results/remesh_toplulugu.json","w")as fh :
        json .dump ({ad :v ["metrik"]for ad ,v in res_ .items ()},fh ,indent =1 )
    print ("\n-> results/remesh_toplulugu.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
