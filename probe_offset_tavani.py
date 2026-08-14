# -*- coding: utf-8 -*-
"""OFFSET/OY BASININ TAVANI — training YOK, karar for.

FIKIR (21.75, literaturden): konum bugun "segmentlenen bolgenin weight
merkezi"nden turetiliyor. Onerilen: agin each TEPESI own CP merkezine a
KAYMA VEKTORU prediction etsin, vertices kaydirilip kumelensin.

BU BETIK KOLU KURMUYOR, **TAVANINI** OLCUYOR. Cunku:
  * kusursuz offset with sonuc TRIVIAL as mukemmeldir -> bilgi vermez;
  * asil karar sorusu this: **offset basi NE KADAR HASSAS must be ki
    bugunku sonucu gecsin?**

Bu yuzden GT offsetlerine gercekci GURULTU eklenir and noise taranir.
Cikan threshold, egitime girmeden before "this hedef ulasilabilir mi" sorusunu
cevaplar. Kiyas noktasi: pose head'in bugun OOF'ta ulastigi residual
(median 0.67 mm) -- i.e. network'in this kind a regresyonda ulasabildigi
real hassasiyet.

KUMELEME: kaydirilmis vertices mean-shift benzeri sabit bantli
yogunlasmayla kumelenir (bant = clustering yaricapi). Her kumenin merkezi
a CP adayidir. Bu, Panoptic-DeepLab/Spatial Embeddings ailesindeki
"shift + cluster" adiminin birebir karsiligidir.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

BANT =float (os .environ .get ("OT_BANT","2.0"))# clustering yaricapi mm
MIN_UYE =int (os .environ .get ("OT_MIN_UYE","3"))# cluster for at least vertex
N_PARCA =int (os .environ .get ("OT_N","40"))


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def kumelen (Q ,bant =BANT ,min_uye =MIN_UYE ):
    """kaydirilmis noktalari yogunluga according to kumele -> cluster merkezleri.

    Basit and deterministik: at most komsusu which is point cipa alinir, `bant`
    icindeki each sey that kumeye girer and cikarilir; tekrarlanir.
    """
    if not len (Q ):
        return np .zeros ((0 ,3 )),[]
    kalan =np .ones (len (Q ),bool )
    center_ ,uyeler =[],[]
    while kalan .any ():
        idx =np .where (kalan )[0 ]
        P =Q [idx ]
        # each candidate for bant icindeki komsu count
        d =np .linalg .norm (P [:,None ,:]-P [None ,:,:],axis =-1 )
        say =(d <=bant ).sum (1 )
        i =int (np .argmax (say ))
        uye =idx [d [i ]<=bant ]
        if len (uye )>=min_uye :
            center_ .append (Q [uye ].mean (0 ))
            uyeler .append (len (uye ))
        kalan [uye ]=False 
    return np .asarray (center_ ).reshape (-1 ,3 ),uyeler 


def main ():
    import thesis_remesh 
    import export_robot_glb as EX 

    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    STEP =K .step_map ()
    s3 =json .load (io .open ("results/split3.json",encoding ="utf-8"))
    val =[str (x )for x in s3 ["val"]["parts"]]
    candidate =[p for p in val if p in STEP and p in kay ][:N_PARCA ]
    print (f"{len (candidate )} VAL parcasi | bant {BANT } mm | min uye {MIN_UYE }\n",
    flush =True )

    SIGMA =[0.0 ,0.25 ,0.5 ,0.75 ,1.0 ,1.5 ,2.0 ,3.0 ]
    tot ={s :{"tespit":[0 ,0 ,0 ],"rob":[0 ,0 ,0 ]}for s in SIGMA }
    rng =np .random .default_rng (0 )
    n_ok =0 
    for pid in candidate :
        try :
            Vr ,Fr =EX .step_to_mesh (STEP [pid ])
            V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 )
        except Exception as e :# noqa: BLE001
            print (f"  {pid }: {type (e ).__name__ } -- atlandi")
            continue 
        r =kay [pid ]
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        if not len (G ):
            continue 
        Gd =_birim (r ["Gd"])
        dg =float (r ["diag"])
        # HANGI TEPELER OY VERIR: a CP'ye 6 mm'den yakin olanlar.
        # (Gercekte bunu `seed` head ogrenir; here ceiling olculuyor.)
        d =np .linalg .norm (V [:,None ,:]-G [None ,:,:],axis =-1 )
        en_yakin =d .argmin (1 )
        dist_ =d .min (1 )
        oy =dist_ <=6.0 
        if oy .sum ()<MIN_UYE :
            continue 
        atama =en_yakin [oy ].copy ()
        # SEED/ATAMA KARISMASI (2026-08-14). Tavan olcumunde "kim oy gives"
        # kahinden geliyordu. Gercekte agin most large hatasi BITISIK AYNI
        # AGIZLARI KARISTIRMAK olacak. `OT_KARISMA` orani up to oy, KOMSU
        # a CP'ye kaydirilir -- i.e. error rastgele not, bizim real
        # error bicimimizde simule edilir.
        _kar =float (os .environ .get ("OT_KARISMA","0"))
        if _kar >0 and len (G )>1 :
            _dg =np .linalg .norm (G [:,None ,:]-G [None ,:,:],axis =-1 )
            np .fill_diagonal (_dg ,np .inf )
            _komsu =_dg .argmin (1 )# each CP'nin most yakin CP'si
            _boz =rng .random (len (atama ))<_kar 
            atama [_boz ]=_komsu [atama [_boz ]]
        hedef =G [atama ]# kusursuz offset sonucu
        for s in SIGMA :
            Q =hedef +(rng .normal (0.0 ,s ,hedef .shape )if s >0 else 0.0 )
            P ,_u =kumelen (Q )
            D =np .zeros ((len (P ),3 ))
            if len (P ):
            # direction: GT'nin most yakin CP yonu (this betik YONU olcmuyor,
            # tespit and unsigned-konum tavanina bakiyor)
                j =np .linalg .norm (P [:,None ,:]-G [None ,:,:],
                axis =-1 ).argmin (1 )
                D =Gd [j ]
            for ad ,tol ,am ,pct in (("tespit",0.0 ,180.0 ,True ),
            ("rob",2.0 ,10.0 ,False )):
                tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,dg ,tol ,am ,pct )
                tot [s ][ad ][0 ]+=tp 
                tot [s ][ad ][1 ]+=fp 
                tot [s ][ad ][2 ]+=fn 
        n_ok +=1 
        if n_ok %10 ==0 :
            print (f"  {n_ok }/{len (candidate )}",flush =True )

    def f1 (t ):
        return 2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )

    print (f"\n{n_ok } part | KAYDIR + KUMELE tavani\n")
    print (f"{'offset gurultusu':>18s} {'tespit F1':>10s} {'konum F1':>10s}")
    for s in SIGMA :
        print (f"{s :15.2f} mm {f1 (tot [s ]['tespit']):10.4f} "
        f"{f1 (tot [s ]['rob']):10.4f}")
    print ("\nKIYAS (ayni criterion, bugunku urun, VAL 100): "
    "tespit 0.7878 / robot-axis 0.5764")
    print ("Pose head'in bugun OOF'ta ulastigi residual: median 0.67 mm")
    json .dump ({str (s ):{k :f1 (v )for k ,v in tot [s ].items ()}
    for s in SIGMA },
    io .open ("results/offset_tavani.json","w",encoding ="utf-8"),
    indent =1 )
    print ("-> results/offset_tavani.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
