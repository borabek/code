# -*- coding: utf-8 -*-
"""J for HAVUZLANMIS karar: DEV + VAL birlikte (200 part).

SORUN: J (UNION temsilcisinin konumu = anlasan uyelerin confidence-agirlikli ortalamasi) single kumede
karar verilemeyecek up to small a etki showed:
    DEV  tespit +0.0054  GA [-0.0096, +0.0203]   robot +0.0108  GA [-0.0001, +0.0245]
    VAL  tespit -0.0001                          robot -0.0033
Iki GA da 0'i iceriyor -> single kumede karar verilemez. Iki kumeyi HAVUZLAMAK n'yi 100'den 200'e
removes and gucu artirir. Kumeler geometri as ayrik oldugu for havuzlamak mesru.

Bu betik new no sey KOSMAZ: exam kosularinin part-basina sayimlarini (results/sinav_*_parca.pkl)
reads. Yani ek a cluster harcamiyoruz.

DECISION: havuzlanmis GA still 0'i iceriyorsa J "olculebilir a kazanc not" diye raporlanir --
urunde kalir (ilkesel rationale: ensemble ortalamasi) but MANSET kazanci as SAYILMAZ.
"""
import os ,sys ,json ,pickle 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import f1w ,pr 

URUN ="URUN (J, 13 feature)"
KAPALI ="J closed"


def main ():
    P ={}
    for k in ("dev","val"):
        f =f"results/sinav_{k }_parca.pkl"
        if not os .path .exists (f ):
            print (f"eksik: {f } -- once `python sina_cluster.py {k }` kosulmali");return 
        P [k ]=pickle .load (open (f ,"rb"))

    print (f"{'cluster':<10}{'n':>5}{'J open tespit':>15}{'J closed':>11}{'difference':>9}")
    for k in ("dev","val"):
        a =P [k ][URUN ][0 ];b =P [k ][KAPALI ][0 ]
        print (f"{k :<10}{len (a ):>5}{f1w (a ):>15.4f}{f1w (b ):>11.4f}{f1w (a )-f1w (b ):>+9.4f}")

    out ={}
    print (f"\nHAVUZLANMIS (DEV+VAL), esli bootstrap 4000 tekrar:")
    print (f"  {'criterion':<10}{'J open':>9}{'J closed':>10}{'difference':>9}{'%95 GA':>21}{'karar':>10}")
    for mi ,mn in ((0 ,"tespit"),(1 ,"robot")):
        a =P ["dev"][URUN ][mi ]+P ["val"][URUN ][mi ]
        b =P ["dev"][KAPALI ][mi ]+P ["val"][KAPALI ][mi ]
        n =len (a )
        rng =np .random .RandomState (0 )
        ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])
        for ix in (rng .randint (0 ,n ,n )for _ in range (4000 ))])
        lo_ ,hi_ =np .percentile (ds ,[2.5 ,97.5 ])
        kar ="BELIRGIN"if lo_ >0 or hi_ <0 else "noise"
        print (f"  {mn :<10}{f1w (a ):>9.4f}{f1w (b ):>10.4f}{ds .mean ():>+9.4f}"
        f"{'['+format (lo_ ,'+.4f')+', '+format (hi_ ,'+.4f')+']':>21}{kar :>10}")
        out [mn ]={"j_acik":float (f1w (a )),"j_kapali":float (f1w (b )),
        "difference":float (ds .mean ()),"ga":[float (lo_ ),float (hi_ )],"karar":kar ,
        "n_parca":n }

        # HAVUZLANMIS MANSET: 200 parcalik durust prediction (two ayrik geometri kumesi)
    print (f"\nHAVUZLANMIS MANSET ({len (P ['dev'][URUN ][0 ])+len (P ['val'][URUN ][0 ])} part):")
    for mi ,mn in ((0 ,"tespit"),(1 ,"robot ")):
        a =P ["dev"][URUN ][mi ]+P ["val"][URUN ][mi ]
        n =len (a )
        rng =np .random .RandomState (1 )
        bs =np .array ([f1w ([a [i ]for i in ix ])
        for ix in (rng .randint (0 ,n ,n )for _ in range (4000 ))])
        lo_ ,hi_ =np .percentile (bs ,[2.5 ,97.5 ])
        print (f"  {mn } = {f1w (a ):.4f}   %95 GA [{lo_ :.4f}, {hi_ :.4f}]")
        out .setdefault ("headline",{})[mn .strip ()]=[float (f1w (a )),float (lo_ ),float (hi_ )]
    p_ ,r_ =pr (P ["dev"][URUN ][0 ]+P ["val"][URUN ][0 ])
    out ["headline"]["precision"]=float (p_ );out ["headline"]["recall"]=float (r_ )
    print (f"  precision {p_ :.3f} | recall {r_ :.3f}")

    json .dump (out ,open ("results/j_havuzlu_karar.json","w"),indent =1 )
    print ("\nmakbuz -> results/j_havuzlu_karar.json")


if __name__ =="__main__":
    main ()
