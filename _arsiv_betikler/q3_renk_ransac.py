# -*- coding: utf-8 -*-
"""Q3: renk<->geometri eslemesi -- RANSAC with (Q2'nin acgozlu esleyicisi cakmisti).

Q1: parcalarin %83.8'inde >=2 ayrik renk present (gri/gumus metal, mavi/turuncu body) -> channel DOLU.
Q2: yaricapla ACGOZLU esleme cakti (residual ~30mm). Sebep tanindi: a parcada onlarca vida deligi
    AYNI yaricapta; first uygun olana eslemek korelasyonu coplestiriyor. Null test de same output.
gmsh yolu KAPALI (dogrulandi): Geometry.OCCImportLabels=1 with bile tum yuzler TEK renk donuyor,
    i.e. OVER_RIDING_STYLED_ITEM uygulanmiyor.

YENI IMKAN: silindir yaricaplari residual TAM DOGRU (cember oturtma duzeltmesi; metin with 327/327
eslesti, medyan error 0.0000mm). Dogru radius = guvenilir candidate-double havuzu. Belirsizligi RANSAC
cozer: 3 double ornekle, rijit donusumu coz, TUM ciftlerde ic-point say.

Donusum part basina degisiyor (measured: kimi parcada unit, kimi parcada real rijit hareket) --
that yuzden each part for ayri cozulur.

KILL: real ratio <0.60 ya da null testi gercege yaklasirsa (difference <0.3) is KAPANIR.
"""
import os ,sys ,json ,glob 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q2_colour_alignment import parse ,yerel_silindirler ,kabsch 


def ransac (lc ,lr ,la ,gc ,gr ,ga ,tol =0.5 ,iters =4000 ,rng =None ,karistir =False ):
    """Yaricap-uyumlu ciftler ten RANSAC rijit donusum. Doner: (R,t,ic_nokta,toplam_cift)."""
    rng =rng or np .random .RandomState (0 )
    pairs =[(i ,j )for i in range (len (lc ))for j in range (len (gc ))
    if abs (lr [i ]-gr [j ])<0.01 ]
    if karistir :# NULL TEST: radius kisitini kaldirip rastgele double uret
        pairs =[(int (rng .randint (len (lc ))),int (rng .randint (len (gc ))))
        for _ in range (len (pairs ))]
    if len (pairs )<6 :
        return None ,None ,0 ,len (pairs )
    P =np .array ([lc [i ]for i ,_ in pairs ]);Q =np .array ([gc [j ]for _ ,j in pairs ])
    best =(0 ,None ,None )
    for _ in range (iters ):
        k =rng .choice (len (pairs ),3 ,replace =False )
        A ,Bm =P [k ],Q [k ]
        if np .linalg .matrix_rank (A -A .mean (0 ))<2 :
            continue 
        R ,t ,_ =kabsch (A ,Bm )
        d =np .linalg .norm (P @R .T +t -Q ,axis =1 )
        n =int ((d <tol ).sum ())
        if n >best [0 ]:
            best =(n ,R ,t )
    n ,R ,t =best 
    if R is not None and n >=4 :# ic-noktalarla yeniden coz (rafine)
        d =np .linalg .norm (P @R .T +t -Q ,axis =1 )
        m =d <tol 
        R ,t ,_ =kabsch (P [m ],Q [m ])
    return R ,t ,n ,len (pairs )


def main ():
    import brep_axes as B 
    n =int (sys .argv [1 ])if len (sys .argv )>1 else 15 
    files =sorted (glob .glob ("all_wscad_stp/*.stp"))
    rng =np .random .RandomState (0 )
    sel =[files [i ]for i in rng .choice (len (files ),min (n ,len (files )),replace =False )]
    print (f"{len (sel )} parcada RANSAC hizalama\n")
    print (f"{'part':<14}{'double':>6}{'ic-point':>10}{'ratio':>7}{'residual':>10}"
    f"{'NULL ic':>9}{'NULL ratio':>11}")
    iyi =nul =tot =0 
    rec_ =[]
    for f in sel :
        pid =os .path .basename (f ).split ("_")[1 ]
        try :
            C ,A ,R_ =B .cylinders (f )
            loc =yerel_silindirler (parse (f ))
            if len (C )<6 or len (loc )<6 :
                print (f"{pid :<14}{'-':>6}{'few silindir':>10}")
                continue 
            lc =np .array ([x [1 ]for x in loc ]);lr =np .array ([x [3 ]for x in loc ])
            la =np .array ([x [2 ]for x in loc ])
            gc =np .asarray (C ,float );gr =np .asarray (R_ ,float );ga =np .asarray (A ,float )
            Rm ,t ,ic ,np_ =ransac (lc ,lr ,la ,gc ,gr ,ga ,rng =np .random .RandomState (1 ))
            _ ,_ ,icn ,_ =ransac (lc ,lr ,la ,gc ,gr ,ga ,rng =np .random .RandomState (1 ),
            karistir =True )
        except Exception as e :
            print (f"{pid :<14}{'-':>6}{'HATA '+type (e ).__name__ :>10}")
            continue 
        tot +=1 
        ratio =ic /max (np_ ,1 );noran =icn /max (np_ ,1 )
        art ="-"
        if Rm is not None :
            d =np .linalg .norm (lc @Rm .T +t -gc [np .argmin (
            np .linalg .norm (lc [:,None ]@Rm .T +t -gc [None ],axis =2 ),axis =1 )],axis =1 )
            art =f"{np .median (d ):.3f}mm"
        ok =ratio >=0.30 and ic >=6 
        nok =noran >=0.30 and icn >=6 
        iyi +=ok ;nul +=nok 
        print (f"{pid :<14}{np_ :>6}{ic :>10}{ratio :>7.2f}{art :>10}{icn :>9}{noran :>11.2f}"
        f"{'  <- GECTI'if ok else ''}",flush =True )
        rec_ .append ({"pid":pid ,"double":np_ ,"ic":ic ,"ratio":ratio ,
        "null_ic":icn ,"null_oran":noran ,"gecti":bool (ok )})
    o =iyi /max (tot ,1 );no =nul /max (tot ,1 )
    print (f"\nGERCEK: {iyi }/{tot } ({o :.3f}) hizalandi")
    print (f"NULL  : {nul }/{tot } ({no :.3f})   <- bu da yuksekse test DEGERSIZ")
    kar ="AC"if (o >=0.60 and no <o -0.3 )else "KAPAT"
    print (f"\nKILL: gercek >=0.60 VE null gercekten >=0.3 dusuk olmali -> {kar }")
    json .dump ({"n":tot ,"real":o ,"null":no ,"karar":kar ,"kayit":rec_ },
    open ("results/q3_renk_ransac.json","w"),indent =1 )
    print ("receipt -> results/q3_renk_ransac.json")


if __name__ =="__main__":
    main ()
