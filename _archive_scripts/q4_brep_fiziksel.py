# -*- coding: utf-8 -*-
"""Q4: B-rep FIZIKSEL ozellikleri gate'in gormedigi bilgiyi tasiyor mu?

KANIT (kayitli FP otopsisi): wrong tespitlerin %86.3'u UC fiziksel imzaya ait --
axis-dik %35, BOYDAN BOYA DELIK %31, radius<1mm %21. Gate'in 13 ozelliginin HICBIRI
bunlari tasimiyor: `size` mesh mouth genisligi (approximately), `depth` isin sondaji, gerisi
probability/komsuluk istatistigi. Yani gate, yanlislarinin %86'sini olusturan fizigi GORMUYOR.

Silindir yaricaplari residual TAM DOGRU (cember oturtma duzeltmesi, metinle 327/327 eslesti) --
this measurement ondan before yapilamazdi because radius 3.5 fold yanlisti.

OLCULEN OZELLIKLER (all of them B-rep'ten, mesh tahmini DEGIL):
  brep_r      : matched silindirin ANALITIK yaricapi   (vida deligi small, tel girisi large)
  esesenli    : same axis cizgisini paylasan silindir count (havsa/dis imzasi)
  r_orani     : es-eksenli yaricaplarin max/min orani  (havsali vida deligi >1)
  bos_derinlik: axis along first "baseline"a up to distance (kor hole) -- otherwise BOYDAN BOYA
  passing       : boydan paint mi (0/1)

YONTEM: DEV+VAL adaylari (200 part, geometri as ayrik) ten TP/FP etiketiyle
Mann-Whitney AUC + BAG duzeltmesi (kisayol AUC formulu dengesiz/ikili ozelliklerde SISIYOR --
2026-07-29'da 0.953 sanip gercegi 0.563 cikmisti) + PERMUTASYON null testi.

KILL (onceden yazildi): no feature AUC >= 0.60 vermezse (null'a according to anlamli) channel KAPANIR.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def auc_mw (x ,y ):
    """Mann-Whitney U -> AUC, BAG duzeltmeli (kisayol formulu ikili ozelliklerde sisiyordu)."""
    x =np .asarray (x ,float );y =np .asarray (y ,bool )
    if y .all ()or not y .any ():
        return float ("nan")
    r =np .empty (len (x ),float )
    o =np .argsort (x ,kind ="mergesort")
    xs =x [o ]
    i =0 
    while i <len (xs ):# bagli degerlere ORTALAMA order
        j =i 
        while j +1 <len (xs )and xs [j +1 ]==xs [i ]:
            j +=1 
        r [o [i :j +1 ]]=(i +j )/2.0 +1.0 
        i =j +1 
    n1 =int (y .sum ());n0 =len (y )-n1 
    return float ((r [y ].sum ()-n1 *(n1 +1 )/2.0 )/(n1 *n0 ))


def ozellikler (p ,d ,cyl ,mesh ):
    """Bir candidate for B-rep fiziksel ozellikleri."""
    import cp_geometry as G 
    C ,A ,R =cyl 
    out =dict (brep_r =0.0 ,esesenli =0 ,r_orani =1.0 ,bos_derinlik =0.0 ,passing =0 )
    if len (C ):
        rel =p -C 
        al =(rel *A ).sum (1 )
        off =np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
        ok =off <=3.0 
        if ok .any ():
            cand =np .where (ok )[0 ]
            j =cand [int (np .argmax (np .abs (A [cand ]@d )))]
            out ["brep_r"]=float (R [j ])
            # same EKSEN CIZGISINI paylasanlar: direction parallel + eksenler cakisik
            par =np .abs (A @A [j ])>0.995 
            dd =C -C [j ]
            coax =par &(np .linalg .norm (dd -(dd @A [j ])[:,None ]*A [j ],axis =1 )<0.5 )
            rr =R [coax ]
            out ["esesenli"]=int (coax .sum ())
            if len (rr )>1 :
                out ["r_orani"]=float (rr .max ()/max (rr .min (),1e-6 ))
    try :
        h =G .ray_hits (mesh ,p +0.05 *d ,d ,max_mm =200.0 )
        h =np .asarray (h ,float ).ravel ()
        if len (h ):
            out ["bos_derinlik"]=float (h [0 ])
            out ["passing"]=int (len (h )<=1 )# single cikis = obur taraftan cikiyor
        else :
            out ["passing"]=1 
    except Exception :
        pass 
    return out 


def main ():
    import cp_openings ,robot_cp ,wire_gate ,trimesh 
    import brep_axes as B 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from j_position_mean import vote_avg 

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])

    X ,Y =[],[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        cache =pickle .load (open (cf ,"rb"))
        for r in cache :
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            mk =lambda pr_ ,**kw :cp_openings .connection_points (
            V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"],**kw )
            merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
            base =merge ([mk (pb )for pb in plist ])
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
            if not cps :
                continue 
            try :
                cyl =B .cylinders (r ["stp"])
            except Exception :
                continue 
            mesh =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
            P =np .array ([c ["point"]for c in cps ],float )
            D =np .array ([c ["direction"]for c in cps ],float )
            G_ ,Gd =r ["G"],r ["Gd"]
            # TP/FP: gevsek (detection) olcutle
            lab =np .zeros (len (P ),bool )
            if len (G_ ):
                diff =P [:,None ,:]-G_ [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                t0 =max (3.0 ,0.06 *r ["diag"])
                us ,ug =set (),set ()
                for dv ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
                for a in range (len (P ))for b in range (len (G_ ))):
                    if dv >t0 or a_ in us or b_ in ug :
                        continue 
                    us .add (a_ );ug .add (b_ );lab [a_ ]=True 
            for k in range (len (P )):
                X .append (ozellikler (P [k ],D [k ],cyl ,mesh ));Y .append (bool (lab [k ]))
        print (f"  {cluster } bitti (total {len (X )} candidate)",flush =True )

    names =list (X [0 ].keys ())
    M =np .array ([[x [n ]for n in names ]for x in X ],float )
    Y =np .array (Y ,bool )
    print (f"\n{len (Y )} candidate | TP {int (Y .sum ())} / FP {int ((~Y ).sum ())}")
    print (f"\n{'feature':<16}{'AUC':>8}{'null p95':>10}{'TP med':>10}{'FP med':>10}{'karar':>10}")
    rng =np .random .RandomState (0 )
    out ={}
    for i ,n in enumerate (names ):
        a =auc_mw (M [:,i ],Y )
        nulls =np .array ([auc_mw (M [:,i ],rng .permutation (Y ))for _ in range (200 )])
        p95 =float (np .percentile (np .abs (nulls -0.5 ),95 )+0.5 )
        kar ="CANLI"if abs (a -0.5 )+0.5 >=max (0.60 ,p95 )else "-"
        print (f"{n :<16}{a :>8.3f}{p95 :>10.3f}{np .median (M [Y ,i ]):>10.3f}"
        f"{np .median (M [~Y ,i ]):>10.3f}{kar :>10}")
        out [n ]={"auc":float (a ),"null_p95":p95 ,
        "tp_med":float (np .median (M [Y ,i ])),"fp_med":float (np .median (M [~Y ,i ])),
        "karar":kar }
    canli =[n for n in names if out [n ]["karar"]=="CANLI"]
    print (f"\nKILL: AUC>=0.60 veren ozellik yoksa kanal KAPANIR -> "
    f"{'AC ('+', '.join (canli )+')'if canli else 'KAPAT'}")
    np .savez ("results/q4_brep_fiziksel.npz",M =M ,Y =Y ,names =np .array (names ))
    json .dump (out ,open ("results/q4_brep_fiziksel.json","w"),indent =1 )
    print ("receipt -> results/q4_brep_fiziksel.json (+ .npz: ozellikler saklandi)")


if __name__ =="__main__":
    main ()
