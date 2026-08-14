# -*- coding: utf-8 -*-
"""Q7: RENK, tel girisi with vidayi AYIRIYOR MU? -- varsayimla not, olcumle.

DURUM: "renk ayirt edici not, because vida da metal kelepce de metal" dedim; this a VARSAYIMDI.
Bu betik onu kanitlar ya da curutur.

YOL (each step ayri dogrulanir):
 1) STEP metninden RENKLI silindirler: OVER_RIDING_STYLED_ITEM -> ADVANCED_FACE ->
    CYLINDRICAL_SURFACE (yerel centre/axis/radius) + stil zinciri -> COLOUR_RGB.
 2) gmsh'ten KURESEL silindirler (brep_axes; yaricaplar residual TAM DOGRU, metinle 327/327).
 3) YEREL->KURESEL rijit donusum: TEKIL yaricapli (two tarafta da a times gecen) ciftlerle
    tohumla, after RANSAC with rafine et. Kabul olcutu ARTIK (<0.5mm) -- ic-point ORANI DEGIL
    (q2/q3'te ratio wrong olcuttu: binlerce candidate double varken ratio no zaman high cikmaz).
 4) Kabul edilen parcalarda each ADAY for renk ozellikleri:
      c_metal   : eslesen silindirin rengi gumusi metal tonunda mi
      c_govde   : eslesen silindirin rengi body (metal olmayan) tonunda mi
      c_yok     : eslesen silindire renk atanamadi
 5) TP/FP'ye karsi AUC (bag duzeltmeli Mann-Whitney) + permutasyon null.

KILL (onceden yazili): no renk ozelligi null'un on AUC >= 0.60 vermezse RENK KANALI
tel/vida ayrimi for OLU ilan edilir and a more acilmaz. Verirse listeye kaldirac as girer.

NOT: kabul edilen part count low may be; that zaman sonuc "olculemedi" diye raporlanir,
"whereas yaramaz" diye DEGIL -- ikisi same sey not.
"""
import os ,sys ,re ,json ,pickle ,collections 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q2_colour_alignment import parse ,kabsch 

REF =re .compile (rb"#(\d+)")
METAL =(0.824 ,0.824 ,0.784 )
TOL =0.06 


def _renk_haritasi (d ):
    """#face -> (r,g,b): OVER_RIDING_STYLED_ITEM zincirini coz."""
    def rgb_of (sid ,depth =0 ):
        """stil varliginden COLOUR_RGB'ye derinlemesine in."""
        if depth >8 or sid not in d :
            return None 
        t ,a =d [sid ]
        if t =="COLOUR_RGB":
            try :
                nums =a .split (b",")[1 :]
                return tuple (round (float (x ),3 )for x in nums [:3 ])
            except Exception :
                return None 
        for r in REF .findall (a ):
            v =rgb_of (int (r ),depth +1 )
            if v :
                return v 
        return None 

    out ={}
    for i ,(t ,a )in d .items ():
        if t not in ("OVER_RIDING_STYLED_ITEM","STYLED_ITEM"):
            continue 
        refs =[int (x )for x in REF .findall (a )]
        if not refs :
            continue 
            # last referans usually hedef ogedir; hepsini deneyip ADVANCED_FACE olani al
        hedef =[r for r in refs if r in d and d [r ][0 ]=="ADVANCED_FACE"]
        if not hedef :
            continue 
        col =None 
        for r in refs :
            if r in hedef :
                continue 
            col =rgb_of (r )
            if col :
                break 
        if col :
            for h in hedef :
                out [h ]=col 
    return out 


def _renkli_silindirler (step ):
    """[(yerel_merkez, axis, radius, rgb|None)] -- ADVANCED_FACE -> CYLINDRICAL_SURFACE."""
    d =parse (step )
    renk =_renk_haritasi (d )
    out =[]
    for i ,(t ,a )in d .items ():
        if t !="ADVANCED_FACE":
            continue 
        refs =[int (x )for x in REF .findall (a )]
        surf =[r for r in refs if r in d and d [r ][0 ]=="CYLINDRICAL_SURFACE"]
        if not surf :
            continue 
        st ,sa =d [surf [0 ]]
        try :
            rad =float (sa .rsplit (b",",1 )[1 ])
        except ValueError :
            continue 
        sr =[int (x )for x in REF .findall (sa )]
        if not sr or sr [0 ]not in d or d [sr [0 ]][0 ]!="AXIS2_PLACEMENT_3D":
            continue 
        p =[int (x )for x in REF .findall (d [sr [0 ]][1 ])]
        if len (p )<2 :
            continue 
        try :
            c =[float (x )for x in d [p [0 ]][1 ].split (b"(")[1 ].split (b")")[0 ].split (b",")]
            n =[float (x )for x in d [p [1 ]][1 ].split (b"(")[1 ].split (b")")[0 ].split (b",")]
        except Exception :
            continue 
        if len (c )==3 and len (n )==3 :
            out .append ((np .array (c ),np .array (n ),rad ,renk .get (i )))
    return out 


def _eksen_artik (lc ,la ,R ,t ,gc ,ga ,gr ,lr ):
    """Her yerel silindiri, DONUSTURULDUKTEN after most uygun kuresel silindirin EKSEN CIZGISINE
    which is dik uzakligiyla olc. Merkez-centre uzakligi DEGIL."""
    P =lc @R .T +t 
    A =la @R .T 
    out =[]
    for k in range (len (P )):
        uy =(np .abs (gr -lr [k ])<0.01 )&(np .abs (ga @A [k ])>0.99 )
        if not uy .any ():
            out .append (np .inf );continue 
        rel =P [k ]-gc [uy ]
        al =(rel *ga [uy ]).sum (1 )
        out .append (float (np .min (np .linalg .norm (rel -al [:,None ]*ga [uy ],axis =1 ))))
    return np .array (out )


def _donusum (loc ,gc ,ga ,gr ):
    """YEREL -> KURESEL rijit donusum.

    KRITIK: silindirin "merkezi" karsilik gelen a point DEGILDIR. STEP metnindeki
    AXIS2_PLACEMENT_3D noktasi axis on HERHANGI a places may be; bizim cember
    oturtmamiz whereas ornekleme araliginin ortasini veriyor. Ayni silindir, same axis, FARKLI
    point. Merkezleri eslestirmek (q2/q3'te yaptigim sey) that is why ~30mm residual uretiyordu.

    DOGRU COZUM two adimda:
      1) DONME: axis YONLERINDEN. Yaricabi tekil which is ciftler dogrudan karsilik gives;
         Kabsch unit vektorler on cozulur (oteleme direction vektorlerini etkilemez).
      2) OTELEME: "donmus yerel point, kuresel axis CIZGISI on must be" kisitindan.
         Her double for (I - a a^T)(R c_l + t - c_g) = 0 -> t'de dogrusal most small kareler.
    """
    lr =np .array ([x [2 ]for x in loc ]);lc =np .array ([x [0 ]for x in loc ])
    la =np .array ([x [1 ]/(np .linalg .norm (x [1 ])+1e-12 )for x in loc ])
    cl =collections .Counter (np .round (lr ,3 ));cg =collections .Counter (np .round (gr ,3 ))
    tek =[(i ,int (np .argmin (np .abs (gr -lr [i ]))))for i in range (len (lr ))
    if cl [round (lr [i ],3 )]==1 and cg .get (round (lr [i ],3 ))==1 ]
    if len (tek )<2 :
        return None ,None ,np .inf 

        # 1) DONME: axis yonleri (sign ambiguous -> kuresel yone most yakin isareti sec)
    A =np .array ([la [i ]for i ,_ in tek ])
    Bm =np .array ([ga [j ]for _ ,j in tek ])
    Bm =np .where ((A *Bm ).sum (1 )[:,None ]>=0 ,Bm ,-Bm )
    U ,_ ,Vt =np .linalg .svd (A .T @Bm )
    R =(U @np .diag ([1 ,1 ,np .sign (np .linalg .det (U @Vt ))])@Vt ).T 
    if np .linalg .matrix_rank (A -A .mean (0 ))<2 and len (A )<3 :
        return None ,None ,np .inf 

        # 2) OTELEME: point-CIZGI kisitindan dogrusal most small kareler
    M =np .zeros ((3 ,3 ));b =np .zeros (3 )
    for i ,j in tek :
        a =ga [j ]
        Pp =np .eye (3 )-np .outer (a ,a )# eksene dik izdusum
        M +=Pp 
        b +=Pp @(gc [j ]-R @lc [i ])
    try :
        t =np .linalg .lstsq (M ,b ,rcond =None )[0 ]
    except np .linalg .LinAlgError :
        return None ,None ,np .inf 
    res =_eksen_artik (lc ,la ,R ,t ,gc ,ga ,gr ,lr )
    ok =np .isfinite (res )
    return R ,t ,(float (np .median (res [ok ]))if ok .any ()else np .inf )


def auc_mw (x ,y ):
    x =np .asarray (x ,float );y =np .asarray (y ,bool )
    if y .all ()or not y .any ():
        return float ("nan")
    r =np .empty (len (x ),float );o =np .argsort (x ,kind ="mergesort");xs =x [o ]
    i =0 
    while i <len (xs ):
        j =i 
        while j +1 <len (xs )and xs [j +1 ]==xs [i ]:
            j +=1 
        r [o [i :j +1 ]]=(i +j )/2.0 +1.0 
        i =j +1 
    n1 =int (y .sum ());n0 =len (y )-n1 
    return float ((r [y ].sum ()-n1 *(n1 +1 )/2.0 )/(n1 *n0 ))


def main ():
    import cp_openings ,robot_cp 
    import brep_axes as B 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from j_position_mean import vote_avg 

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    nmax =int (sys .argv [1 ])if len (sys .argv )>1 else 60 

    parts =[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            parts .append (r )
    rng =np .random .RandomState (0 )
    parts =[parts [i ]for i in rng .choice (len (parts ),min (nmax ,len (parts )),replace =False )]
    print (f"{len (parts )} part | adim 1-3: renk + hizalama",flush =True )

    X ,Y =[],[]
    kabul =red =renksiz =0 
    for r in parts :
        try :
            gc ,ga ,gr =B .cylinders (r ["stp"])
            loc =_renkli_silindirler (r ["stp"])
        except Exception :
            red +=1 ;continue 
        if len (gc )<4 or len (loc )<4 :
            red +=1 ;continue 
        if not any (x [3 ]for x in loc ):
            renksiz +=1 ;continue 
        R ,t ,art =_donusum (loc ,np .asarray (gc ,float ),np .asarray (ga ,float ),
        np .asarray (gr ,float ))
        if R is None or art >=0.5 :
            red +=1 ;continue 
        kabul +=1 
        # kuresel silindirlere renk ata: donusturulmus yerel merkeze most yakin kuresel silindir
        # RENK ATAMASI da EKSEN CIZGISIYLE: centre karsilik gelen point not (see _donusum).
        lcP =np .array ([x [0 ]for x in loc ])@R .T +t 
        laP =np .array ([x [1 ]/(np .linalg .norm (x [1 ])+1e-12 )for x in loc ])@R .T 
        renk_of ={}
        for k ,x in enumerate (loc ):
            if x [3 ]is None :
                continue 
            uy =(np .abs (np .asarray (gr ,float )-x [2 ])<0.01 )&(np .abs (ga @laP [k ])>0.99 )
            if not uy .any ():
                continue 
            idx =np .where (uy )[0 ]
            rel =lcP [k ]-gc [idx ]
            al =(rel *ga [idx ]).sum (1 )
            dperp =np .linalg .norm (rel -al [:,None ]*ga [idx ],axis =1 )
            j =int (idx [int (np .argmin (dperp ))])
            if dperp .min ()<0.5 :
                renk_of [j ]=x [3 ]

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
        P =np .array ([c ["point"]for c in cps ],float )
        D =np .array ([c ["direction"]for c in cps ],float )
        G_ ,Gd =r ["G"],r ["Gd"]
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
            p =P [k ];dd =D [k ]
            rel =p -gc ;alm =(rel *ga ).sum (1 )
            off =np .linalg .norm (rel -alm [:,None ]*ga ,axis =1 )
            ok =off <=3.0 
            c_metal =c_govde =c_yok =0.0 
            metal_yakin =0.0 
            if ok .any ():
                cand =np .where (ok )[0 ]
                j =int (cand [int (np .argmax (np .abs (ga [cand ]@dd )))])
                col =renk_of .get (j )
                if col is None :
                    c_yok =1.0 
                elif all (abs (col [q ]-METAL [q ])<=TOL for q in range (3 )):
                    c_metal =1.0 
                else :
                    c_govde =1.0 
                    # channel along (es-eksenli) metal present mi
                par =np .abs (ga @ga [j ])>0.99 
                ddv =gc -gc [j ]
                coax =np .where (par &(np .linalg .norm (
                ddv -(ddv @ga [j ])[:,None ]*ga [j ],axis =1 )<1.0 ))[0 ]
                for q in coax :
                    cq =renk_of .get (int (q ))
                    if cq and all (abs (cq [w ]-METAL [w ])<=TOL for w in range (3 )):
                        metal_yakin =1.0 
            else :
                c_yok =1.0 
            X .append ([c_metal ,c_govde ,c_yok ,metal_yakin ]);Y .append (bool (lab [k ]))

    print (f"hizalama: KABUL {kabul } | red {red } | renksiz {renksiz } "
    f"(kabul orani {kabul /max (len (parts ),1 ):.3f})")
    if kabul <5 or len (Y )<100 :
        print ("\nOLCULEMEDI: kabul edilen part/candidate sayisi yetersiz. "
        "Bu 'renk ise yaramaz' DEMEK DEGILDIR -- hizalama yetersiz.")
        json .dump ({"kabul":kabul ,"red":red ,"renksiz":renksiz ,"n_aday":len (Y ),
        "karar":"OLCULEMEDI"},open ("results/q7_colour_degeri.json","w"),indent =1 )
        return 
    X =np .array (X ,float );Y =np .array (Y ,bool )
    names =["c_metal","c_govde","c_yok","kanalda_metal"]
    print (f"\n{len (Y )} candidate | TP {int (Y .sum ())} | {kabul } part")
    print (f"\n{'ozellik':<16}{'AUC':>8}{'null p95':>10}{'TP ort':>9}{'FP ort':>9}{'karar':>9}")
    rng2 =np .random .RandomState (0 );out ={}
    for i ,n in enumerate (names ):
        a =auc_mw (X [:,i ],Y )
        nl =np .array ([auc_mw (X [:,i ],rng2 .permutation (Y ))for _ in range (200 )])
        p95 =float (np .percentile (np .abs (nl -0.5 ),95 )+0.5 )
        kar ="CANLI"if abs (a -0.5 )+0.5 >=max (0.60 ,p95 )else "-"
        print (f"{n :<16}{a :>8.3f}{p95 :>10.3f}{X [Y ,i ].mean ():>9.3f}{X [~Y ,i ].mean ():>9.3f}{kar :>9}")
        out [n ]={"auc":float (a ),"null_p95":p95 ,"tp_ort":float (X [Y ,i ].mean ()),
        "fp_ort":float (X [~Y ,i ].mean ()),"karar":kar }
    canli =[n for n in names if out [n ]["karar"]=="CANLI"]
    print (f"\nKILL: AUC>=0.60 veren renk ozelligi yoksa RENK tel/vida ayrimi icin OLU -> "
    f"{'CANLI ('+', '.join (canli )+')'if canli else 'OLU'}")
    json .dump (out |{"kabul":kabul ,"red":red ,"n_aday":len (Y ),
    "karar":"CANLI"if canli else "OLU"},
    open ("results/q7_colour_degeri.json","w"),indent =1 )
    print ("receipt -> results/q7_colour_degeri.json")


if __name__ =="__main__":
    main ()
