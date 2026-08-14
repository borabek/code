# -*- coding: utf-8 -*-
"""Q8: RENK tel girisi with vidayi ayiriyor mu? -- DECISION OLCUMU.

Q7 "OLCULEMEDI" dedi: silindir MERKEZLERINI eslestirmeye calisiyordum, oysa metnin
AXIS2_PLACEMENT_3D noktasi with bizim cember-oturtma noktamiz same axis ten FARKLI
points; hizalama %12'de kaldi.

COZUM: hizalamaya never gerek absent. `step_face_colors.read_by_order` rengi metinden (face order),
geometriyi OCC'den (KURESEL koordinat) takes. Sira hipotezi PARCA BASINA dogrulanir: each two
taraf da silindir yaricapini gives, eleman eleman karsilastirilir (8 parcada 7'si full).
Dogrulamayan part ATLANIR -- varsayim corpus genelinde kabul edilmez.

OZELLIKLER (each candidate for):
  c_metal       : adayin oturdugu silindirin rengi gumusi metal tonunda mi
  c_govde       : rengi metal-disi (body) mi
  kanalda_metal : same axis cizgisi uzerindeki silindirlerden biri metal mi (kanalin dibi)
  metal_mesafe  : most yakin metal yuze uzaklik (mm)

KILL (onceden yazili, degismedi): null'un ten AUC >= 0.60 veren renk ozelligi otherwise RENK
tel/vida ayrimi for OLU. Verirse listeye kaldirac as girer and gate'e feature as eklenir.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q7_colour_degeri import auc_mw 


def main ():
    import cp_openings ,robot_cp 
    import step_face_colors as SC 
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
    print (f"{len (parts )} part",flush =True )

    X ,Y =[],[]
    kabul =red =0 
    for r in parts :
        try :
            faces ,ok =SC .read_by_order (r ["stp"])
        except Exception :
            red +=1 ;continue 
        if not ok :
            red +=1 ;continue # order dogrulanmadi -> part ATLANIR
        cyl =[f for f in faces if f ["axis"]is not None and f ["rgb"]is not None ]
        if len (cyl )<2 :
            red +=1 ;continue # renk cozulemedi (STYLED_ITEM yuze not katiya bagli)
        kabul +=1 
        mcom =np .array ([f ["com"]for f in cyl if f ["is_metal"]],float )

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

        C =np .array ([f ["com"]for f in cyl ],float )if cyl else np .zeros ((0 ,3 ))
        A =np .array ([f ["axis"]/(np .linalg .norm (f ["axis"])+1e-12 )for f in cyl ],float )if cyl else np .zeros ((0 ,3 ))
        MET =np .array ([f ["is_metal"]for f in cyl ],bool )if cyl else np .zeros (0 ,bool )
        HAS =np .array ([f ["rgb"]is not None for f in cyl ],bool )if cyl else np .zeros (0 ,bool )
        for k in range (len (P )):
            p =P [k ];d =D [k ]
            c_metal =c_govde =kanal =0.0 
            if len (C ):
                rel =p -C 
                al =(rel *A ).sum (1 )
                off =np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
                near =off <=3.0 
                if near .any ():
                    cand =np .where (near )[0 ]
                    j =int (cand [int (np .argmax (np .abs (A [cand ]@d )))])
                    if HAS [j ]:
                        c_metal =float (MET [j ]);c_govde =float (not MET [j ])
                    par =np .abs (A @A [j ])>0.99 
                    dd =C -C [j ]
                    coax =par &(np .linalg .norm (dd -(dd @A [j ])[:,None ]*A [j ],axis =1 )<1.5 )
                    kanal =float (bool ((MET &coax ).any ()))
            md =99.0 
            if len (mcom ):
                md =float (np .min (np .linalg .norm (mcom -p ,axis =1 )))
            X .append ([c_metal ,c_govde ,kanal ,md ]);Y .append (bool (lab [k ]))

    print (f"sira DOGRULANAN part: {kabul } | skipped: {red }")
    if kabul <8 or len (Y )<200 :
        print ("\nOLCULEMEDI (yetersiz part/candidate) -- 'renk whereas yaramaz' DEMEK DEGILDIR.")
        json .dump ({"kabul":kabul ,"red":red ,"n":len (Y ),"karar":"OLCULEMEDI"},
        open ("results/q8_colour_decision.json","w"),indent =1 )
        return 
    X =np .array (X ,float );Y =np .array (Y ,bool )
    names =["c_metal","c_govde","kanalda_metal","metal_mesafe"]
    print (f"\n{len (Y )} candidate | TP {int (Y .sum ())} | {kabul } part")
    print (f"\n{'feature':<16}{'AUC':>8}{'null p95':>10}{'TP ort':>10}{'FP ort':>10}{'karar':>9}")
    rng2 =np .random .RandomState (0 );out ={}
    for i ,n in enumerate (names ):
        a =auc_mw (X [:,i ],Y )
        nl =np .array ([auc_mw (X [:,i ],rng2 .permutation (Y ))for _ in range (300 )])
        p95 =float (np .percentile (np .abs (nl -0.5 ),95 )+0.5 )
        kar ="CANLI"if abs (a -0.5 )+0.5 >=max (0.60 ,p95 )else "-"
        print (f"{n :<16}{a :>8.3f}{p95 :>10.3f}{X [Y ,i ].mean ():>10.3f}"
        f"{X [~Y ,i ].mean ():>10.3f}{kar :>9}")
        out [n ]={"auc":float (a ),"null_p95":p95 ,"tp_ort":float (X [Y ,i ].mean ()),
        "fp_ort":float (X [~Y ,i ].mean ()),"karar":kar }
    canli =[n for n in names if out [n ]["karar"]=="CANLI"]
    print (f"\nKILL: AUC>=0.60 veren renk ozelligi yoksa RENK tel/vida ayrimi for OLU -> "
    f"{'CANLI ('+', '.join (canli )+')'if canli else 'OLU'}")
    np .savez ("results/q8_renk.npz",X =X ,Y =Y )
    json .dump (out |{"kabul":kabul ,"red":red ,"n":len (Y ),
    "karar":"CANLI"if canli else "OLU"},
    open ("results/q8_colour_decision.json","w"),indent =1 )
    print ("receipt -> results/q8_colour_decision.json")


if __name__ =="__main__":
    main ()
