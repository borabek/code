# -*- coding: utf-8 -*-
"""C2: PARCA DUZEYINDE YON -- C1'in sozlugundeki bosluk.

C1 SONUCU: fiziksel direction sozlugu uzerindeki kahin 0.6263 (bar 0.70) -> "correct direction
elimizdeki geometriden turetilemiyor" dedim. O cumle FAZLA GENISTI: correct direction BENIM
KURDUGUM sozlukte yoktu. Sozlugun tamami YERELDI (candidate cevresindeki geometri):
mevcut/ham direction, uye yonleri, B-rep silindir ekseni, channel ekseni, surface normali,
mouth duzlemi eksenleri.

EKSIK OLAN: PARCA DUZEYI. Klemens guclu a global yonelime sahiptir -- teller parcanin
ON YUZUNDEN girer and takma yonu genellikle parcanin KENDI eksenleriyle hizalidir. Ayni
yuzdeki butun CP'ler AYNI yone bakar. Bu bilgi calisma aninda mevcuttur (GT gerekmez).

EKLENEN GIRISLER (all of them calisma aninda turetilebilir):
    obb_x/y/z (+/-)   parcanin yonelimli boundary kutusu eksenleri (6 direction)
    uzlasi            parcadaki TUM adaylarin direction medyani (most large cluster)
    yuz_uzlasi        adayla AYNI YUZDEKI adaylarin direction uzlasisi
    duzlem_obb        adayin eksenine DIK duzlemde, OBB eksenlerine most yakin 4 direction

ONCE DIAGNOSIS: GT yonu ne up to siklikla a OBB eksenine yakin? Cevap yuksekse arm canli.

KILL (C1 with same bar): genisletilmis kahin < 0.70 whereas Rota C GERCEKTEN kapanir.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def _birim (v ):
    v =np .asarray (v ,float );n =np .linalg .norm (v )
    return v /n if n >1e-9 else None 


def obb_eksenleri (V ):
    """Parcanin yonelimli boundary kutusu eksenleri (PCA)."""
    Q =V -V .mean (0 )
    try :
        _ ,_ ,W =np .linalg .svd (Q ,full_matrices =False )
        return [_birim (W [i ])for i in range (3 )]
    except Exception :
        return []


def uzlasi_yonu (D ,wgt_ =None ):
    """Yonlerin EN BUYUK kumesinin medyani (sign duyarli not -> hizala)."""
    if not len (D ):
        return None 
    ref =D [0 ]
    A =np .array ([d if float (d @ref )>=0 else -d for d in D ])
    # most large cluster: each yona 20 derece inside kac komsu present
    C =np .abs (A @A .T )>=np .cos (np .radians (20.0 ))
    i =int (C .sum (1 ).argmax ())
    grup =A [C [i ]]
    return _birim (grup .mean (0 ))


def main ():
    import gate_bench as T 
    import measure_set 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from gece_kilit import guard 

    guard ("c2")
    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    X ,y ,pid ,keep =D ["X"],D ["y"],D ["pid"],D ["keep"]
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],y [keep ]),
    "n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    TESHIS =[]
    rob_mevcut ,rob_k1 ,rob_k2 ,gg =[],[],[],[]
    say ={}
    t0 =time .time ()
    for kk ,r in enumerate (D ["DER"],1 ):
        if kk %25 ==0 :
            print (f"  {kk }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        SOZ1 ,SOZ2 =[],[]
        if r ["X"]is not None and r .get ("XR")is not None :
            Xr =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
            if k .any ():
                P =r ["P"][k ].copy ();Pham =r ["Pd"][k ].copy ()
                c =[{"point":P [i ],"direction":Pham [i ]}for i in range (len (P ))]
                c =wire_gate .pose_correct (Xr [k ],c )
                if cfg .get ("robot_aci_secici"):
                    c =wire_gate .angle_correct (Xr [k ],c )
                if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                    c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
                try :
                    Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                    V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                    V =np .ascontiguousarray (V ,float )
                except Exception :
                    V =None 
                OBB =obb_eksenleri (V )if V is not None else []
                UZ =uzlasi_yonu (Pd )
                for i in range (len (P )):
                    d0 =_birim (Pd [i ])
                    S1 ={"mevcut":d0 ,"ham":_birim (Pham [i ])}
                    if r .get ("UYE"):
                        n_u =0 
                        for lst in r ["UYE"]:
                            for cc in (lst if isinstance (lst ,(list ,tuple ))else []):
                                if not isinstance (cc ,dict ):
                                    continue 
                                pt =np .asarray (cc .get ("point",P [i ]),float )
                                dd =cc .get ("direction")
                                if dd is None or np .linalg .norm (pt -P [i ])>5.0 :
                                    continue 
                                b =_birim (dd )
                                if b is not None :
                                    S1 [f"uye{n_u }"]=b ;n_u +=1 
                                if n_u >=4 :
                                    break 
                    S2 =dict (S1 )
                    for j ,a in enumerate (OBB ):
                        if a is not None :
                            S2 [f"obb{j }"]=a ;S2 [f"obb{j }_"]=-a 
                    if UZ is not None :
                        S2 ["uzlasi"]=UZ 
                        # AYNI YUZ: most large OBB ekseni along benzer konumdakiler
                    if OBB and len (P )>2 :
                        n0 =OBB [0 ]
                        t_ =P @n0 
                        same_ =np .abs (t_ -t_ [i ])<=3.0 
                        if same_ .sum ()>=2 :
                            fu =uzlasi_yonu (Pd [same_ ])
                            if fu is not None :
                                S2 ["yuz_uzlasi"]=fu 
                                # adayin eksenine DIK duzlemde, OBB eksenlerine most yakin yonler
                    if d0 is not None :
                        for j ,a in enumerate (OBB ):
                            if a is None :
                                continue 
                            pr =_birim (a -float (a @d0 )*d0 )
                            if pr is not None :
                                S2 [f"dik{j }"]=pr ;S2 [f"dik{j }_"]=-pr 
                    SOZ1 .append ({a :b for a ,b in S1 .items ()if b is not None })
                    SOZ2 .append ({a :b for a ,b in S2 .items ()if b is not None })
                    for a in SOZ2 [-1 ]:
                        kk2 =a .rstrip ("0123456789_")if (a .startswith ("uye")or a .startswith ("obb")or a .startswith ("dik"))else a 
                        say [kk2 ]=say .get (kk2 ,0 )+1 
        rob_mevcut .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        for SOZ ,hedef in ((SOZ1 ,rob_k1 ),(SOZ2 ,rob_k2 )):
            Pk =Pd .copy ()
            if len (P )and len (G )and SOZ :
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                for i in range (len (P )):
                    if not np .isfinite (pe [i ]).any ():
                        continue 
                    b =int (np .argmin (pe [i ]))
                    en ,ed =None ,None 
                    for a ,v in SOZ [i ].items ():
                        ang =abs (float (v @Gd [b ]))
                        if en is None or ang >en :
                            en ,ed =ang ,v 
                    if ed is not None :
                        Pk [i ]=ed 
                    if SOZ is SOZ2 and len (TESHIS )<100000 :
                        OBBl =[v for a ,v in SOZ [i ].items ()if a .startswith ("obb")]
                        if OBBl :
                            TESHIS .append (max (abs (float (v @Gd [b ]))for v in OBBl ))
            hedef .append ((rj ,)+esle (P ,Pk ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        gg .append (r ["geo"])

    if TESHIS :
        A =np .degrees (np .arccos (np .clip (np .array (TESHIS ),0 ,1 )))
        print (f"\nTESHIS: GT yonunun EN YAKIN OBB eksenine acisi ({len (A )} cift)")
        print (f"  medyan {np .median (A ):.1f} deg | <=10 deg {float ((A <=10 ).mean ()):.1%} | "
        f"<=20 deg {float ((A <=20 ).mean ()):.1%}")
    print (f"\nsozluk kapsami: {dict (sorted (say .items (),key =lambda x :-x [1 ]))}")
    rm =T .f1w (rob_mevcut );r1 =T .f1w (rob_k1 );r2 =T .f1w (rob_k2 )
    ust =0.7584 
    print (f"\n{'arm':<28}{'robot':>9}")
    print (f"{'MEVCUT':<28}{rm :>9.4f}")
    print (f"{'KAHIN yerel sozluk (C1)':<28}{r1 :>9.4f}   direction boslugunun %{100 *(r1 -rm )/max (ust -rm ,1e-9 ):.0f}'i")
    print (f"{'KAHIN +PARCA DUZEYI (C2)':<28}{r2 :>9.4f}   direction boslugunun %{100 *(r2 -rm )/max (ust -rm ,1e-9 ):.0f}'i")
    print (f"{'ust sinir (=tespit)':<28}{ust :>9.4f}")
    gecti =r2 >=0.70 
    print (f"\nKILL: kahin >= 0.70 -> {'GECTI, ayrik selector egitilir'if gecti else 'GECMEDI'}")
    with io .open ("results/c2_parca_yon.json","w",encoding ="utf-8")as f :
        json .dump ({"mevcut":rm ,"kahin_yerel":r1 ,"kahin_parca":r2 ,
        "obb_aci_medyan":float (np .median (A ))if TESHIS else None ,
        "obb_10deg":float ((A <=10 ).mean ())if TESHIS else None ,
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/c2_parca_yon.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
