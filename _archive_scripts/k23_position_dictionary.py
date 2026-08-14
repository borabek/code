# -*- coding: utf-8 -*-
"""K2+K3: KONUM SOZLUGUNU ZENGINLESTIR -- lateral tarafin %44'lik kapsamini yukselt.

OLCULEN DURUM (r3): konum sozlugu KAHINI 0.6146, own tavani 0.6469 -> kapsam %44.
Yon sozlugu same olcumde %90 kapsiyordu. Eksik which is taraf BURASI.

K1 SONUCU: lateral error hizalama artigindan GELMIYOR (Spearman +0.049, residual medyani
0.18mm). Yani model hatasi; yatirim mesru.

MEVCUT SOZLUGUN KUSURU: "mouth" girisi, mouth bandindaki noktalarin ORTALAMASI. Hafizada
[[brep-radius-arc-bug]] kaydi present: "silindirik yuzlerin %0'i full kind; centroid tabanli
radius 3.5 fold KUCUKTU; cember oturtma sart". Ayni yanlilik MERKEZI de kaydirir --
kismi ornekleme varsa mean, full tarafa cekilir.

EKLENEN GIRISLER:
    cember     mouth noktalarina EN KUCUK KARELER cember merkezi (mean not)
    open       EN OPEN NOKTA: mouth duzleminde, most yakin malzemeye uzakligi MAKSIMUM which is
               point (Chebyshev merkezi). Kare/yarik agizlarda fiziksel as DOGRU
               tanim budur -- telin gecebilecegi most genis yer.
    kesit7/10  more deep kesit merkezleri (GT kanalin derininde may be)
    kirpik     kirpilmis mean (aykiri boundary noktalarina dayanikli)

KILL: konum kahini, YANAL tavaninin >=%70'ini kapsamiyorsa konum kolu kapanir.
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


def cember_merkezi (Q2 ):
    """2B noktalara EN KUCUK KARELER cember merkezi (Kasa yontemi)."""
    if len (Q2 )<5 :
        return None 
    A =np .hstack ([2 *Q2 ,np .ones ((len (Q2 ),1 ))])
    b =(Q2 **2 ).sum (1 )
    try :
        s ,*_ =np .linalg .lstsq (A ,b ,rcond =None )
        return s [:2 ]
    except Exception :
        return None 


def en_acik_nokta (Q2 ,R =8.0 ,n =41 ):
    """Chebyshev merkezi: most yakin malzemeye uzakligi MAKSIMUM which is point.

    Kare/yarik agizlarda 'centre'in fiziksel tanimi budur -- telin gecebilecegi
    most genis yer. Ortalama, kismi ornekleme varsa full tarafa cekilir.
    """
    if len (Q2 )<6 :
        return None 
    g =np .linspace (-R ,R ,n )
    GX ,GY =np .meshgrid (g ,g ,indexing ="ij")
    P =np .stack ([GX .ravel (),GY .ravel ()],1 )
    d =np .linalg .norm (P [:,None ,:]-Q2 [None ,:,:],axis =-1 ).min (1 )
    ic =np .linalg .norm (P ,axis =1 )<=R 
    d =np .where (ic ,d ,-1.0 )
    return P [int (np .argmax (d ))]


def konum_sozlugu (V ,p ,d ,cyl ,ba ):
    S ={}
    d =_birim (d )
    if d is None or V is None :
        return S 
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    v =np .cross (d ,u )
    rel =V -p 
    t =rel @d 
    rad =np .linalg .norm (rel -t [:,None ]*d ,axis =1 )
    mouth =(np .abs (t )<=1.5 )&(rad <=9.0 )
    if mouth .sum ()>=6 :
        S ["mouth"]=V [mouth ].mean (0 )
        Q =rel [mouth ]
        Q2 =np .stack ([Q @u ,Q @v ],1 )
        # KIRPIK ORTALAMA: aykiri boundary noktalarina dayanikli
        med =np .median (Q2 ,0 )
        dd =np .linalg .norm (Q2 -med ,axis =1 )
        kal =dd <=np .percentile (dd ,80 )
        if kal .sum ()>=5 :
            m2 =Q2 [kal ].mean (0 )
            S ["kirpik"]=p +m2 [0 ]*u +m2 [1 ]*v 
        c2 =cember_merkezi (Q2 )
        if c2 is not None and np .linalg .norm (c2 )<=12.0 :
            S ["cember"]=p +c2 [0 ]*u +c2 [1 ]*v 
        a2 =en_acik_nokta (Q2 )
        if a2 is not None :
            S ["open"]=p +a2 [0 ]*u +a2 [1 ]*v 
    for dep in (1.0 ,3.0 ,5.0 ,7.0 ,10.0 ):
        m2 =(np .abs (t +dep )<=1.0 )&(rad <=9.0 )
        if m2 .sum ()>=5 :
            Q =rel [m2 ]
            Q2 =np .stack ([Q @u ,Q @v ],1 )
            S [f"kesit{int (dep )}"]=V [m2 ].mean (0 )
            a2 =en_acik_nokta (Q2 )
            if a2 is not None and dep in (3.0 ,5.0 ):
                S [f"acik{int (dep )}"]=p +a2 [0 ]*u +a2 [1 ]*v 
    if cyl is not None and ba is not None :
        try :
            ax =ba .axis_at (p ,d ,cyl )
            if ax is not None :
                a_ ,c_ =(ax if isinstance (ax ,tuple )else (ax ,None ))
                a_ =_birim (a_ )
                if a_ is not None and c_ is not None :
                    c_ =np .asarray (c_ ,float )
                    w =p -c_ 
                    S ["axis"]=c_ +float (w @a_ )*a_ 
        except Exception :
            pass 
    return S 


def main ():
    import gate_bench as T 
    import measure_set 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from night_kilit import guard 

    guard ("k23")
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
    try :
        import brep_axes as ba 
    except Exception :
        ba =None 

    KOL ={k :[]for k in ("mevcut","yanal_mukemmel","kahin_eski","kahin_yeni")}
    ESKI ={"mevcut","ham","uye0","uye1","uye2","uye3","mouth",
    "kesit1","kesit3","kesit5","axis"}
    say ={}
    t0 =time .time ()
    for kk ,r in enumerate (D ["DER"],1 ):
        if kk %25 ==0 :
            print (f"  {kk }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "low"
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ));SOZ =[]
        if r ["X"]is not None and r .get ("XR")is not None :
            Xr =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
            if k .any ():
                Pham =r ["P"][k ].copy ();Dham =r ["Pd"][k ].copy ()
                c =[{"point":Pham [i ],"direction":Dham [i ]}for i in range (len (Pham ))]
                c =wire_gate .pose_correct (Xr [k ],c )
                if cfg .get ("robot_aci_secici"):
                    c =wire_gate .angle_correct (Xr [k ],c )
                if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                    c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
                V =None 
                try :
                    Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                    V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                    V =np .ascontiguousarray (V ,float )
                except Exception :
                    V =None 
                cyl =None 
                if ba is not None :
                    try :
                        cyl =ba .cylinders (stp [r ["pid"]])
                    except Exception :
                        cyl =None 
                for i in range (len (P )):
                    S ={"mevcut":P [i ],"ham":Pham [i ]}
                    if r .get ("UYE"):
                        n_u =0 
                        for lst in r ["UYE"]:
                            for cc in (lst if isinstance (lst ,(list ,tuple ))else []):
                                if not isinstance (cc ,dict ):
                                    continue 
                                pt =cc .get ("point")
                                if pt is None :
                                    continue 
                                pt =np .asarray (pt ,float )
                                if np .linalg .norm (pt -P [i ])>6.0 :
                                    continue 
                                S [f"uye{n_u }"]=pt ;n_u +=1 
                                if n_u >=4 :
                                    break 
                    S .update (konum_sozlugu (V ,P [i ],Pd [i ],cyl ,ba ))
                    SOZ .append (S )
                    for a in S :
                        say [a ]=say .get (a ,0 )+1 
        KOL ["mevcut"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        KOL ["yanal_mukemmel"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,10.0 ,True ))
        for ad ,alt in (("kahin_eski",ESKI ),("kahin_yeni",None )):
            Pk =P .copy ()
            if len (P )and len (G )and SOZ :
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                for i in range (len (P )):
                    if not np .isfinite (pe [i ]).any ():
                        continue 
                    b =int (np .argmin (pe [i ]))
                    en ,ep =None ,None 
                    for a ,q in SOZ [i ].items ():
                        if alt is not None and a not in alt :
                            continue 
                        w =np .asarray (q ,float )-G [b ]
                        yan =float (np .linalg .norm (w -float (w @Gd [b ])*Gd [b ]))
                        if en is None or yan <en :
                            en ,ep =yan ,np .asarray (q ,float )
                    if ep is not None :
                        Pk [i ]=ep 
            KOL [ad ].append ((rj ,)+esle (Pk ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))

    F ={k :T .f1w (v )for k ,v in KOL .items ()}
    me =F ["mevcut"];ya =F ["yanal_mukemmel"]
    print (f"\nsozluk kapsami: {dict (sorted (say .items (),key =lambda x :-x [1 ]))}")
    print (f"\n{'arm':<26}{'robot':>9}{'kapsam':>9}")
    print (f"{'MEVCUT':<26}{me :>9.4f}")
    print (f"{'KAHIN old dictionary':<26}{F ['kahin_eski']:>9.4f}"
    f"{100 *(F ['kahin_eski']-me )/max (ya -me ,1e-9 ):>8.0f}%")
    print (f"{'KAHIN ZENGIN dictionary':<26}{F ['kahin_yeni']:>9.4f}"
    f"{100 *(F ['kahin_yeni']-me )/max (ya -me ,1e-9 ):>8.0f}%")
    print (f"{'YANAL tavani':<26}{ya :>9.4f}")
    kaps =(F ["kahin_yeni"]-me )/max (ya -me ,1e-9 )
    gecti =kaps >=0.70 
    print (f"\nKILL: zengin dictionary YANAL tavaninin >=%70'i -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/k23_position.json","w",encoding ="utf-8")as f :
        json .dump ({k :float (v )for k ,v in F .items ()}|
        {"kapsam":float (kaps ),"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/k23_position.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
