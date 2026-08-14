# -*- coding: utf-8 -*-
"""R3: YANAL TAVAN + DOGRU PAYDA -- robot fazinin never olculmemis yarisi.

IKI IS BIRDEN, TEK KOSUDA (paydalar tutarli olsun diye):

1. YANAL TAVAN (never olculmedi): poz kafasi deployed (+0.0428, CI-kanitli) but
   "konum MUKEMMEL olsa ne olurdu" never sorulmadi. Bir KONUM SOZLUGU kurulur and kahin
   sinavi is done -- direction tarafinda yaptigimizin aynisi.

2. PAYDA DUZELTMESI: C1-C3'te direction sozlugunun kapsamini 0.7584'e bolerek hesapladim.
   O YANLIS. 0.7584 HEM lateral HEM angle mukemmelken ulasilan tavandir; direction sozlugu
   YALNIZ ACI hatalarini duzeltebilir. Dogru payda "ACI MUKEMMEL" tavanidir.

Olculen tavanlar (all of them same eslesme koduyla, same kumede):
    robot MEVCUT              lateral<=2 VE angle<=10
    ACI MUKEMMEL tavani       only lateral<=2   (angle serbest)
    YANAL MUKEMMEL tavani     only angle<=10    (lateral serbest -> tespit toleransi)
    ikisi mukemmel            = tespit F1

KONUM SOZLUGU (all of them calisma aninda turetilebilir, fiziksel):
    mevcut      poz kafasi ciktisi
    ham         duzeltmesiz candidate noktasi
    uye_k       uye havuzundaki points
    mouth        yerel mouth boundary noktalarinin ortalamasi (tezin v_o'this, yeniden hesap)
    axis       B-rep silindir EKSENINE dik izdusum (lateral hatayi dogrudan hedefler)
    kesit_d     channel kesitinin d mm derinlikteki weight merkezi (d = 1,3,5)

KILL: konum kahini, ACI-MUKEMMEL tavaninin %70'ini kapsamiyorsa lateral kolu kapanir.
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


def konum_sozlugu (V ,p ,d ,cyl ,ba ):
    """Aday for fiziksel KONUM adaylari."""
    S ={}
    d =_birim (d )
    if d is None or V is None :
        return S 
    rel =V -p 
    t =rel @d 
    rad =np .linalg .norm (rel -t [:,None ]*d ,axis =1 )
    # AGIZ: |t|<=1.5mm bandindaki noktalarin ortalamasi (tezin v_o'this)
    m =(np .abs (t )<=1.5 )&(rad <=9.0 )
    if m .sum ()>=6 :
        S ["mouth"]=V [m ].mean (0 )
        # KESIT: d mm derinlikte channel kesitinin merkezi
    for dd in (1.0 ,3.0 ,5.0 ):
        m2 =(np .abs (t +dd )<=1.0 )&(rad <=9.0 )
        if m2 .sum ()>=5 :
            S [f"kesit{int (dd )}"]=V [m2 ].mean (0 )
            # B-REP EKSENINE dik izdusum: lateral hatayi DOGRUDAN hedefler
    if cyl is not None and ba is not None :
        try :
            ax =ba .axis_at (p ,d ,cyl )
            if ax is not None :
                a_ ,c_ =(ax if isinstance (ax ,tuple )else (ax ,None ))
                a_ =_birim (a_ )
                if a_ is not None and c_ is not None :
                    c_ =np .asarray (c_ ,float )
                    v =p -c_ 
                    S ["axis"]=c_ +float (v @a_ )*a_ 
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
    from gece_kilit import guard 

    guard ("r3")
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

    KOL ={k :[]for k in ("tespit","mevcut","aci_mukemmel","yanal_mukemmel",
    "konum_kahin","konum_kahin_aci_mukemmel")}
    boy =[]
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
                    SOZ .append (S );boy .append (len (S ))
        KOL ["tespit"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        KOL ["mevcut"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        # ACI MUKEMMEL tavani: angle serbest, lateral<=2
        KOL ["aci_mukemmel"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,180.0 ,False ))
        # YANAL MUKEMMEL tavani: lateral serbest (tespit toleransi), angle<=10
        KOL ["yanal_mukemmel"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,10.0 ,True ))
        # KONUM KAHINI
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
                    v =np .asarray (q ,float )-G [b ]
                    yan =float (np .linalg .norm (v -float (v @Gd [b ])*Gd [b ]))
                    if en is None or yan <en :
                        en ,ep =yan ,np .asarray (q ,float )
                if ep is not None :
                    Pk [i ]=ep 
        KOL ["konum_kahin"].append ((rj ,)+esle (Pk ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        KOL ["konum_kahin_aci_mukemmel"].append (
        (rj ,)+esle (Pk ,Pd ,G ,Gd ,r ["diag"],2.0 ,180.0 ,False ))

    print (f"\nkonum sozlugu boyutu: medyan {np .median (boy ):.0f}")
    F ={k :T .f1w (v )for k ,v in KOL .items ()}
    print (f"\n{'ceiling / arm':<30}{'robot':>9}")
    print (f"{'MEVCUT':<30}{F ['mevcut']:>9.4f}")
    print (f"{'ACI mukemmel olsa (lateral<=2)':<30}{F ['aci_mukemmel']:>9.4f}")
    print (f"{'YANAL mukemmel olsa (angle<=10)':<30}{F ['yanal_mukemmel']:>9.4f}")
    print (f"{'IKISI mukemmel (=tespit)':<30}{F ['tespit']:>9.4f}")
    print (f"{'KONUM KAHINI (dictionary)':<30}{F ['konum_kahin']:>9.4f}")
    ac =F ["aci_mukemmel"];me =F ["mevcut"]
    kaps =(F ["konum_kahin"]-me )/max (ac -me ,1e-9 )
    print (f"\nKONUM sozlugu, ACI-MUKEMMEL tavaninin %{100 *kaps :.0f}'ini kapsiyor")
    print (f"  (dogru payda {ac :.4f}; C1-C3'te yanlislikla {F ['tespit']:.4f} kullanmistim)")
    ya =F ["yanal_mukemmel"]
    print (f"\nDUZELTME -- YON sozlugunun dogru kapsami:")
    print (f"  YON sozlugu yalniz ACI hatalarini duzeltir; tavani {ya :.4f}")
    print (f"  C3 kahini 0.6635 -> kapsam %{100 *(0.6635 -me )/max (ya -me ,1e-9 ):.0f}"
    f"  (once %44 demistim -- YANLIS PAYDA)")
    gecti =kaps >=0.70 
    print (f"\nKILL: konum kahini ACI-MUKEMMEL tavaninin >=%70'ini kapsamali -> "
    f"{'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/r3_yanal_tavan.json","w",encoding ="utf-8")as f :
        json .dump ({k :float (v )for k ,v in F .items ()}|
        {"konum_kapsam":float (kaps ),"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/r3_yanal_tavan.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
