# -*- coding: utf-8 -*-
"""K4: AYRIK KONUM SECICI + IKISI BIRLIKTE -- robot metriginin last kolu.

RATIONALE: zengin konum sozlugu KAHINI 0.6244 (lateral tavani 0.6469, kapsam %61).
Sozluk-kapsami barim (%70) kacirildi AMA that bar YANLIS ALETTIR: direction kolunda same sinifta
a bar (0.70) YAPISAL OLARAK IMKANSIZ output and arm yine de +0.0238 KANITLI kazanc verdi.
Dogru criterion SECICININ UCTAN UCA kazancidir and that bar DEGISMIYOR: robot +0.01 VE GA>0.

TASARIM (C4 with same ayrik mantik, konum semantigiyle):
  * each (candidate, konum girisi) cifti a SATIR
  * label: this giris GT'ye YANAL as <=2mm mi
  * ozellikler: adayin 58 gate sutunu + girise ozgu (source tipi one-hot, mevcut noktaya
    uzaklik, mouth merkezine uzaklik, dictionary-ici order/size)
  * inference: grup-capraz OOF -> candidate basina ARGMAX; MARJ esigiyle mevcut korunur
  * marj DURUST secilir (yarida sec / diger yaride olc)

SON OLCUM: KONUM selector + YON selector (C4) BIRLIKTE -- robot metriginin ulasabildigi yer.
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ONB ="results/k4_konum_sozluk.pkl"
KAYNAK =["mevcut","ham","uye","mouth","kirpik","cember","open",
"kesit","acik3","acik5","axis"]


def _tip (a ):
    for t in ("uye","kesit"):
        if a .startswith (t ):
            return t 
    return a if a in KAYNAK else "kesit"


def main ():
    import gate_bench as T 
    import measure_set 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 
    from k23_position_dictionary import konum_sozlugu 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from night_kilit import guard 

    guard ("k4")
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

    if os .path .exists (ONB ):
        with open (ONB ,"rb")as f :
            PARCA =pickle .load (f )
        print (f"konum sozlugu onbellekten: {len (PARCA )} part",flush =True )
    else :
        PARCA ={}
        t0 =time .time ()
        for kk ,r in enumerate (D ["DER"],1 ):
            if kk %25 ==0 :
                print (f"  dictionary {kk }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
            if r ["X"]is None or r .get ("XR")is None :
                continue 
            Xr =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
            if not k .any ():
                continue 
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
            SOZ =[]
            for i in range (len (P )):
                S =[("mevcut",P [i ]),("ham",Pham [i ])]
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
                            S .append ((f"uye{n_u }",pt ));n_u +=1 
                            if n_u >=4 :
                                break 
                S +=list (konum_sozlugu (V ,P [i ],Pd [i ],cyl ,ba ).items ())
                SOZ .append (S )
            PARCA [r ["pid"]]={"P":P ,"Pd":Pd ,"X":Xr [k ],"SOZ":SOZ }
        with open (ONB ,"wb")as f :
            pickle .dump (PARCA ,f )
        print (f"-> {ONB }",flush =True )

    RX ,RY ,RG ,RI ,RJ =[],[],[],[],[]
    for r in D ["DER"]:
        d_ =PARCA .get (r ["pid"])
        if d_ is None :
            continue 
        P ,Xk ,SOZ =d_ ["P"],d_ ["X"],d_ ["SOZ"]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        for i in range (len (P )):
            if not np .isfinite (pe [i ]).any ():
                continue 
            b =int (np .argmin (pe [i ]))
            if pe [i ,b ]>max (3.0 ,0.06 *float (r ["diag"])):
                continue 
            mouth =None 
            for a ,q in SOZ [i ]:
                if a =="mouth":
                    mouth =np .asarray (q ,float )
            for gi ,(a ,q )in enumerate (SOZ [i ]):
                q =np .asarray (q ,float )
                w =q -G [b ]
                yan =float (np .linalg .norm (w -float (w @Gd [b ])*Gd [b ]))
                feat =[1.0 if _tip (a )==t2 else 0.0 for t2 in KAYNAK ]
                feat .append (float (np .linalg .norm (q -P [i ])))
                feat .append (float (np .linalg .norm (q -mouth ))if mouth is not None else 9.0 )
                feat .append (float (gi ));feat .append (float (len (SOZ [i ])))
                RX .append (np .concatenate ([Xk [i ],feat ]))
                RY .append (int (yan <=2.0 ));RG .append (r ["geo"])
                RI .append ((r ["pid"],i ));RJ .append (gi )
    RX =np .array (RX ,float );RY =np .array (RY );RG =np .array (RG );RJ =np .array (RJ )
    print (f"\negitim satiri {len (RY )} | correct giris orani {RY .mean ():.1%} | sutun {RX .shape [1 ]}")

    o =np .zeros (len (RY ))
    for tr ,te in GroupKFold (n_splits =5 ).split (RX ,RY ,RG ):
        o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (RX [tr ],RY [tr ]).predict_proba (RX [te ])[:,1 ]
    SK ={}
    for n_ ,(key ,gi )in enumerate (zip (RI ,RJ )):
        SK .setdefault (key ,{})[gi ]=o [n_ ]

        # --- YON secicisi (C4) skorlarini da yukle
    YON =None 
    if os .path .exists ("results/c4_sozluk.pkl"):
        with open ("results/c4_sozluk.pkl","rb")as f :
            YSOZ =pickle .load (f )
        import c4_direction_selector as _c4 # noqa: F401  (only KAYNAK for)
        YON =YSOZ 
        print ("direction sozlugu yuklendi (birlesik measurement for)")

    def puanla (marj ,alt ,yon_uygula =False ,yon_marj =0.05 ,yon_skor =None ):
        rob ,det ,gg =[],[],[]
        for r in alt :
            d_ =PARCA .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            if d_ is None :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            else :
                P =d_ ["P"].copy ();Pd =d_ ["Pd"].copy ()
                for i in range (len (P )):
                    s =SK .get ((r ["pid"],i ))
                    if s :
                        gi =max (s ,key =s .get );s0 =s .get (0 ,0.0 )
                        if gi !=0 and s [gi ]-s0 >=marj :
                            P [i ]=np .asarray (d_ ["SOZ"][i ][gi ][1 ],float )
                    if yon_uygula and yon_skor is not None and YON is not None :
                        ys =yon_skor .get ((r ["pid"],i ))
                        yd =YON .get (r ["pid"])
                        if ys and yd is not None and i <len (yd ["SOZ"]):
                            gj =max (ys ,key =ys .get );y0 =ys .get (0 ,0.0 )
                            if gj !=0 and ys [gj ]-y0 >=yon_marj :
                                Pd [i ]=yd ["SOZ"][i ][gj ][1 ]
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg 

    gruplar =sorted ({r ["geo"]for r in D ["DER"]})
    rng =np .random .default_rng (0 );kar =list (gruplar );rng .shuffle (kar )
    A =set (kar [:len (kar )//2 ])
    altA =[r for r in D ["DER"]if r ["geo"]in A ]
    altB =[r for r in D ["DER"]if r ["geo"]not in A ]
    MARJ =(0.02 ,0.05 ,0.10 ,0.15 ,0.20 ,0.30 )
    rc ,rt ,gg2 =[],[],[]
    for sec ,olc in ((altA ,altB ),(altB ,altA )):
        en ,ea =-1 ,MARJ [0 ]
        for m in MARJ :
            rr ,_ ,_ =puanla (m ,sec )
            if T .f1w (rr )>en :
                en ,ea =T .f1w (rr ),m 
        r1 ,_ ,g1 =puanla (ea ,olc );r0 ,_ ,_ =puanla (1e9 ,olc )
        rc +=r1 ;rt +=r0 ;gg2 +=g1 
        print (f"  yarida selected marj {ea :.2f} -> diger yaride measured ({len (olc )} part)")
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (rt ,rc )),gg2 ,fn ,n =3000 )
    dk =T .f1w (rc )-T .f1w (rt )
    print (f"\nKONUM SECICI (durust marj): {T .f1w (rt ):.4f} -> {T .f1w (rc ):.4f} ({dk :+.4f})")
    print (f"  GA[{lo :+.4f},{hi :+.4f}]")
    gecti =dk >=0.01 and lo >0 
    print (f"KILL: robot +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/k4_position_selector.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":T .f1w (rt ),"konum_secici":T .f1w (rc ),"difference":dk ,
        "ga":[lo ,hi ],"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/k4_position_selector.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
