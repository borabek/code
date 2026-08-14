# -*- coding: utf-8 -*-
"""C4: AYRIK YON SECICI -- kahini kazanca cevirme denemesi.

RATIONALE (r3'te duzeltildi): direction sozlugu, KENDI tavaninin %90'ini kapsiyor.
    mevcut robot                     0.5893
    direction sozlugu KAHINI               0.6635
    only aciyi duzeltmenin tavani  0.6717
Yani dictionary neredeyse full; is SECMEYE indi. (Onceki kill bari 0.70 YAPISAL OLARAK
IMKANSIZDI -- aciyi duzelterek 0.6717'nin ustune cikilamaz. Bar yanlisti, arm not.)

WHY AYRIK, WHY REGRESYON DEGIL: R2'de surekli regresor each esikte DUSTU (-0.0256).
Burada model a YON URETMIYOR; sozlukteki adaylardan BIRINI seciyor. Uretemedigi a
yonu uydurma riski absent.

TASARIM:
  * each (candidate, dictionary girisi) cifti a SATIR
  * label: this giris GT yonune <=10 derece mi (only eslesen ciftlerde tanimli)
  * ozellikler: adayin 58 gate sutunu + girise OZGU 8 column (source tipi one-hot,
    mevcut yonle angle, part uzlasisiyla angle, dictionary-ici order)
  * inference: part-ici GRUP-CAPRAZ OOF skoru -> candidate basina ARGMAX giris
  * SAFETY: selector, mevcut yonden however skoru BELIRGIN yuksekse sapar (marj esigi),
    otherwise mevcut korunur -- "herkese uygula" tuzagi this havuzda four arm oldurdu.

KILL (onceden): robot +0.01 VE grup bootstrap GA'si sifiri dislamali. Tespit YAPISAL
as does not change (tespit olcutu aciya bakmaz) but yine de olculur. Uretici-disi dusmemeli.
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

ONBELLEK ="results/c4_sozluk.pkl"
KAYNAK =["mevcut","ham","uye","obb","dik","uzlasi","yuz_uzlasi","yuzn"]


def _birim (v ):
    v =np .asarray (v ,float );n =np .linalg .norm (v )
    return v /n if n >1e-9 else None 


def main ():
    import gate_bench as T 
    import measure_set 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from c2_part_level_direction import obb_eksenleri ,uzlasi_yonu 
    from infer_step_cp import step_to_mesh 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from gece_kilit import guard 

    guard ("c4")
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

    # ---------- 1) SOZLUKLERI KUR (onbellekli)
    if os .path .exists (ONBELLEK ):
        with open (ONBELLEK ,"rb")as f :
            PARCA =pickle .load (f )
        print (f"sozluk onbellekten: {len (PARCA )} part",flush =True )
    else :
        PARCA ={}
        t0 =time .time ()
        for kk ,r in enumerate (D ["DER"],1 ):
            if kk %25 ==0 :
                print (f"  sozluk {kk }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
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
            OBB =obb_eksenleri (V )if V is not None else []
            UZ =uzlasi_yonu (Pd )
            YUZN =[]
            try :
                import brep_axes as _ba 
                pl =_ba .planes (stp [r ["pid"]])
                if pl is not None and len (pl ):
                    _n =np .asarray (pl [1 ],float );_rr =np .asarray (pl [2 ],float )
                    for j in np .argsort (-_rr )[:6 ]:
                        b =_birim (_n [j ])
                        if b is not None :
                            YUZN .append (b )
            except Exception :
                pass 
            SOZ =[]
            for i in range (len (P )):
                d0 =_birim (Pd [i ])
                S =[("mevcut",d0 ),("ham",_birim (Dham [i ]))]
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
                                S .append (("uye",b ));n_u +=1 
                            if n_u >=4 :
                                break 
                for a in OBB :
                    if a is not None :
                        S +=[("obb",a ),("obb",-a )]
                if UZ is not None :
                    S .append (("uzlasi",UZ ))
                if OBB and len (P )>2 :
                    t_ =P @OBB [0 ]
                    ay =np .abs (t_ -t_ [i ])<=3.0 
                    if ay .sum ()>=2 :
                        fu =uzlasi_yonu (Pd [ay ])
                        if fu is not None :
                            S .append (("yuz_uzlasi",fu ))
                if d0 is not None :
                    for a in OBB :
                        if a is None :
                            continue 
                        pr =_birim (a -float (a @d0 )*d0 )
                        if pr is not None :
                            S +=[("dik",pr ),("dik",-pr )]
                for a in YUZN :
                    S +=[("yuzn",a ),("yuzn",-a )]
                SOZ .append ([(t_ ,v )for t_ ,v in S if v is not None ])
            PARCA [r ["pid"]]={"P":P ,"Pd":Pd ,"X":Xr [k ],"SOZ":SOZ ,
            "UZ":UZ if UZ is not None else np .zeros (3 )}
        with open (ONBELLEK ,"wb")as f :
            pickle .dump (PARCA ,f )
        print (f"-> {ONBELLEK }",flush =True )

        # ---------- 2) EGITIM MATRISI
    RX ,RY ,RG ,RP ,RI ,RJ =[],[],[],[],[],[]
    for r in D ["DER"]:
        d_ =PARCA .get (r ["pid"])
        if d_ is None :
            continue 
        P ,Pd ,Xk ,SOZ =d_ ["P"],d_ ["Pd"],d_ ["X"],d_ ["SOZ"]
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
                continue # only TESPIT EDILMIS ciftlerde label present
            d0 =Pd [i ]
            for gi ,(tip ,v )in enumerate (SOZ [i ]):
                aci =float (np .degrees (np .arccos (np .clip (abs (float (v @Gd [b ])),0 ,1 ))))
                oz =[1.0 if tip ==t2 else 0.0 for t2 in KAYNAK ]
                oz .append (float (np .degrees (np .arccos (np .clip (abs (float (v @d0 )),0 ,1 )))))
                uz =d_ ["UZ"]
                oz .append (float (np .degrees (np .arccos (np .clip (abs (float (v @uz )),0 ,1 ))))
                if np .linalg .norm (uz )>0.5 else 90.0 )
                oz .append (float (gi ));oz .append (float (len (SOZ [i ])))
                RX .append (np .concatenate ([Xk [i ],oz ]))
                RY .append (int (aci <=10.0 ));RG .append (r ["geo"]);RP .append (r ["pid"])
                RI .append ((r ["pid"],i ));RJ .append (gi )
    RX =np .array (RX ,float );RY =np .array (RY )
    RG =np .array (RG );RJ =np .array (RJ )
    print (f"\negitim satiri {len (RY )} | dogru giris orani {RY .mean ():.1%} | "
    f"sutun {RX .shape [1 ]}")

    # ---------- 3) OOF SECICI
    o =np .zeros (len (RY ))
    for tr ,te in GroupKFold (n_splits =5 ).split (RX ,RY ,RG ):
        o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (RX [tr ],RY [tr ]).predict_proba (RX [te ])[:,1 ]
    SKOR ={}
    for n_ ,(key ,gi )in enumerate (zip (RI ,RJ )):
        SKOR .setdefault (key ,{})[gi ]=o [n_ ]

        # ---------- 4) UCTAN UCA (marj esigi taranir, DURUST capraz secimle)
    def puanla (marj ):
        rob ,det ,gg =[],[],[]
        for r in D ["DER"]:
            d_ =PARCA .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            if d_ is None :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            else :
                P =d_ ["P"];Pd =d_ ["Pd"].copy ()
                for i in range (len (P )):
                    s =SKOR .get ((r ["pid"],i ))
                    if not s :
                        continue 
                    gi =max (s ,key =s .get )
                    s0 =s .get (0 ,0.0 )# 0 = 'mevcut'
                    if gi !=0 and s [gi ]-s0 >=marj :
                        Pd [i ]=d_ ["SOZ"][i ][gi ][1 ]
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg 
    print (f"\n{'marj':<8}{'robot':>9}{'tespit':>9}")
    EN =None 
    for marj in (0.02 ,0.05 ,0.10 ,0.15 ,0.20 ,0.30 ):
        rob ,det ,gg =puanla (marj )
        print (f"{marj :<8.2f}{T .f1w (rob ):>9.4f}{T .f1w (det ):>9.4f}")
        if EN is None or T .f1w (rob )>EN [1 ]:
            EN =(marj ,T .f1w (rob ),rob ,det ,gg )
    rob0 ,det0 ,gg0 =puanla (1e9 )# no deviation = MEVCUT
    baseline =T .f1w (rob0 )
    marj ,en ,rob ,det ,gg =EN 
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (rob0 ,rob )),gg ,fn ,n =3000 )
    print (f"\nTABAN (deviation yok) {baseline :.4f} | EN IYI marj {marj :.2f} -> {en :.4f} "
    f"({en -baseline :+.4f})")
    print (f"  GA[{lo :+.4f},{hi :+.4f}] | tespit {T .f1w (det0 ):.4f} -> {T .f1w (det ):.4f}")
    gecti =(en -baseline )>=0.01 and lo >0 
    print (f"\nKILL: robot +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/c4_yon_secici.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":baseline ,"en_iyi":en ,"marj":marj ,"fark":en -baseline ,
        "ga":[lo ,hi ],"tespit_once":T .f1w (det0 ),"tespit_sonra":T .f1w (det ),
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/c4_yon_secici.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
