# -*- coding: utf-8 -*-
"""R9: YON SECICISINE ISARET GOREN OZNITELIK VER -- kalan +0.0399'un pesinde.

R8 OLCUMU: dagitilan zincir 0.6181; D1 sozlugunun ISARETLI tavani 0.6580. Yakalanan
pay %48, kalan **+0.0399**.

ROOT CAUSE (koda bakinca gorunuyor, `d1_direction_distribute.satirla`):
    oz.append(... abs(float(v @ d0)) ...)      # ISARETSIZ
    oz.append(... abs(float(v @ UZ)) ...)      # ISARETSIZ
Sozluk BILEREK +- ciftleri iceriyor (obb/dik/yuzn). Bu two feature `+a` and `-a` for
BIREBIR AYNI degeri veriyor. Geriye kalan single difference, girisin DIZIN NUMARASI (`gi`).
Yani model bugun isareti only sozlukteki siradan prediction edebiliyor -- fiziksel
no sign bilgisi GORMUYOR.

Bugun ETIKETI signed yaptim (+0.0363 kazandirdi). OZNITELIKLER still unsigned.
Ayni hatanin ikinci yarisi.

EKLENEN OZNITELIKLER (all of them FIZIKSEL, none of them GT gormez):
    v.d0 ISARETLI          mevcut yonle same tarafta mi (mevcut ~%80 correct -> guclu onsel)
    v.UZ ISARETLI          part uzlasisiyla same tarafta mi
    disariliK              v with (p - part weight merkezi) arasindaki signed uyum
    koni sayimi +-         v yonunde 45 derecelik koni inside MESH TEPESI count, ileri/geri.
                           Agiz DIS yuzeydedir: disari bakan yonde vertex AZ (bosluk),
                           ice bakan yonde COK (body yuzeyleri). "Hangi taraf disarisi"
                           sorusunun dogrudan fiziksel olcumu -- and `+a`/`-a` ayrimi TAM BUDUR.
    same measurement 8mm and 15mm yaricapta (yerel and genis scale)

Bu betik DAGITMAZ: measurement kumesinde GRUP-CAPRAZ as "unsigned features" with
"signed features" secicilerini AYNI katlarda yaristirir. Kazanirsa R10'da training
korpusunda yeniden turetilip dagitilir.
"""
import io ,json ,os ,pickle ,sys ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set ,thesis_remesh ,wire_gate 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from sina_cluster import esle ,f1w 
from d1_direction_distribute import KAYNAK 

ONB ="results/_r9_geo.pkl"
R8 ="results/_r8_sozluk.pkl"
MARJ =0.05 


def koni (V ,p ,v ,R ):
    """v yonunde 45 derecelik koni inside, R yaricapinda MESH TEPESI count (ileri, geri)."""
    rel =V -p 
    n =np .linalg .norm (rel ,axis =1 )
    m =(n >1e-6 )&(n <=R )
    if not m .any ():
        return 0 ,0 
    c =(rel [m ]/n [m ,None ])@v 
    return int ((c >=0.7071 ).sum ()),int ((c <=-0.7071 ).sum ())


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    DER ,gate ,ek =T2 .yukle ()
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    PARCA =pickle .load (open (R8 ,"rb"))

    # ---- geometrik ek bilgi (mesh is required, R8 onbelleginde absent)
    if os .path .exists (ONB ):
        GEO =pickle .load (open (ONB ,"rb"))
        print (f"geo onbellekten: {len (GEO )} part",flush =True )
    else :
        GEO ={};t0 =time .time ()
        for kk ,(pid ,d_ )in enumerate (PARCA .items (),1 ):
            if kk %20 ==0 :
                print (f"  geo {kk }/{len (PARCA )}  {time .time ()-t0 :.0f}s",flush =True )
                pickle .dump (GEO ,open (ONB ,"wb"))
            try :
                Vr ,Fr =step_to_mesh (stp [pid ])
                V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,float )
            except Exception :
                continue 
            mrk =V .mean (0 )
            E =[]
            for i in range (len (d_ ["P"])):
                p =d_ ["P"][i ]
                dis =p -mrk 
                nd =np .linalg .norm (dis )
                dis =dis /nd if nd >1e-9 else np .zeros (3 )
                row =[]
                for _t ,v in d_ ["SD"][i ]:
                    i8 ,g8 =koni (V ,p ,v ,8.0 )
                    i15 ,g15 =koni (V ,p ,v ,15.0 )
                    row .append ([float (v @dis ),i8 ,g8 ,(g8 -i8 )/(g8 +i8 +1.0 ),
                    i15 ,g15 ,(g15 -i15 )/(g15 +i15 +1.0 )])
                E .append (np .array (row ,float )if row else np .zeros ((0 ,7 )))
            GEO [pid ]=E 
        pickle .dump (GEO ,open (ONB ,"wb"))
        print (f"-> {ONB }",flush =True )

        # ---- satirlari kur: ESKI (unsigned) and YENI (signed) feature kumeleri
    EX ,YX ,LY ,LG ,LK =[],[],[],[],[]
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        if d_ is None :
            continue 
        E =GEO .get (r ["pid"])
        P =d_ ["P"];Pd =d_ ["Pd"];X =d_ ["X"];UZ =d_ ["UZ"]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (P )or not len (G ):
            continue 
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        uzn =float (np .linalg .norm (UZ ))
        for i in range (len (P )):
            if not np .isfinite (pe [i ]).any ():
                continue 
            b =int (np .argmin (pe [i ]))
            if pe [i ,b ]>tol :
                continue # label only eslesen adaylarda tanimli
            d0 =Pd [i ]
            for gi ,(tip ,v )in enumerate (d_ ["SD"][i ]):
                oh =[1.0 if tip ==t2 else 0.0 for t2 in KAYNAK ]
                # --- ESKI (dagitilan) features: ikisi de abs()
                esk =oh +[float (np .degrees (np .arccos (np .clip (abs (float (v @d0 )),0 ,1 )))),
                float (np .degrees (np .arccos (np .clip (abs (float (v @UZ )),0 ,1 ))))
                if uzn >0.5 else 90.0 ,
                float (gi ),float (len (d_ ["SD"][i ]))]
                # --- YENI: same + ISARETLI olcumler + fiziksel disari testi
                ek =[float (v @d0 ),
                float (v @UZ )if uzn >0.5 else 0.0 ,
                float (np .degrees (np .arccos (np .clip (float (v @d0 ),-1 ,1 )))),
                float (np .degrees (np .arccos (np .clip (float (v @UZ ),-1 ,1 ))))
                if uzn >0.5 else 90.0 ]
                ek +=list (E [i ][gi ])if (E is not None and i <len (E )and gi <len (E [i ]))else [0.0 ]*7 
                EX .append (np .concatenate ([X [i ],esk ]))
                YX .append (np .concatenate ([X [i ],esk ,ek ]))
                LY .append (int (np .degrees (np .arccos (np .clip (float (v @Gd [b ]),-1 ,1 )))<=10.0 ))
                LG .append (r ["geo"]);LK .append ((r ["pid"],i ,gi ))
    EX =np .array (EX ,float );YX =np .array (YX ,float )
    LY =np .array (LY );LG =np .array (LG )
    print (f"\nsatir {len (LY )} | dogru giris {LY .mean ():.1%} | "
    f"sutun ESKI {EX .shape [1 ]} / YENI {YX .shape [1 ]}",flush =True )

    ug =np .unique (LG );rng =np .random .RandomState (0 );rng .shuffle (ug )
    fold ={g :j %5 for j ,g in enumerate (ug )}
    kk_ =np .array ([fold [g ]for g in LG ])
    from sklearn .metrics import roc_auc_score 
    OOF ={}
    for ad ,M in (("ESKI",EX ),("YENI",YX )):
        o =np .zeros (len (LY ))
        for f_ in range (5 ):
            tr =kk_ !=f_ 
            c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],LY [tr ])
            o [~tr ]=c .predict_proba (M [~tr ])[:,1 ]
        OOF [ad ]=o 
        print (f"{ad }: grup-capraz AUC {roc_auc_score (LY ,o ):.4f}",flush =True )

    SC ={ad :{}for ad in OOF }
    for ad ,o in OOF .items ():
        for j ,(pid ,i ,gi )in enumerate (LK ):
            SC [ad ].setdefault ((pid ,i ),{})[gi ]=o [j ]

    def kos (ad ,marj ):
        rob ,det ,gg =[],[],[]
        for r in DER :
            d_ =PARCA .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            if d_ is None :
                P =np .zeros ((0 ,3 ));Pn =np .zeros ((0 ,3 ))
            else :
                P =d_ ["P"];Pn =d_ ["Pd"].copy ()
                if ad is not None :
                    for i in range (len (P )):
                        s =SC [ad ].get ((r ["pid"],i ))
                        if not s :
                            continue 
                        g =max (s ,key =s .get )
                        if g !=0 and s [g ]-s .get (0 ,0.0 )>=marj :
                            Pn [i ]=d_ ["SD"][i ][g ][1 ]
            rob .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            det .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg 

    r0 ,d0_ ,gg =kos (None ,9.0 )
    b_rob ,b_det =f1w (r0 ),f1w (d0_ )
    print (f"\ntaban (dagitilan zincir): robot(FIZ) {b_rob :.4f} | tespit {b_det :.4f}")
    print (f"ISARETLI ceiling (R8): 0.6580 -> kalan alan {0.6580 -b_rob :+.4f}")
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    print (f"\n{'oznitelik':<8}{'marj':>6}{'robot':>10}{'d':>9}{'GA':>22}{'tespit':>10}{'pay':>7}")
    SON ={}
    for ad in ("ESKI","YENI"):
        for marj in (0.02 ,0.05 ,0.10 ):
            ra ,da ,_ =kos (ad ,marj )
            _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (r0 ,ra )),gg ,fn ,n =1500 )
            dr =f1w (ra )-b_rob 
            pay =100 *dr /max (0.6580 -b_rob ,1e-9 )
            print (f"{ad :<8}{marj :>6.2f}{f1w (ra ):>10.4f}{dr :>+9.4f}"
            f"   [{lo :+.4f},{hi :+.4f}]{f1w (da ):>10.4f}{pay :>6.0f}%")
            SON [f"{ad }_{marj }"]={"oznitelik":ad ,"marj":marj ,"robot":f1w (ra ),
            "d_robot":dr ,"ga":[lo ,hi ],"tespit":f1w (da ),
            "kalan_pay_yuzde":pay }
    e_iyi =max ((v for k ,v in SON .items ()if v ["oznitelik"]=="ESKI"),
    key =lambda v :v ["d_robot"])
    y_iyi =max ((v for k ,v in SON .items ()if v ["oznitelik"]=="YENI"),
    key =lambda v :v ["d_robot"])
    print (f"\nESKI en iyi {e_iyi ['d_robot']:+.4f} | YENI en iyi {y_iyi ['d_robot']:+.4f} "
    f"| FARK {y_iyi ['d_robot']-e_iyi ['d_robot']:+.4f}")
    deger =y_iyi ["d_robot"]-e_iyi ["d_robot"]>=0.01 
    print (f"\nHUKUM: {'ISARETLI OZNITELIKLER KAZANDIRIYOR -> R10 (training korpusunda yeniden turet + dagit)'if deger else 'signed oznitelikler anlamli katki vermiyor'}")
    with io .open ("results/r9_isaretli_ozellik.json","w",encoding ="utf-8")as f :
        json .dump ({"taban_robot":b_rob ,"taban_tespit":b_det ,"ceiling":0.6580 ,
        "auc":{ad :float (roc_auc_score (LY ,o ))for ad ,o in OOF .items ()},
        "tarama":SON ,"eski_en_iyi":e_iyi ,"yeni_en_iyi":y_iyi ,
        "deger":bool (deger )},f ,indent =1 )
    print ("receipt -> results/r9_isaretli_ozellik.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
