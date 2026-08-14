# -*- coding: utf-8 -*-
"""R1: BIRLESIK KAHIN -- robot carpaninin tavani.

robot = 0.777 x tespit (measured). Carpani yukseltmek gecis oranini yukseltmek demek:
this an detected ciftlerin %80.3'u hem lateral<=2mm hem angle<=10 derece geciyor.

AYRI AYRI measured:
   direction sozlugu kahini    -> robot 0.6635 (angle tavani 0.6717'nin %90'i)
   konum sozlugu kahini  -> robot 0.6244 (lateral tavani 0.6469'un %61'i)
BIRLIKTE never olculmedi. %90-95 gecis orani MUMKUN MU, bunu R1 soyler.

Ayrica R2'nin (poz kafasi) savunmasi for gereken number da here: poz kafasi tezin v_o
noktasini ORTALAMA NE KADAR kaydiriyor? Buyukse "agzin inside merkezleme" savunmasi zayif.
"""
import io ,json ,sys ,time 
import numpy as np 
sys .path .insert (0 ,".")
import tezgah2 as T2 
import thesis_remesh ,wire_gate 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from sina_cluster import esle ,f1w 
from k23_konum_sozlugu import konum_sozlugu 
from c2_part_level_direction import obb_eksenleri ,uzlasi_yonu 
from d1_direction_distribute import sozluk_kur ,_birim 

DER ,gate ,ek =T2 .yukle ()
stp ={p :s for m ,p ,jf ,s in eligible ()}
try :
    import brep_axes as ba 
except Exception :
    ba =None 

KAYMA =[]
KOL ={k :[]for k in ("mevcut","yon_kahin","konum_kahin","IKISI")}
t0 =time .time ()
for kk ,r in enumerate (DER ,1 ):
    if kk %30 ==0 :print (f"  {kk }/{len (DER )} {time .time ()-t0 :.0f}s",flush =True )
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    rj ="cok"if r ["n"]>=8 else "dusuk"
    P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ));SY =[];SK =[]
    if r ["X"]is not None and r .get ("XR")is not None :
        X =np .hstack ([r ["X"],r ["XR"]])
        if X .shape [1 ]*2 ==gate ["n_feat"]:
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
            if k .any ():
                Pham =np .asarray (r ["P"],float )[k ].copy ()
                Dham =np .asarray (r ["Pd"],float )[k ].copy ()
                c =[{"point":Pham [i ],"direction":Dham [i ]}for i in range (len (Pham ))]
                c =wire_gate .pose_correct (X [k ],c );c =wire_gate .angle_correct (X [k ],c )
                if r .get ("UYE"):c =wire_gate .pick_member_direction (X [k ],c ,r ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
                KAYMA +=list (np .linalg .norm (P -Pham ,axis =1 ))
                V =None 
                try :
                    Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                    V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                    V =np .ascontiguousarray (V ,float )
                except Exception :V =None 
                cyl =None 
                if ba is not None :
                    try :cyl =ba .cylinders (stp [r ["pid"]])
                    except Exception :cyl =None 
                YUZN =[]
                if ba is not None :
                    try :
                        pl =ba .planes (stp [r ["pid"]])
                        if pl is not None and len (pl ):
                            _n =np .asarray (pl [1 ],float );_rr =np .asarray (pl [2 ],float )
                            for j in np .argsort (-_rr )[:6 ]:
                                b =_birim (_n [j ])
                                if b is not None :YUZN .append (b )
                    except Exception :pass 
                SY ,_uz =sozluk_kur (V ,P ,Pd ,YUZN =YUZN )
                for i in range (len (P )):
                    s ={"mevcut":P [i ]}
                    s .update (konum_sozlugu (V ,P [i ],Pd [i ],cyl ,ba ))
                    SK .append (s )
    def oracle_ (direction ,konum ):
        Pk =P .copy ();Dk =Pd .copy ()
        if len (P )and len (G ):
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            for i in range (len (P )):
                if not np .isfinite (pe [i ]).any ():continue 
                b =int (np .argmin (pe [i ]))
                if direction and i <len (SY )and SY [i ]:
                    en ,ed =None ,None 
                    for _t ,v in SY [i ]:
                        a_ =abs (float (v @Gd [b ]))
                        if en is None or a_ >en :en ,ed =a_ ,v 
                    if ed is not None :Dk [i ]=ed 
                if konum and i <len (SK )and SK [i ]:
                    en ,ep =None ,None 
                    for _a ,q in SK [i ].items ():
                        w =np .asarray (q ,float )-G [b ]
                        yan =float (np .linalg .norm (w -float (w @Gd [b ])*Gd [b ]))
                        if en is None or yan <en :en ,ep =yan ,np .asarray (q ,float )
                    if ep is not None :Pk [i ]=ep 
        return (rj ,)+esle (Pk ,Dk ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False )
    KOL ["mevcut"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
    KOL ["yon_kahin"].append (oracle_ (True ,False ))
    KOL ["konum_kahin"].append (oracle_ (False ,True ))
    KOL ["IKISI"].append (oracle_ (True ,True ))

TESPIT =f1w ([(x [0 ],)+esle (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),1 ,0 ,180 ,True )for x in []])if False else None 
det ,_ ,_ =T2 .puanla (DER ,gate )
tesp =f1w (det )
print (f"\ntespit (ust sinir) {tesp :.4f}")
print (f"{'arm':<22}{'robot':>9}{'gecis orani':>13}")
S ={}
for ad in ("mevcut","yon_kahin","konum_kahin","IKISI"):
    v =f1w (KOL [ad ]);S [ad ]=v 
    print (f"{ad :<22}{v :>9.4f}{v /max (tesp ,1e-9 ):>13.1%}")
K =np .array (KAYMA )
print (f"\nPOZ KAFASI KAYMASI ({len (K )} candidate): medyan {np .median (K ):.2f}mm | "
f"%90 {np .percentile (K ,90 ):.2f}mm | max {K .max ():.2f}mm")
print ("  (tezin v_o'sundan ne up to uzaklastigimiz -- 'agzin inside merkezleme' savunmasi for)")
hedef =0.90 *tesp 
print (f"\n%90 gecis orani icin robot {hedef :.4f} gerekir; birlesik kahin {S ['IKISI']:.4f}")
print ("VERDICT:","%90 MUMKUN (kahin asiyor)"if S ["IKISI"]>=hedef else "%90 SOZLUKLERLE ULASILAMAZ")
json .dump ({k :float (v )for k ,v in S .items ()}|{"tespit":tesp ,
"gecis_ikisi":S ["IKISI"]/tesp ,"kayma_medyan":float (np .median (K )),
"kayma_p90":float (np .percentile (K ,90 ))},
io .open ("results/r1_birlesik_kahin.json","w"),indent =1 )
print ("receipt -> results/r1_birlesik_kahin.json")
