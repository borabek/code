# -*- coding: utf-8 -*-
"""FAZ 2 / P4 -- FAMILY TRANSFER'I DURUST OLC (kullanici kurallari birebir).
KURALLAR:
 * hedefin GT CP'leri HIZALAMADA KULLANILMAZ -- hizalama SADECE CAD mesh geometrisiyle
 * template CP'leri only TRAIN-fold parcalarindan alinir (part-out CV; kardesler different fold'a duser)
 * singleton aileler agregada SIFIR katkiyla dahil edilir
 * known-family / unseen-family AYRI raporlanir
 * EK DURUSTLUK: geometri-duplike template'ler AYRI isaretlenir (orada transfer trivial -> siskinlik)
GO: agirlikli ALL kazanci >= +0.010, degilse family-transfer KAPANIR."""
import os ,sys ,json 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh 
from big_arbiter import eligible 

lock =json .load (open ("results/split_lock.json"))
META =lock ["parts"];LOCKED =set (lock ["locked_parts"])
os .environ ["BA_ALLOW_SEEN"]="1"
paths ={p :(jf ,stp )for m ,p ,jf ,stp in eligible ()}
work =[p for p in META if p not in LOCKED and p in paths ]
fam ={}
for p in work :fam .setdefault (META [p ]["family"],[]).append (p )
multi ={f :v for f ,v in fam .items ()if len (v )>1 }
print (f"WORK {len (work )} part | {len (fam )} aile | cok-parcali {len (multi )} aile / "
f"{sum (len (v )for v in multi .values ())} part",flush =True )

CACHE ={}


def load (pid ):
    if pid in CACHE :return CACHE [pid ]
    jf ,stp =paths [pid ]
    j =json .load (open (jf ,encoding ="utf-8-sig"))
    Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
    G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
    Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j ["ConnectionPoints"]],float )
    Vr ,Fr =step_to_mesh (stp )
    R ,t ,res =align_frames (Vr ,Vj )# mesh -> json
    Gm =(G -t )@R # GT'yi MESH frame'ine tasi
    Gdm =(Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 ))@R 
    CACHE [pid ]=(Vr ,Gm ,Gdm ,float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
    return CACHE [pid ]


def match (P ,G ,Gd ,tol ,axis_tol =40.0 ):
    if not len (P )or not len (G ):return 0 ,len (P ),len (G )
    diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .where (np .abs (al )<=axis_tol ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
    up ,ug ,tp =set (),set (),0 
    for d ,a ,b in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
        if d >tol :break 
        if a in up or b in ug :continue 
        up .add (a );ug .add (b );tp +=1 
    return tp ,len (P )-tp ,len (G )-tp 


rows =[]
for f ,members in multi .items ():
    for tgt in members :
    # part-out CV benzetimi: template = AYNI ailenin BASKA a parcasi (train-fold'da varsayilir)
        tpls =[p for p in members if p !=tgt ]
        try :
            Vt ,Gt_m ,Gtd_m ,_ =load (tgt )
        except Exception :
            continue 
        best =None 
        for tp_id in tpls [:3 ]:
            try :
                Vs ,Gs_m ,Gsd_m ,_ =load (tp_id )
                # HIZALAMA: SADECE CAD mesh'leri (hedefin GT'si KULLANILMAZ)
                Rx ,tx ,res =align_frames (Vs ,Vt )
                Pp =Gs_m @Rx .T +tx # template CP'leri -> hedef mesh frame
                Dd =Gsd_m @Rx .T 
                if best is None or res <best [0 ]:best =(res ,Pp ,Dd ,tp_id )
            except Exception :
                continue 
        if best is None :continue 
        res ,Pp ,Dd ,tp_id =best 
        diag =float (np .linalg .norm (Vt .max (0 )-Vt .min (0 )))
        tol =max (3.0 ,0.06 *diag )
        tp ,fp ,fn =match (Pp ,Gt_m ,Gtd_m ,tol )
        rows .append ({"pid":tgt ,"tpl":tp_id ,"res":res ,"tp":tp ,"fp":fp ,"fn":fn ,
        "gt":len (Gt_m ),"same_geom":META [tgt ]["geom"]==META [tp_id ]["geom"]})

print (f"\ntransfer denenen: {len (rows )} part")


def agg (rs ):
    T =sum (r ["tp"]for r in rs );F =sum (r ["fp"]for r in rs );N =sum (r ["fn"]for r in rs )
    p =T /max (T +F ,1 );rr =T /max (T +N ,1 )
    return p ,rr ,2 *p *rr /max (p +rr ,1e-9 ),sum (r ["gt"]for r in rs )


for nm ,rs in (("TUM transfer edilebilir",rows ),
("  |- geometri-DUPLIKE template (trivial, siskinlik)",[r for r in rows if r ["same_geom"]]),
("  |- GERCEK farkli-geometri kardes (durust)",[r for r in rows if not r ["same_geom"]])):
    if not rs :print (f"{nm :52s} (yok)");continue 
    p ,rr ,f1 ,gt =agg (rs )
    print (f"{nm :52s} n={len (rs ):3d} GT={gt :4d} | P {p :.3f} R {rr :.3f} F1 {f1 :.3f} | hizalama-artik ort {np .mean ([r ['res']for r in rs ]):.2f}mm")

CUR =0.756 # mevcut ML family-out ALL (WORK)
gt_all =sum (META [p ]["n_cps"]for p in work )
for nm ,rs in (("TUM",rows ),("SADECE gercek farkli-geometri",[r for r in rows if not r ["same_geom"]])):
    if not rs :continue 
    p ,rr ,f1 ,gt =agg (rs )
    w =gt /max (gt_all ,1 )
    gain =w *(f1 -CUR )
    print (f"\nAGIRLIKLI ALL kazanci ({nm }): kapsam {100 *w :.1f}% x (F1 {f1 :.3f} - mevcut {CUR :.3f}) = {gain :+.4f}"
    +("  -> GO (>=+0.010)"if gain >=0.010 else "  -> KALIR (<+0.010)"))
print ("\nNOT: unseen-family (singleton) parts transfer ALAMAZ -> agregada SIFIR katki (kural geregi dahil).")
print ("NOT: family-out CV'de kardesler AYNI fold'da olur -> transfer orada TANIMSIZ; bu bir 'known-family' yetenegidir.")
json .dump (rows ,open ("results/p4_transfer.json","w"),indent =1 )
