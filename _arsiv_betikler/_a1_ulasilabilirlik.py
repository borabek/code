# -*- coding: utf-8 -*-
"""A1: 188 ADAYSIZ GT -- kaci gercekten ULASILABILIR a aciklikta?

Kaybin most large single kalemi (GT'nin %14.2'si) and bugune up to WHY oldugu sorulmadi.
B2 showed ki turetme esiklerini gevsetmek kurtarmiyor -> network orada CE/CT uretmiyor.
Ama two BAMBASKA sebep may be:
   (a) AG HATASI      : disaridan open a channel present, network gormuyor -> KAZANILABILIR
   (b) YAPISAL        : CP malzemenin arkasinda, disaridan no opening absent
                        (manufacturer IC KONTAGI listelemis) -> no surface yontemi bulamaz

AYIRMA TESTI: CP'nin ekseni along DISARIDAN iceri isin at. Isin, malzemeye carpmadan
CP'ye 2mm'ye up to yaklasabiliyorsa OPEN KANAL vardir (a). Yaklasamıyorsa CP arkada (b).

Kendi Moller-Trumbore isinimimiz is used (trimesh.ray rtree olmadan patliyor -- see.
[[trimesh-rtree-silent-failure]]).
"""
import io ,json ,os ,sys ,time 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1");sys .path .insert (0 ,".")
import measure_set ,thesis_remesh 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from y2_fp_render import kesisim # Moller-Trumbore, single isin

stp ={p :s for m ,p ,jf ,s in eligible ()}
DER ,rap =measure_set .cluster ("results/_der_tam.pkl");measure_set .rapor_bas (rap )
ACIK =KAPALI =0 ;DERIN =[]
t0 =time .time ()
for k ,r in enumerate (DER ,1 ):
    if k %25 ==0 :print (f"  {k }/{len (DER )} acik={ACIK } kapali={KAPALI } {time .time ()-t0 :.0f}s",flush =True )
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    if not len (G ):continue 
    P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
    # hangi GT'ye candidate ULASMIYOR
    if len (P ):
        diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 );pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        ulasan =pe .min (0 )<=max (3.0 ,0.06 *float (r ["diag"]))
    else :ulasan =np .zeros (len (G ),bool )
    if ulasan .all ():continue 
    try :
        Vr ,Fr =step_to_mesh (stp [r ["pid"]]);V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
    except Exception :continue 
    for b in np .where (~ulasan )[0 ]:
        g =G [b ];d =Gd [b ]/(np .linalg .norm (Gd [b ])+1e-9 )
        # two yonu de dene: hangisi DISARIYI gosteriyorsa oradan isin at
        en =None 
        for s in (+1 ,-1 ):
            o =g +s *d *60.0 
            t =kesisim (V ,F ,o ,-s *d )# first carpma mesafesi
            if t is None :continue 
            kalan =60.0 -t # carpma noktasinin CP'ye uzakligi (isin along)
            if en is None or abs (kalan )<abs (en ):en =kalan 
        if en is None :KAPALI +=1 ;continue 
        DERIN .append (abs (en ))
        if abs (en )<=2.0 :ACIK +=1 
        else :KAPALI +=1 
D =np .array (DERIN )
top =ACIK +KAPALI 
print (f"\n{top } adaysiz GT incelendi")
print (f"  ACIK KANAL  (isin CP'ye <=2mm yaklasti): {ACIK :>4}  ({ACIK /max (top ,1 ):.1%})  -> AG HATASI, kazanilabilir")
print (f"  KAPALI      (CP malzemenin arkasinda)  : {KAPALI :>4}  ({KAPALI /max (top ,1 ):.1%})  -> YAPISAL, hicbir yuzey yontemi bulamaz")
if len (D ):
    print (f"\n  isin carpma noktasinin CP'ye uzakligi: medyan {np .median (D ):.1f}mm  %90 {np .percentile (D ,90 ):.1f}mm")
print ()
print ("OKUMA: KAPALI orani yuksekse candidate havuzu tavani (0.8579) SAHTE DUSUK --")
print ("       o CP'ler zaten ulasilamaz, recall'imiz gorundugunden IYI.")
json .dump ({"toplam":top ,"acik":ACIK ,"kapali":KAPALI ,
"acik_oran":ACIK /max (top ,1 ),"medyan_mm":float (np .median (D ))if len (D )else None },
io .open ("results/a1_ulasilabilirlik.json","w"),indent =1 )
print ("receipt -> results/a1_ulasilabilirlik.json")
