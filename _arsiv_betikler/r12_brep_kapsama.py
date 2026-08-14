# -*- coding: utf-8 -*-
"""R12: B-REP EKSEN KAPSAMASI -- gorulmemis ureticide neden %24?

FINDING (R11): B-rep ekseni bulunan candidate orani tanidikta %47, gorulmemiste %24. B-rep
bulunmayinca `channel_axis + _snap_axis` yoluna dusuluyor -- and that path EGIMI GOREMIYOR
(kod yorumu: ray-probe cozunurlugu deligin koni acisindan ~11 derece keskin olamaz).
Uretici yonlerinin %19.1'i 10 dereceden egik oldugu for this, sistematik dik hataya cevriliyor.

KAPILAR (cp_openings.py): CP_BREP_MAX_OFF=5.0mm, max_turn_deg=60, r_range=(0.5, BREP_R_MAX)
Soru: kapsama kapilardan mi low, otherwise opening gercekten silindir DEGIL mi?

OLCUM: each exam parcasinda STEP'teki silindirleri oku, candidate noktasina most yakin silindirin
mesafesini/angle sapmasini cikar. Kapiya TAKILAN mi, HIC SILINDIR YOK mu?
"""
import sys ,os ,json ,io ,pickle ,collections 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,'.')
import numpy as np 
import brep_axes as BX 
from big_arbiter import eligible 

sv =json .load (io .open ('results/d5_4_sinav_kumesi.json',encoding ='utf-8'))
G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
yol ={p :s for m ,p ,jf ,s in eligible ()}
D =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ][:80 ]
st =collections .Counter ();off =[];turn =[]
for r in D :
    P =np .asarray (r ['P'],float );Pd =np .asarray (r ['Pd'],float )
    if not len (P ):continue 
    try :cyl =BX .cylinders (yol [r ['pid']])
    except Exception :cyl =None 
    if cyl is None or not len (cyl [0 ]):
        st ['parca_SILINDIRSIZ']+=len (P );continue 
    C =np .asarray (cyl [0 ],float )# centre
    A =np .asarray (cyl [1 ],float )# axis
    for i in range (len (P )):
        p =P [i ];d =Pd [i ]/(np .linalg .norm (Pd [i ])+1e-9 )
        rel =p -C 
        t =(rel *A ).sum (1 )
        dik =np .linalg .norm (rel -t [:,None ]*A ,axis =1 )# eksene dik distance
        aci =np .degrees (np .arccos (np .clip (np .abs (A @d ),0 ,1 )))
        j =int (np .argmin (dik ))
        off .append (float (dik [j ]));turn .append (float (aci [j ]))
        if dik [j ]<=5.0 and aci [j ]<=60 :st ['GECER']+=1 
        elif dik [j ]>5.0 and aci [j ]<=60 :st ['MESAFE_KAPISI']+=1 
        elif dik [j ]<=5.0 :st ['ACI_KAPISI']+=1 
        else :st ['IKISI_DE']+=1 
o =np .array (off );tn =np .array (turn )
tot =sum (st .values ())
print (f"candidate {tot } | {dict (st )}")
print (f"  GECER %{100 *st ['GECER']/max (tot ,1 ):.0f}")
print (f"\nEN YAKIN SILINDIRE dik mesafe (mm): ortanca {np .median (o ):.2f} | %75 {np .percentile (o ,75 ):.2f}")
print (f"  <=5mm %{100 *(o <=5 ).mean ():.0f} | <=8mm %{100 *(o <=8 ).mean ():.0f} | <=12mm %{100 *(o <=12 ).mean ():.0f}")
print (f"ACI sapmasi (derece): ortanca {np .median (tn ):.0f} | <=60 %{100 *(tn <=60 ).mean ():.0f} | <=80 %{100 *(tn <=80 ).mean ():.0f}")
print (f"\nKAPI GEVSETME SIMULASYONU (mesafe 5->8mm, aci 60->80):")
print (f"  mevcut  %{100 *((o <=5 )&(tn <=60 )).mean ():.0f}   ->  gevsek %{100 *((o <=8 )&(tn <=80 )).mean ():.0f}")
