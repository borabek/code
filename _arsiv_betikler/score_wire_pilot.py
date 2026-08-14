# -*- coding: utf-8 -*-
"""WIRE-PILOT skorlayici: insan cevaplarini URETICI listesiyle kiyasla.
DECISION KAPISI:
  uyum >= %90 -> insan etiketi ureticinin tanimini yeniden uretiyor => 2838 STEP'i insan etiketiyle
                 acabiliriz (data acligina dogrudan ilac, scale meselesi)
  %75-90      -> kismi; only high-safe (unsure disi) cevaplar kullanilabilir
  <  %75      -> ureticinin listesinde geometride OLMAYAN katalog bilgisi present (R4 kesinlesir)
                 => insan etiketi GURULTU adds, path KAPATILIR
Kullanim: python score_wire_pilot.py results/wire_pilot/wire_pilot_answers.json"""
import sys ,json 
import numpy as np 

ans =json .load (open (sys .argv [1 ]if len (sys .argv )>1 else "results/wire_pilot/wire_pilot_answers.json"))
truth =json .load (open ("results/wire_pilot/truth.json"))

tp =fp =fn =tn =uns =0 
per_part ={}
for k ,v in ans .items ():
    pid ,i =k .rsplit ("#",1 )
    if pid not in truth :continue 
    i =int (i )
    if i >=len (truth [pid ]):continue 
    y =truth [pid ][i ]# 1 = manufacturer CP with eslesen candidate
    if v =="unsure":uns +=1 ;continue 
    h =1 if v =="wire"else 0 
    per_part .setdefault (pid ,[0 ,0 ])
    per_part [pid ][1 ]+=1 
    if h ==y :per_part [pid ][0 ]+=1 
    if h ==1 and y ==1 :tp +=1 
    elif h ==1 and y ==0 :fp +=1 
    elif h ==0 and y ==1 :fn +=1 
    else :tn +=1 

n =tp +fp +fn +tn 
if n ==0 :
    print ("cevap yok");sys .exit ()
acc =(tp +tn )/n 
prec =tp /max (tp +fp ,1 );rec =tp /max (tp +fn ,1 )
f1 =2 *prec *rec /max (prec +rec ,1e-9 )
print (f"=== WIRE PILOT: insan vs URETICI ({len (per_part )} part, {n } kesin cevap, {uns } 'emin degilim') ===")
print (f"  UYUM (accuracy)        : {acc :.3f}")
print (f"  insan 'TEL' dedi & manufacturer de CP diyor (TP): {tp }")
print (f"  insan 'TEL' dedi ama manufacturer CP DEMIYOR (FP): {fp }")
print (f"  insan 'ALET' dedi ama manufacturer CP DIYOR  (FN): {fn }")
print (f"  ikisi de 'degil'                        (TN): {tn }")
print (f"  insan-precision {prec :.3f} | insan-recall {rec :.3f} | insan-F1 {f1 :.3f}")
print (f"\n  KIYAS: mevcut MODEL ayni gorevde family-out F1 ~0.752")
print (f"         insan-F1 modelden YUKSEKse -> ogrenilecek sinyal VAR (ceiling insan seviyesi)")
print (f"         insan-F1 modele YAKINSA    -> model zaten insan seviyesinde")
print ("\n=== KARAR ===")
if acc >=0.90 :
    print (f"  GECTI (%{100 *acc :.0f}) -> insan etiketi ureticinin tanimini yeniden uretiyor.")
    print ("  => 2838 STEP'i insan etiketiyle acmak MESRU. Olcek plani yapilabilir.")
elif acc >=0.75 :
    print (f"  KISMI (%{100 *acc :.0f}) -> sadece 'unsure' disi yuksek-guvenli cevaplar kullanilabilir;")
    print ("  => sinirli olcekte dene, noise etkisini ayrica olc.")
else :
    print (f"  KALDI (%{100 *acc :.0f}) -> manufacturer listesi geometriden turetilemiyor (katalog bilgisi).")
    print ("  => insan etiketi GURULTU ekler. YOL KAPATILIR (R4 teshisi kesinlesir).")
json .dump ({"accuracy":acc ,"tp":tp ,"fp":fp ,"fn":fn ,"tn":tn ,"unsure":uns ,
"human_f1":f1 ,"human_precision":prec ,"human_recall":rec },
open ("results/wire_pilot/score.json","w"),indent =1 )
print ("-> results/wire_pilot/score.json")
