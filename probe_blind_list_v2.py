# -*- coding: utf-8 -*-
"""KOR NOKTA LISTESI v2 -- insan emegine according to siralanmis.

v1 KUSURLUYDU and kullanici yakaladi: list `(eslesme, -GT_sayisi)` with
siralanmisti, i.e. at most CP'li parts ONE geliyordu (first five part 60-64 CP'li
VTK very katli klemens -- biri bile saatler takes). Ayrica geometri anahtari
30/30 "different" dese de parts AYNI URUN AILESININ varyantlariydi
(VTK-5MD-16P / 5ME-16P / 5MD-15P): olcu different, BILGI same.

v2 kurallari:
  * GT count [2, 12] -- etiketlemesi HIZLI, ogrenme sinyali yine yeterli
  * AILE basina most extra 1 part (name kokunden: rakam bloklari silinir)
  * brand basina most extra `MARKA_TAVANI`
  * eslesme orani low which is before (otomatigin most kor oldugu yer)
"""
import collections ,json ,os ,pickle ,re ,sys 
import numpy as np 
sys .path .insert (0 ,".")
os .environ .setdefault ("BA_ALLOW_SEEN","1")
import p2_brep_mouth_label as P2 
import d6_record ,canonical_d7 as K 

GT_ALT ,GT_UST =2 ,12 
MARKA_TAVANI =6 
HEDEF =24 


def aile (pid ):
    """Urun ailesi kokunu cikar: rakam bloklari and ayraclar silinir."""
    return re .sub (r"[0-9]+","#",str (pid )).strip ("-_. ").upper ()


isler =[("corpus","results/_brepegit_silindirler.pkl",
"results/_brepegit_acikliklar.pkl",
K .yukle ([str (p )for p in json .load (
open ("results/brep_training_set.json"))["pidler"]])),
("d6","results/_d6_silindirler.pkl","results/_d6_acikliklar.pkl",
d6_record .yukle (set (d6_record .exam ()["pidler"])))]
sat =[]
for ad ,cylf ,acf ,kay in isler :
    cy =pickle .load (open (cylf ,"rb"));ac =pickle .load (open (acf ,"rb"))
    for pid ,r in kay .items ():
        G =np .asarray (r .get ("G",[]),float )
        if not (GT_ALT <=len (G )<=GT_UST ):
            continue 
        Gd =np .asarray (r ["Gd"],float )
        A =P2 .agizlar (cy .get (str (pid )),ac .get (str (pid )))
        if not A :
            sat .append ((str (pid ),r ["mfg"],len (G ),0.0 ));continue 
        M =np .asarray ([a [0 ]for a in A ],float )
        AX =np .asarray ([a [1 ]for a in A ],float )
        es =0 
        for j in range (len (G )):
            u =P2 ._birim (Gd [j ])
            if u is None :
                continue 
            w =G [j ][None ]-M 
            e =np .einsum ("ij,ij->i",w ,AX )
            y =np .linalg .norm (w -e [:,None ]*AX ,axis =1 )
            a2 =np .degrees (np .arccos (np .clip (np .abs (AX @u ),-1.0 ,1.0 )))
            if ((y <=P2 .ESLES_YANAL )&(np .abs (e )<=P2 .ESLES_EKSENEL )&
            (a2 <=P2 .ESLES_ACI )).any ():
                es +=1 
        sat .append ((str (pid ),r ["mfg"],len (G ),es /len (G )))

kor =[s for s in sat if s [3 ]<0.2 ]
kor .sort (key =lambda x :(x [3 ],x [2 ]))# before most kor, after most AZ CP'li
sec ,gorulen_aile ,brand =[],set (),collections .Counter ()
for p ,m ,n ,o in kor :
    a =aile (p )
    if a in gorulen_aile or brand [m ]>=MARKA_TAVANI :
        continue 
    sec .append ((p ,m ,n ,o ));gorulen_aile .add (a );brand [m ]+=1 
    if len (sec )>=HEDEF :
        break 
print (f"kor pool (GT {GT_ALT }-{GT_UST }, eslesme<0.2): {len (kor )}")
print (f"selected {len (sec )} | different aile {len (gorulen_aile )} | "
f"brand {dict (brand )}")
print (f"\n{'pid':<20} {'brand':<7} {'GT':>3} {'eslesme':>8}")
for p ,m ,n ,o in sec :
    print (f"{p :<20} {m :<7} {n :>3} {o :>8.2f}")
print (f"\ntoplam etiketlenecek CP: {sum (x [2 ]for x in sec )} "
f"(v1'de ilk 5 part already {60 +64 +60 +64 +60 })")
with open ("results/p2_blind_v2.txt","w")as f :
    for p ,_ ,_ ,_ in sec :
        f .write (p +"\n")
json .dump ({"selected":[list (x )for x in sec ],"kor_havuz":len (kor ),
"rule":{"gt":[GT_ALT ,GT_UST ],"aile_basina":1 ,
"marka_tavani":MARKA_TAVANI },
"not":"v1 en COK CP'liyi one koyuyordu (error). v2 hizli etiketlenen, "
"aile-cesitli, markaya yayilmis parts."},
open ("results/p2_blind_v2.json","w"),indent =1 )
print ("-> results/p2_blind_v2.txt")
