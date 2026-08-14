# -*- coding: utf-8 -*-
"""B-rep oto-etiketinin KOR OLDUGU parts -- insan etiketi buraya gitmeli.

Oto-label GT'lerin only %35.2'siyle eslesti. Eslesmeyen %64.8, silindir
as modellenmemis girisler (kare mouth, yay-kelepce). Havuz recall'unu dibe
ceken markalar da onlar. Insan emegini otomatigin ZATEN yaptigi yere not,
YAPAMADIGI yere koymak is required.

Cikti: part basina GT eslesme orani; most dusukler onceliklidir.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
sys .path .insert (0 ,".")
os .environ .setdefault ("BA_ALLOW_SEEN","1")
import p2_brep_agiz_etiket as P2 
import d6_record ,canonical_d7 as K 

isler =[("corpus","results/_brepegit_silindirler.pkl",
"results/_brepegit_acikliklar.pkl",
K .yukle ([str (p )for p in json .load (
open ("results/brep_egitim_kumesi.json"))["pidler"]])),
("d6","results/_d6_silindirler.pkl","results/_d6_acikliklar.pkl",
d6_record .yukle (set (d6_record .exam ()["pidler"])))]
sat =[]
for ad ,cylf ,acf ,kay in isler :
    cy =pickle .load (open (cylf ,"rb"));ac =pickle .load (open (acf ,"rb"))
    for pid ,r in kay .items ():
        G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
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
sat .sort (key =lambda x :(x [3 ],-x [2 ]))
mk =collections .defaultdict (list )
for p ,m ,n ,o in sat :
    mk [m ].append (o )
print (f"{'brand':<8} {'part':>6} {'ort eslesme':>12} {'kor (<0.2)':>11}")
for m in sorted (mk ,key =lambda k :np .mean (mk [k ])):
    v =np .asarray (mk [m ])
    print (f"{m :<8} {len (v ):>6} {v .mean ():>12.3f} {int ((v <0.2 ).sum ()):>11}")
kor =[s for s in sat if s [3 ]<0.2 and s [2 ]>=4 ]
print (f"\nKOR part (eslesme<0.2, GT>=4): {len (kor )}")
print (collections .Counter (m for _ ,m ,_ ,_ in kor ).most_common (8 ))
with open ("results/p2_kor_nokta_pidler.txt","w")as f :
    for p ,m ,n ,o in kor [:60 ]:
        f .write (p +"\n")
json .dump ({"n_kor":len (kor ),"ilk60":[list (x )for x in kor [:60 ]],
"not":"B-rep oto-etiketinin eslestiremedigi parts -- insan etiketi "
"ONCELIGI. Otomatigin already yaptigi yere emek harcanmaz."},
open ("results/p2_kor_nokta.json","w"),indent =1 )
print ("-> results/p2_kor_nokta_pidler.txt (first 60)")
