# -*- coding: utf-8 -*-
"""BAGLAYICI KISIT HAVUZ: B-rep fiziksel onerileri havuza EKLEMEK recall'u acar mi?

Olculdu (`results/havuz_recall_d7.json`): D7 brand-disi pool recall G7'de
**0.6654**. Donusum ~%48 oldugu for robot tavani ~0.32 -- i.e. MEVCUT HAVUZLA
0.50 IMKANSIZ. Gate/selector/count uzerine kurulan each arm this tavanin under.

Bu probe single soruyu sorar: segmentasyon havuzuna B-rep'ten gelen FIZIKSEL mouth
onerilerini (silindir agizlari + duzlemsel ic halkalar) EKLERSEK recall nereye
cikar? Kaydin `mouth_a`/`mouth_b`/opening merkezleri ZATEN diskte.

DURUSTLUK: this a TEZ SONUCU DEGIL. Tezin `v_o` turetmesi and 5 sinif DEGISMIYOR;
B-rep onerileri EK a candidate KAYNAGI. Kullanicinin own plani bunu "tez-omurgali
geometrik post-process genisletmesi" as ayirmisti and ayri raporlanir.

Ayrica precision bedeli ACIKCA sayilir: recall bedava not, candidate count artar.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
from sina_cluster import match_hungarian 

d7 =set (map (str ,json .load (open ("results/d7_sinav_kumesi.json"))["pidler"]))
R =[r for r in pickle .load (open ("results/_der_yeni_G7BIRLESIK.pkl","rb"))
if str (r ["pid"])in d7 ]
cy =pickle .load (open ("results/_d7_silindirler.pkl","rb"))
ac =pickle .load (open ("results/_d7_acikliklar.pkl","rb"))


def brep_oneriler (pid ,r_max =None ):
    """Silindir agizlari (two three) + opening merkezleri. Yon = axis/normal."""
    P ,D =[],[]
    for c in cy .get (pid )or []:
        if r_max is not None and float (c .get ("radius",0 ))>r_max :
            continue 
        ax =np .asarray (c ["axis"],float )
        ax =ax /(np .linalg .norm (ax )+1e-12 )
        for u ,s in (("mouth_a",1.0 ),("mouth_b",-1.0 )):
            if c .get (u )is not None :
                P .append (np .asarray (c [u ],float ));D .append (s *ax )
    for o in ac .get (pid )or []:
        if isinstance (o ,dict )and o .get ("center")is not None :
            n =np .asarray (o .get ("normal",[0 ,0 ,1 ]),float )
            n =n /(np .linalg .norm (n )+1e-12 )
            P .append (np .asarray (o ["center"],float ));D .append (n )
    return (np .asarray (P ,float ).reshape (-1 ,3 ),
    np .asarray (D ,float ).reshape (-1 ,3 ))


KOLLAR ={
"SEG (kanonik)":lambda pid ,P ,D :(P ,D ),
"SEG + B-rep (tum)":None ,
"SEG + B-rep (r<=3mm)":None ,
"B-rep TEK BASINA":None ,
}
sonuc ={}
for ad in KOLLAR :
    TP =FN =nA =0 ;per =collections .defaultdict (lambda :[0 ,0 ])
    for r in R :
        pid =str (r ["pid"]);G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
            continue 
        P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
        if ad =="SEG + B-rep (tum)":
            bP ,bD =brep_oneriler (pid )
            P =np .vstack ([P ,bP ])if len (bP )else P 
            D =np .vstack ([D ,bD ])if len (bD )else D 
        elif ad =="SEG + B-rep (r<=3mm)":
            bP ,bD =brep_oneriler (pid ,r_max =3.0 )
            P =np .vstack ([P ,bP ])if len (bP )else P 
            D =np .vstack ([D ,bD ])if len (bD )else D 
        elif ad =="B-rep TEK BASINA":
            P ,D =brep_oneriler (pid )
        tp ,fp ,fn =match_hungarian (P ,D ,G ,np .asarray (r ["Gd"],float ),r ["diag"],
        max (3.0 ,0.06 *r ["diag"]),180.0 ,True )[:3 ]
        TP +=tp ;FN +=fn ;nA +=len (P )
        a =per [r ["mfg"]];a [0 ]+=tp ;a [1 ]+=fn 
    rc =TP /max (TP +FN ,1 )
    sonuc [ad ]={"recall":rc ,"aday_per_parca":nA /max (len (R ),1 ),
    "brand":{m :a [0 ]/max (a [0 ]+a [1 ],1 )for m ,a in per .items ()}}
    print (f"{ad :<22} recall {rc :.4f} | candidate/part {sonuc [ad ]['aday_per_parca']:>6.1f}",
    flush =True )

t =sonuc ["SEG (kanonik)"]
print (f"\n{'brand':<8} {'SEG':>8} {'+B-rep':>8} {'fark':>8}")
u =sonuc ["SEG + B-rep (tum)"]
for m in sorted (t ["brand"],key =lambda k :t ["brand"][k ]):
    print (f"  {m :<7} {t ['brand'][m ]:>7.4f} {u ['brand'].get (m ,0 ):>8.4f} "
    f"{u ['brand'].get (m ,0 )-t ['brand'][m ]:>+8.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":sonuc ,
"not":"HAVUZ RECALL. B-rep onerileri TEZ TURETMESI DEGIL, ek candidate "
"kaynagi -- ayri raporlanir. D7 brand-disi."},
open ("results/brep_havuz_birlesim.json","w"),indent =1 )
