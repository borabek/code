# -*- coding: utf-8 -*-
"""B-rep onerilerini KIRP: recall'u tutup candidate sayisini dusuren filtre present mi?

Ham birlesim recall'u 0.6654 -> 0.8465 acti but candidate/part 13.5 -> 379.9 (28x).
O yogunlukta gate'in kesinligi cokerdi. Burada UC ucuz fiziksel filtre taranir:
  1. YARICAP BANDI  -- tel girisi agzinin yaricapi sinirli a aralikta
  2. DEDUPE         -- same mouth bircok yuzeyden birden onerilebiliyor
  3. KAPALILIK      -- silindirin uzunlugu/radius orani (hole mi, edge mi)
Once GT with ESLESEN onerilerin fiziksel dagilimi olculur, filtreler ORADAN secilir
(kor tarama not). Sonra recall-vs-candidate egrisi cikarilir and DIZ noktasi bulunur.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
from sina_cluster import match_hungarian 

d7 =set (map (str ,json .load (open ("results/d7_exam_set.json"))["pidler"]))
R =[r for r in pickle .load (open ("results/_der_yeni_G7BIRLESIK.pkl","rb"))
if str (r ["pid"])in d7 and len (r .get ("G",[]))]
cy =pickle .load (open ("results/_d7_silindirler.pkl","rb"))


def oneriler (pid ):
    """Silindir agizlari. Doner: P, D, radius, uzunluk/radius orani."""
    P ,D ,rad ,ora =[],[],[],[]
    for c in cy .get (pid )or []:
        ax =np .asarray (c ["axis"],float );ax /=(np .linalg .norm (ax )+1e-12 )
        ma ,mb =c .get ("mouth_a"),c .get ("mouth_b")
        rr =float (c .get ("radius",0.0 ))
        L =float (np .linalg .norm (np .asarray (ma ,float )-np .asarray (mb ,float )))if (ma is not None and mb is not None )else 0.0 
        for u ,s in ((ma ,1.0 ),(mb ,-1.0 )):
            if u is not None :
                P .append (np .asarray (u ,float ));D .append (s *ax )
                rad .append (rr );ora .append (L /max (rr ,1e-6 ))
    return (np .asarray (P ,float ).reshape (-1 ,3 ),np .asarray (D ,float ).reshape (-1 ,3 ),
    np .asarray (rad ,float ),np .asarray (ora ,float ))


    # --- 1) GT with matched onerilerin FIZIKSEL dagilimi -------------------------
es_r ,es_o ,tum_r ,tum_o =[],[],[],[]
for r in R :
    P ,D ,rad ,ora =oneriler (str (r ["pid"]))
    if not len (P ):
        continue 
    G =np .asarray (r ["G"],float );tol =max (3.0 ,0.06 *r ["diag"])
    d =np .linalg .norm (P [:,None ]-G [None ],axis =-1 ).min (1 )
    es =d <=tol 
    es_r +=rad [es ].tolist ();es_o +=ora [es ].tolist ()
    tum_r +=rad .tolist ();tum_o +=ora .tolist ()
q =lambda a ,p :float (np .percentile (a ,p ))# noqa: E731
print (f"GT'ye YAKIN oneriler n={len (es_r )} | TUM oneriler n={len (tum_r )}")
print (f"  yaricap  matched  p2 {q (es_r ,2 ):.2f} p50 {q (es_r ,50 ):.2f} p98 {q (es_r ,98 ):.2f} mm")
print (f"           TUMU     p2 {q (tum_r ,2 ):.2f} p50 {q (tum_r ,50 ):.2f} p98 {q (tum_r ,98 ):.2f} mm")
print (f"  L/r      matched  p2 {q (es_o ,2 ):.2f} p50 {q (es_o ,50 ):.2f} p98 {q (es_o ,98 ):.2f}")
print (f"           TUMU     p2 {q (tum_o ,2 ):.2f} p50 {q (tum_o ,50 ):.2f} p98 {q (tum_o ,98 ):.2f}")

R_LO ,R_HI =q (es_r ,2 ),q (es_r ,98 )
O_LO =q (es_o ,2 )
print (f"\nFILTRE (eslesenlerin p2-p98'inden): yaricap [{R_LO :.2f}, {R_HI :.2f}] mm, L/r >= {O_LO :.2f}\n")


def dedupe (P ,D ,mm ):
    if mm <=0 or len (P )<2 :
        return np .ones (len (P ),bool )
    tut =np .ones (len (P ),bool )
    for i in range (len (P )):
        if not tut [i ]:
            continue 
        for j in range (i +1 ,len (P )):
            if tut [j ]and np .linalg .norm (P [i ]-P [j ])<mm :
                tut [j ]=False 
    return tut 


    # --- 2) recall-vs-candidate egrisi ----------------------------------------------
KOLLAR =[("filtre YOK",False ,False ,0.0 ),
("yaricap",True ,False ,0.0 ),
("yaricap+L/r",True ,True ,0.0 ),
("yaricap+L/r+dedupe1",True ,True ,1.0 ),
("yaricap+L/r+dedupe2",True ,True ,2.0 ),
("yaricap+L/r+dedupe3",True ,True ,3.0 )]
print (f"{'arm':<22} {'recall':>8} {'candidate/part':>11} {'CWT':>8} {'KLM':>8}")
cik ={}
for ad ,fr ,fo ,dd in KOLLAR :
    TP =FN =nA =0 ;per =collections .defaultdict (lambda :[0 ,0 ])
    for r in R :
        pid =str (r ["pid"])
        P0 =np .asarray (r ["P"],float );D0 =np .asarray (r ["Pd"],float )
        P ,D ,rad ,ora =oneriler (pid )
        if len (P ):
            k =np .ones (len (P ),bool )
            if fr :
                k &=(rad >=R_LO )&(rad <=R_HI )
            if fo :
                k &=ora >=O_LO 
            P ,D =P [k ],D [k ]
            if len (P )and dd >0 :
                m =dedupe (P ,D ,dd );P ,D =P [m ],D [m ]
        P =np .vstack ([P0 ,P ])if len (P )else P0 
        D =np .vstack ([D0 ,D ])if len (D )else D0 
        tp ,fp ,fn =match_hungarian (P ,D ,np .asarray (r ["G"],float ),
        np .asarray (r ["Gd"],float ),r ["diag"],
        max (3.0 ,0.06 *r ["diag"]),180.0 ,True )[:3 ]
        TP +=tp ;FN +=fn ;nA +=len (P )
        a =per [r ["mfg"]];a [0 ]+=tp ;a [1 ]+=fn 
    pm ={m :a [0 ]/max (a [0 ]+a [1 ],1 )for m ,a in per .items ()}
    cik [ad ]={"recall":TP /max (TP +FN ,1 ),"aday_per_parca":nA /max (len (R ),1 ),
    "brand":pm }
    c =cik [ad ]
    print (f"{ad :<22} {c ['recall']:>8.4f} {c ['aday_per_parca']:>11.1f} "
    f"{pm .get ('CWT',0 ):>8.4f} {pm .get ('KLM',0 ):>8.4f}",flush =True )
json .dump ({"damga":receipt_hash .damga (),"filtre":{"r_lo":R_LO ,"r_hi":R_HI ,
"lr_lo":O_LO },"kollar":cik ,
"not":"HAVUZ RECALL, D7 brand-disi. Filtre esikleri GT'ye matched "
"onerilerin p2-p98'inden, kor tarama DEGIL."},
open ("results/brep_filtre_sweep.json","w"),indent =1 )
