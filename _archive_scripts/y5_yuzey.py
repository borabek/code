# -*- coding: utf-8 -*-
"""Y5: YUZEY RESMI -- "bakinca anlasilan" gorsel.

DERINLIK HARITASI YANLIS ALETTI. Iki times denedim, ikisi de yanildi:
  1. Derinlik NOKTAYA according to olculuyordu; manufacturer CP'si kanalin ICINDE, robotunki AGIZDA
     tanimli -> same opening zit gorunuyordu.
  2. Duzeltince de okunmadi: manufacturer yonu (`InsertDirection`) GOVDENIN ICINE bakiyor,
     robotunki disari. Olcum boru hattimiz direction isaretine KOR (`abs()` kullanir), that yuzden
     this error bugune up to never patlamadi -- but cizimde patliyor.

Bu betik bunun instead of YUZEYIN KENDISINI cizer: noktanin cevresindeki agi clips, disari
bakan yonden isiklandirip render eder. Insan "orada hole present mi" sorusunu a isi
haritasindan not, a RESIMDEN yanitlar.

YON NORMALIZASYONU (each two source for same rule): direction, part merkezinden DISARI bakacak
sekilde isaretlenir -- (p - centre) . d < 0 whereas ters cevrilir. Boylece manufacturer and robot
noktalari AYNI konvansiyonda cizilir.
"""
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import matplotlib # noqa: E402
matplotlib .use ("Agg")
import matplotlib .pyplot as plt # noqa: E402
from mpl_toolkits .mplot3d .art3d import Poly3DCollection # noqa: E402

KIRP =13.0 # mm, yerel kirpma yaricapi


def disari (p ,d ,center_ ):
    """Yonu part merkezinden DISARI isaretle -- two kaynagi same konvansiyona getirir."""
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    return -d if float ((np .asarray (p ,float )-center_ )@d )<0 else d 


def cerceve (d ):
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    return u ,np .cross (d ,u )


def yuzey_ciz (ax ,V ,F ,p ,d ,center_ ,baslik ,renk ):
    """Noktanin cevresindeki yuzeyi, disaridan bakan a kamerayla ciz."""
    d =disari (p ,d ,center_ )
    u ,v =cerceve (d )
    c =V [F ].mean (1 )
    yak =np .linalg .norm (c -p ,axis =1 )<=KIRP 
    Fk =F [yak ]
    if not len (Fk ):
        ax .set_axis_off ();ax .set_title (baslik +"\n(surface absent)",fontsize =9 );return 
        # yerel koordinat: x=u, y=v, z=d yonunde (disari pozitif)
    R =np .stack ([u ,v ,d ])
    Y =(V -p )@R .T 
    T =Y [Fk ]
    nrm =np .cross (T [:,1 ]-T [:,0 ],T [:,2 ]-T [:,0 ])
    nl =np .linalg .norm (nrm ,axis =1 ,keepdims =True )+1e-12 
    nrm =nrm /nl 
    # ARKA YUZ ELEME: kameraya sirtini donen ucgenler cizilmez. Bunlar govdenin IC yuzeyi
    # and karartip resmi okunamaz hale getiriyordu (first denemede left kare simsiyahti).
    kam =np .array ([0.0 ,0.0 ,1.0 ])
    on =(nrm @kam )>0.02 
    if on .sum ()>=8 :
        T =T [on ];nrm =nrm [on ]
    isik =np .array ([0.35 ,0.25 ,0.90 ]);isik /=np .linalg .norm (isik )
    sh =np .clip (nrm @isik ,0 ,1 )*0.72 +0.24 
    face_ =np .stack ([sh *0.94 ,sh *0.95 ,sh ],axis =1 )# hafif soguk gri
    pc =Poly3DCollection (T ,facecolors =face_ ,edgecolors ="none",linewidths =0 )
    ax .add_collection3d (pc )
    L =KIRP *0.85 
    ax .set_xlim (-L ,L );ax .set_ylim (-L ,L );ax .set_zlim (-L ,L )
    ax .set_box_aspect ((1 ,1 ,1 ))
    ax .view_init (elev =58 ,azim =-62 )# DISARIDAN bakan kamera (z = disari direction)
    ax .set_axis_off ()
    # isaretci: noktadan disari kisa a ok
    ax .plot ([0 ,0 ],[0 ,0 ],[0 ,L *0.62 ],color =renk ,lw =2.6 ,zorder =10 )
    ax .scatter ([0 ],[0 ],[0 ],s =46 ,c =renk ,depthshade =False ,zorder =11 )
    ax .set_title (baslik ,fontsize =9.5 ,color =renk ,fontweight ="bold",pad =2 )


def main ():
    import measure_set 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 

    pid =sys .argv [1 ]if len (sys .argv )>1 else "3273094"
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,_ =measure_set .cluster ("results/_der_tam.pkl")
    r =[x for x in DER if x ["pid"]==pid ][0 ]
    Vr ,Fr =step_to_mesh (stp [pid ])
    V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
    V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
    center_ =V .mean (0 )
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    P =np .asarray (r ["P"],float );Pd =np .asarray (r ["Pd"],float )
    j =int (np .argsort (np .linalg .norm (G -G .mean (0 ),axis =1 ))[len (G )//2 ])

    fig =plt .figure (figsize =(9.2 ,4.4 ))
    ax0 =fig .add_subplot (121 ,projection ="3d")
    yuzey_ciz (ax0 ,V ,F ,G [j ],Gd [j ],center_ ,
    "URETICININ LISTELEDIGI GIRIS","#15703F")
    ax1 =fig .add_subplot (122 ,projection ="3d")
    yuzey_ciz (ax1 ,V ,F ,P [0 ],Pd [0 ],center_ ,"EXAMPLE ROBOT NOKTASI","#A8332A")
    fig .tight_layout ()
    fig .savefig ("results/_y5_deneme.png",dpi =95 ,bbox_inches ="tight")
    print ("-> results/_y5_deneme.png")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
