# -*- coding: utf-8 -*-
"""GEO-G2: YARIK / PUSH-IN candidate uretimi -- silindir OLMAYAN tel girisleri.

RATIONALE (G1 otopsisi, measured): kacan 75 GT'nin **70'i (%93.3)** no silindirik B-rep yuzeyine
3mm'den yakin not. Yani tel girislerinin large cogunlugu SILINDIR DEGIL -- yay-kelepceli /
push-in terminallerde giris, KARSILIKLI IKI DUZLEM arasindaki a yariktir. Silindir arayan a
dedektor onlari never goremez; radius penceresini daraltmak recall'u that is why coketmisti.

YONTEM (ogrenilmis parametre YOK):
  1. B-rep duzlemlerinden KARSILIKLI CIFTLER bul: normalleri zit, aralari SLOT_MIN..SLOT_MAX mm,
     bbox'lari ortusuyor.
  2. Ciftin middle duzleminde, ortusme bolgesinin merkezini AGIZ adayi say.
  3. Fiziksel dogrulama (cp_geometry.ray_hits): agizdan disari isin empty must be, iceri correct channel
     at least MIN_DEPTH devam etmeli.

NOT: duzlem normali gmsh'ten dogrudan gelmiyor; bbox'i YASSI which is yuzeyin ince ekseni normaldir.
Bu, duz duzlemler for conclusive; egri yuzeyler already Cylinder/Cone/Torus as gelir.
"""
import numpy as np 

SLOT_MIN ,SLOT_MAX =1.0 ,6.0 # tel girisi yarik acikligi (mm)
MIN_DEPTH =2.0 # channel at least this up to devam etmeli
MIN_OVERLAP =2.0 # double, at least this up to ortusmeli (mm)
MERGE_MM =3.0 


def _plane_axis (lo ,hi ):
    """Yassi bbox'in EN INCE ekseni = duzlem normali. Doner: (normal_ekseni, kalinlik)."""
    ext =hi -lo 
    k =int (np .argmin (ext ))
    n =np .zeros (3 );n [k ]=1.0 
    return n ,float (ext [k ]),k 


def detect_slots (surfaces ,mesh ,slot_min =SLOT_MIN ,slot_max =SLOT_MAX ,
min_depth =MIN_DEPTH ,min_overlap =MIN_OVERLAP ):
    """B-rep duzlem ciftlerinden yarik agzi adaylari."""
    from cp_geometry import ray_hits 
    planes =[]
    for t ,area ,com ,lo ,hi ,rad in surfaces :
        if t !="Plane"or area <=0 :
            continue 
        n ,thick ,k =_plane_axis (lo ,hi )
        planes .append ((n ,k ,com ,lo ,hi ,area ))

    cands =[]
    for i in range (len (planes )):
        ni ,ki ,ci ,loi ,hii ,ai =planes [i ]
        for j in range (i +1 ,len (planes )):
            nj ,kj ,cj ,loj ,hij ,aj =planes [j ]
            if ki !=kj :# same eksene dik olmalilar (karsilikli)
                continue 
            gap =abs (float (ci [ki ]-cj [ki ]))
            if not (slot_min <=gap <=slot_max ):
                continue 
                # diger two eksende ORTUSME
            other =[a for a in range (3 )if a !=ki ]
            ov =[]
            for a in other :
                o =min (hii [a ],hij [a ])-max (loi [a ],loj [a ])
                ov .append (o )
            if min (ov )<min_overlap :
                continue 
                # yarik AGZI: ortusme bolgesinin merkezi, ciftin ORTA duzleminde
            mid =np .zeros (3 )
            mid [ki ]=(ci [ki ]+cj [ki ])/2.0 
            for a in other :
                mid [a ]=(max (loi [a ],loj [a ])+min (hii [a ],hij [a ]))/2.0 
                # yarik uzunlugu = ortusmenin UZUN ekseni -> mouth that eksenin UCUNDA
            la =other [int (np .argmax (ov ))]
            half =ov [int (np .argmax (ov ))]/2.0 
            for sgn in (+1.0 ,-1.0 ):
                mouth =mid .copy ();mouth [la ]=mid [la ]+sgn *half 
                out =np .zeros (3 );out [la ]=sgn 
                if len (ray_hits (mesh ,mouth +out *0.3 ,out ,8.0 )):
                    continue # mouth disari acilmali
                hin =ray_hits (mesh ,mouth +out *0.3 ,-out ,40.0 )
                depth =float (hin [0 ])if len (hin )else 40.0 
                if depth <min_depth :
                    continue 
                cands .append ({"point":mouth ,"direction":-out ,"radius":gap /2.0 ,
                "depth":depth ,"area":float (min (ai ,aj )),"kind":"slot"})
    return cands 


def detect (step_path ,V ,F ,**kw ):
    """Silindir adaylari (geo_brep_cp) + YARIK adaylari -> birlesik, tekillestirilmis."""
    import geo_brep_cp ,tel_g_brep as B 
    surf =B .read_brep (step_path )
    mesh =(np .asarray (V ,float ),np .asarray (F ,np .int64 ))
    cyl =geo_brep_cp .detect (step_path ,V ,F )
    for c in cyl :
        c .setdefault ("kind","cyl")
    slots =detect_slots (surf ,mesh ,**kw )
    out =[]
    for c in sorted (cyl +slots ,key =lambda x :-x ["depth"]):
        p =np .asarray (c ["point"],float )
        if any (np .linalg .norm (p -np .asarray (k ["point"],float ))<MERGE_MM for k in out ):
            continue 
        out .append (c )
    return out 
