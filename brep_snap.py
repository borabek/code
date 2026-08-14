# -*- coding: utf-8 -*-
"""B1/B2: ML'in buldugu CP'yi B-rep'in TAM geometrisine yapistir (konum + axis).

WHY (tez):
  Tez CP'yi acikligin AGIZ MERKEZI (v_o) and INSERT YONU with tanimlar. Bizim turetmemiz mesh
  bilesenlerinin weight merkezinden geliyor; olculen dik error 0.1-2.3 mm. Robot hedefi <2mm.
  B-rep whereas silindirin TAM merkezini and TAM eksenini tasiyor -- tessellation'da atilan bilgi.

NE DEGISIR, NE DEGISMEZ:
  * TESPIT ML'de kalir (hangi opening CP'dir sorusuna geometri KARISMAZ -- three bagimsiz measurement
    geometrinin bunu yapamadigini showed).
  * Geometri SADECE already bulunmus a CP'nin KONUMUNU and YONUNU fixes.
  * Bu yuzden CP-F1 neredeyse does not change; hedeflenen sey ROBOT ISABETIDIR (mm).

KILL KRITERI (baslamadan yazildi): medyan dik error >= %20 azalmiyorsa aile OLU.
"""
import numpy as np 

MAX_SNAP_MM =3.0 # this mesafeden uzaktaki B-rep ozelligine yapistirma
R_MIN ,R_MAX =0.5 ,4.5 # tel girisi radius araligi


def _axis_and_len (lo ,hi ):
    ext =hi -lo 
    k =int (np .argmax (ext ))
    a =np .zeros (3 );a [k ]=1.0 
    return a ,float (ext [k ]),k 


def snap (surfaces ,point ,direction ,max_mm =MAX_SNAP_MM ):
    """CP'yi most yakin uygun B-rep silindirinin mouth merkezine + eksenine tasi.

    Doner: (yeni_nokta, yeni_yon, tasima_mm, yapisti_mi)
    Uygun candidate otherwise input AYNEN dondurulur -- sessizce bozma absent.
    """
    p =np .asarray (point ,float )
    d =np .asarray (direction ,float )
    n =float (np .linalg .norm (d ))
    d =d /n if n >1e-9 else np .array ([0.0 ,0.0 ,1.0 ])

    best =None 
    for t ,area ,com ,lo ,hi ,rad in surfaces :
        if t !="Cylinder"or not (R_MIN <=rad <=R_MAX ):
            continue 
        ax ,L ,k =_axis_and_len (lo ,hi )
        if L <1.0 :
            continue 
            # silindirin IKI agzi; CP'ye yakin olani hedef
        for sgn in (+1.0 ,-1.0 ):
            mouth =com +sgn *ax *(L /2.0 )
            dist =float (np .linalg .norm (mouth -p ))
            if dist >max_mm :
                continue 
                # axis isareti: CP'nin mevcut yonune yakin olani sec (direction ters cevrilmesin)
            axis =sgn *ax 
            if float (np .dot (axis ,d ))<0 :
                axis =-axis 
            if best is None or dist <best [0 ]:
                best =(dist ,mouth ,axis )
    if best is None :
        return p ,d ,0.0 ,False 
    return best [1 ],best [2 ],float (best [0 ]),True 


def exact_cylinders (step_path ):
    """TAM silindir parametreleri (OCP): axis, radius, mouth noktalari.

    bbox tabanli prediction YETERSIZ kaldi (measured: dik error 1.31 -> 1.58mm, KOTULESTI). Ekseni
    bbox'a hizali olmayan silindirde bbox real yuzeyden tasar and 'mouth' wrong yere dusuyordu.
    BRepAdaptor_Surface real axis/yaricapi, parametre araligi da real uzunlugu gives.
    """
    import numpy as _np 
    from OCP .STEPControl import STEPControl_Reader 
    from OCP .TopExp import TopExp_Explorer 
    from OCP .TopAbs import TopAbs_FACE 
    from OCP .TopoDS import TopoDS 
    from OCP .BRepAdaptor import BRepAdaptor_Surface 
    from OCP .GeomAbs import GeomAbs_Cylinder 
    rd =STEPControl_Reader ()
    if rd .ReadFile (step_path )!=1 :
        return []
    rd .TransferRoots ()
    shape =rd .OneShape ()
    out =[]
    exp =TopExp_Explorer (shape ,TopAbs_FACE )
    while exp .More ():
        f =TopoDS .Face_s (exp .Current ());exp .Next ()
        try :
            ad =BRepAdaptor_Surface (f )
            if ad .GetType ()!=GeomAbs_Cylinder :
                continue 
            cyl =ad .Cylinder ()
            ax =cyl .Axis ();loc =ax .Location ();dr =ax .Direction ()
            c =_np .array ([loc .X (),loc .Y (),loc .Z ()],float )
            a =_np .array ([dr .X (),dr .Y (),dr .Z ()],float )
            a /=(_np .linalg .norm (a )+1e-12 )
            r =float (cyl .Radius ())
            v0 ,v1 =float (ad .FirstVParameter ()),float (ad .LastVParameter ())
            out .append ({"center":c ,"axis":a ,"radius":r ,
            "mouth_a":c +a *v0 ,"mouth_b":c +a *v1 })
        except Exception :
            continue 
    return out 


def snap_exact (cyls ,point ,direction ,max_mm =MAX_SNAP_MM ):
    """TAM silindir parametreleriyle yapistirma."""
    p =np .asarray (point ,float );d =np .asarray (direction ,float )
    n =float (np .linalg .norm (d ));d =d /n if n >1e-9 else np .array ([0.0 ,0.0 ,1.0 ])
    best =None 
    for c in cyls :
        if not (R_MIN <=c ["radius"]<=R_MAX ):
            continue 
        for m in (c ["mouth_a"],c ["mouth_b"]):
            dist =float (np .linalg .norm (m -p ))
            if dist >max_mm :
                continue 
            axis =c ["axis"]if float (np .dot (c ["axis"],d ))>=0 else -c ["axis"]
            if best is None or dist <best [0 ]:
                best =(dist ,np .asarray (m ,float ),axis )
    if best is None :
        return p ,d ,0.0 ,False 
    return best [1 ],best [2 ],float (best [0 ]),True 


def snap_cps (step_path ,cps ,max_mm =MAX_SNAP_MM ):
    """cp_openings/robot_cp ciktisini yerinde duzelt. Doner: (yeni_liste, istatistik)."""
    import tel_g_brep as B 
    surf =B .read_brep (step_path )
    out ,moved ,hit =[],[],0 
    for c in cps :
        p ,d ,dist ,ok =snap (surf ,c ["point"],c ["direction"],max_mm )
        c2 =dict (c );c2 ["point"]=p ;c2 ["direction"]=d 
        c2 ["snapped"]=bool (ok );c2 ["snap_mm"]=dist 
        out .append (c2 )
        if ok :
            hit +=1 ;moved .append (dist )
    return out ,{"n":len (cps ),"snapped":hit ,
    "median_move_mm":float (np .median (moved ))if moved else 0.0 }
