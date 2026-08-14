# -*- coding: utf-8 -*-
"""TEL-G: B-rep ozellikleri -- boru hattinin meshlerken ATTIGI bilgi.

WHY BU, WHY SIMDI:
  [[low-cp-information-gap]] olcumu conclusive: low-CP gate'inde GENELLESTIRME boslugu +0.003 but
  BILGI boslugu +0.151. Yani more very data and more guclu model TUKENDI; only YENI BILGI oynatir.
  Mesh'ten turetilen geometri de neredeyse tukendi (TEL-B channel profili: +0.003 F1).
  STEP'in B-rep'i whereas never kullanilmadi: tessellation during surface TIPI, TAM radius and feature
  yapisi atiliyor. Tel girisi standart olculu silindirik/prizmatik a ozelliktir -- full da atilan sey.

NOTE: dar a B-rep sondasi (huni/taper) more before measured and AUC 0.504 with OLDU. Bu yuzden
here TEK ipucu not, TAM a aile denenir.

CERCEVE TUZAGI (before dogrulandi, after kod yazildi):
  build_rich_feats candidate noktalarini npz'ye JSON (URETICI) cercevesinde writes:
      P_json = p_mesh @ R.T + t          (R, t = cad_eval.align_frames(Vr, Vj))
  gmsh B-rep whereas STEP cercevesindedir. `pos` dogrudan kullanilirsa candidates parcanin DISINDA kalir
  (measured: 3211485 and 3209612'de 0/3 and 0/4 candidate bbox inside, 25-28mm deviation) and tum feature ailesi
  SESSIZCE cop becomes. Geri donusum:  p_step = (P_json - t) @ R
"""
import numpy as np 

FEAT_NAMES =[
"en_yakin_silindir_mm",# most yakin silindirik yuzeye uzaklik
"silindir_yaricap",# that silindirin yaricapi (tel girisi ~1-4mm bekleniyor)
"silindir_alan",# that silindirin alani
"silindir_sayisi_6mm",# 6mm icindeki silindirik surface count
"duzlem_sayisi_6mm",# 6mm icindeki duzlem count (yarik = very duzlem, hole = silindir)
"koni_torus_6mm",# 6mm icindeki koni/torus count (pah/radyus)
"silindir_alan_toplam_6mm",# 6mm icindeki total silindirik alan
"duzlem_alan_toplam_6mm",# 6mm icindeki total duzlem alani
"silindir_orani_6mm",# silindir alani / total alan  (YUVARLAK mi YARIK mi)
"en_yakin_yuzey_tipi",# 0=duzlem 1=silindir 2=koni 3=torus 4=diger
]
_TYPE_ID ={"Plane":0 ,"Cylinder":1 ,"Cone":2 ,"Torus":3 }


def read_brep (step_path ):
    """STEP -> surface listesi [(type, alan, centre, bbox_min, bbox_max, radius)]. gmsh MESHLEMEZ."""
    import gmsh 
    gmsh .initialize ();gmsh .option .setNumber ("General.Terminal",0 )
    try :
        gmsh .open (step_path )
        out =[]
        for dim ,tag in gmsh .model .getEntities (2 ):
            t =gmsh .model .getType (dim ,tag )
            try :
                area =float (gmsh .model .occ .getMass (dim ,tag ))
                com =np .array (gmsh .model .occ .getCenterOfMass (dim ,tag ),float )
            except Exception :
                area ,com =0.0 ,np .zeros (3 )
            bb =gmsh .model .getBoundingBox (dim ,tag )
            lo =np .array (bb [:3 ],float );hi =np .array (bb [3 :],float )
            ext =np .sort (hi -lo )
            # silindirde most small two bbox boyutu capa esittir -> radius ~ min/2
            rad =float (ext [0 ]/2.0 )if t =="Cylinder"else 0.0 
            out .append ((t ,area ,com ,lo ,hi ,rad ))
        return out 
    finally :
        gmsh .finalize ()


def _dist_to_box (p ,lo ,hi ):
    return float (np .linalg .norm (np .maximum (np .maximum (lo -p ,p -hi ),0.0 )))


def feats_for_points (surfaces ,points ,R =6.0 ):
    """(n_nokta, 10) B-rep ozelligi. `surfaces` = read_brep ciktisi, `points` STEP cercevesinde."""
    P =np .asarray (points ,float )
    if not len (P ):
        return np .zeros ((0 ,len (FEAT_NAMES )))
    out =np .zeros ((len (P ),len (FEAT_NAMES )))
    for i ,p in enumerate (P ):
        best_cyl =(1e9 ,0.0 ,0.0 )# (uzaklik, radius, alan)
        best_any =(1e9 ,4 )# (uzaklik, type)
        n_cyl =n_pln =n_ct =0 
        a_cyl =a_pln =0.0 
        for t ,area ,com ,lo ,hi ,rad in surfaces :
            d =_dist_to_box (p ,lo ,hi )
            tid =_TYPE_ID .get (t ,4 )
            if d <best_any [0 ]:
                best_any =(d ,tid )
            if t =="Cylinder"and d <best_cyl [0 ]:
                best_cyl =(d ,rad ,area )
            if d <=R :
                if t =="Cylinder":n_cyl +=1 ;a_cyl +=area 
                elif t =="Plane":n_pln +=1 ;a_pln +=area 
                elif t in ("Cone","Torus"):n_ct +=1 
        tot =a_cyl +a_pln 
        out [i ]=[min (best_cyl [0 ],99.0 ),best_cyl [1 ],best_cyl [2 ],
        n_cyl ,n_pln ,n_ct ,a_cyl ,a_pln ,
        a_cyl /tot if tot >1e-9 else 0.0 ,best_any [1 ]]
    return out 
