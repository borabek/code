# -*- coding: utf-8 -*-
"""D20: MESH-ONLY KORPUS -- STEP olmadan segmentasyon training verisi.

`_ds1/DataSet/*.json` dosyalarinda Graphic3d (Points+Indices) VE ConnectionPoints
AYNI dosyada. STEP GEREKMEZ -- segmentasyon already mesh ten calisiyor.

WHY IMPORTANT: FN'lerin %57'si ADAY_YOK (temsil). Ve lateral error SEG KALITESIYLE
aciklanıyor ([[lateral-error-segmentasyon-kalitesiyle-aciklanir]]) -- i.e. this corpus
YALNIZ tespite not DONUSUME de candidate.

TEZE SADIK: ~6000 uniform izotropik remesh (thesis_remesh.remesh_uniform),
5 sinif, `v_o` does not change. Yalniz EGITIM VERISI eklenir.

SIZINTI: exam kumelerinin URETICILERI dislanir (D6+D7 markalari).
"""
import glob ,io ,json ,os ,sys ,time 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import thesis_remesh 
from g5_mouth_label import agiz_etiketle # AYNI boyama fonksiyonu
import connector3d 
CE =int (connector3d .CABLE_ENTRY )# urunun own sinif tanimi

KAYNAK ="_ds1/DataSet"
HEDEF ="_label_ds1_obj"
# D6 + D7 exam markalari -- korpusa GIRMEZ
SINAV_MFG ={"MOR","SE","UTL","S+S","SUPU","UPUN","NIT","ONV",
"A-B","C3","CCD","CEM","CWT","DEG","DIN","EFX","ELMEX","KLM","WEG","WIE"}


def _mfg_norm (b ):
    """Marka onekini NORMALIZE et.

    SIZINTI (2026-08-09, measured): file adi `A-B_N.1492-J16_...json`.
    Ham onek "A-B_N" and yasak listede "A-B" present -> filtre DELINDI and
    **D7'nin 84 A-B test parcasi** training ciktisina input.
    Cozum: `_`/`-` with ayrilan ekleri soyarak each ten-eki dene.
    """
    ham =b .split (".",1 )[0 ]
    candidates ={ham }
    x =ham 
    while "_"in x :
        x =x .rsplit ("_",1 )[0 ]
        candidates .add (x )
    return candidates 


def _sinav_pidleri ():
    """D6 + D7 part kimlikleri -- brand filtresinin YEDEGI (kemer + askı)."""
    import json as _j 
    out =set ()
    for f in ("results/d6_exam_set.json","results/d7_exam_set.json"):
        if os .path .exists (f ):
            out |={str (x )for x in _j .load (io .open (f ,encoding ="utf-8"))["pidler"]}
    return out 


def _xyz (o ):
    """{'X':..,'Y':..,'Z':..} -> [x,y,z]. Bu data kumesinde points SOZLUK."""
    return [float (o ["X"]),float (o ["Y"]),float (o ["Z"])]


def mesh_al (d ):
    g =d .get ("Graphic3d")or {}
    P =g .get ("Points");I =g .get ("Indices")
    if not P or not I or len (I )%3 :
        return None ,None 
    V =np .asarray ([_xyz (q )for q in P ],float )
    F =np .asarray (I ,int ).reshape (-1 ,3 )
    if F .max ()>=len (V ):
        return None ,None 
    return V ,F 


def cp_al (d ):
    """GT CP'leri. YON ANAHTARI `InsertDirection` -- varsayilana DUSMEYIZ.

    Ilk surumde "Direction"/"Normal"/"Axis" ariyordum and none of them yoktu; tum
    yonler [0,0,1] varsayilanina duserdi and SESSIZCE BOZUK label uretirdi.
    Yon otherwise that CP ATLANIR -- uydurma direction boyamaktansa missing boyamak yeg.
    """
    G ,Gd =[],[]
    for c in d .get ("ConnectionPoints")or []:
        p =c .get ("Point");v =c .get ("InsertDirection")
        if not isinstance (p ,dict )or not isinstance (v ,dict ):
            continue 
        G .append (_xyz (p ));Gd .append (_xyz (v ))
    return np .asarray (G ,float ),np .asarray (Gd ,float )


def main ():
    os .makedirs (HEDEF ,exist_ok =True )
    fs =sorted (glob .glob (f"{KAYNAK }/*.json"))
    present ={os .path .basename (os .path .normpath (x ))for x in glob .glob (f"{HEDEF }/*/")}
    global SINAV_PID 
    SINAV_PID =_sinav_pidleri ()
    print (f"exam pid yasagi: {len (SINAV_PID )} part",flush =True )
    say ={"yazildi":0 ,"sinav_markasi":0 ,"sinav_parcasi":0 ,"terminal_degil":0 ,
    "cp_yok":0 ,"mesh_yok":0 ,"remesh_hata":0 ,"boya_yok":0 ,"already":0 }
    t0 =time .time ()
    for i ,f in enumerate (fs ,1 ):
        b =os .path .basename (f )
        if _mfg_norm (b )&SINAV_MFG :
            say ["sinav_markasi"]+=1 ;continue 
        if "ElectricalTerminal"not in b :
            say ["terminal_degil"]+=1 ;continue 
        pid =b [:-5 ]
        # IKINCI KATMAN: part KIMLIGI exam kumesinde mi (brand filtresi delinirse)
        _kim =pid .split (".",1 )[1 ].split ("_Electrical")[0 ]if "."in pid else pid 
        if _kim in SINAV_PID :
            say ["sinav_parcasi"]+=1 ;continue 
        if pid in present :
            say ["already"]+=1 ;continue 
        try :
            d =json .load (io .open (f ,encoding ="utf-8"))
        except Exception :
            say ["mesh_yok"]+=1 ;continue 
        V ,F =mesh_al (d )
        if V is None or not len (V )or not len (F ):
            say ["mesh_yok"]+=1 ;continue 
        G ,Gd =cp_al (d )
        if not len (G ):
            say ["cp_yok"]+=1 ;continue 
        try :
            V2 ,F2 =thesis_remesh .remesh_uniform (V ,F ,target =6000 )
        except Exception :
            say ["remesh_hata"]+=1 ;continue 
        if V2 is None or len (V2 )<100 :
            say ["remesh_hata"]+=1 ;continue 
        try :
            import trimesh 
            m =trimesh .Trimesh (vertices =V2 ,faces =F2 ,process =False )
            L ,_bilgi =agiz_etiketle (V2 ,F2 ,m ,G ,Gd ,CE )
        except Exception :
            say ["boya_yok"]+=1 ;continue 
        if not (np .asarray (L )==CE ).any ():
            say ["boya_yok"]+=1 ;continue 
        dd =os .path .join (HEDEF ,pid );os .makedirs (dd ,exist_ok =True )
        with io .open (os .path .join (dd ,pid +".obj"),"w",encoding ="utf-8")as h :
            h .write ("".join (f"v {x :.6f} {y :.6f} {z :.6f}\n"for x ,y ,z in V2 ))
            h .write ("".join (f"f {a +1 } {b_ +1 } {c +1 }\n"for a ,b_ ,c in F2 ))
        with io .open (os .path .join (dd ,pid +".labels.txt"),"w",encoding ="utf-8")as h :
            h .write (" ".join (map (str ,np .asarray (L ).tolist ())))
        say ["yazildi"]+=1 
        if i %200 ==0 :
            print (f"  {i }/{len (fs )} {(time .time ()-t0 )/i :.2f}s/file | {say }",flush =True )
    print (f"\nBITTI {say }")
    json .dump (say ,open ("results/d20_mesh_corpus.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
