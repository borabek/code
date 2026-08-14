# -*- coding: utf-8 -*-
"""D20 UCUNCU SIZINTI KATMANI: GEOMETRI IKIZI dislama.

Marka onegi and part kimligi yetmez -- same parcanin BASKA a brand/kimlik
altindaki IKIZI de exam bilgisini carries ([[geometry-twin-leakage]]: parcalarin
%80'inin ikizi present). `results/_geo_yasak.json` exam/LOCKED parcalarinin
geometri anahtarlarini tutuyor (609 count).

Bu betik URETILMIS korpusu tarar, ikiz cikanlari KARANTINAYA carries.
"""
import io ,json ,os ,shutil ,sys 
import numpy as np 
sys .path .insert (0 ,".")
import geometri_anahtar 

KORPUS ="_label_ds1_obj"
KARANTINA ="_KARANTINA_d20_geo_ikiz"


def obj_oku (yol ):
    V ,F =[],[]
    for ln in io .open (yol ,encoding ="utf-8"):
        if ln .startswith ("v "):
            V .append ([float (x )for x in ln .split ()[1 :4 ]])
        elif ln .startswith ("f "):
            F .append ([int (t .split ("/")[0 ])-1 for t in ln .split ()[1 :4 ]])
    return np .asarray (V ,float ),np .asarray (F ,int )


def main ():
    yasak =set (json .load (io .open ("results/_geo_yasak.json",encoding ="utf-8")).values ())
    print (f"yasak geometri anahtari: {len (yasak )}")
    adlar =[d for d in os .listdir (KORPUS )if os .path .isdir (os .path .join (KORPUS ,d ))]
    print (f"corpus: {len (adlar )} part")
    os .makedirs (KARANTINA ,exist_ok =True )
    ikiz ,error =[],0 
    for i ,ad in enumerate (adlar ,1 ):
        o =os .path .join (KORPUS ,ad ,ad +".obj")
        if not os .path .exists (o ):
            error +=1 ;continue 
        try :
            V ,F =obj_oku (o )
            k =geometri_anahtar .key_ (V ,F )
        except Exception :
            error +=1 ;continue 
        if k in yasak :
            ikiz .append (ad )
            shutil .move (os .path .join (KORPUS ,ad ),os .path .join (KARANTINA ,ad ))
        if i %500 ==0 :
            print (f"  {i }/{len (adlar )} | ikiz {len (ikiz )}",flush =True )
    print (f"\nIKIZ BULUNAN: {len (ikiz )} part -> {KARANTINA }")
    print (f"KALAN TEMIZ KORPUS: {len (adlar )-len (ikiz )-error } | okunamayan {error }")
    json .dump ({"ikiz":ikiz ,"n_yasak_anahtar":len (yasak ),
    "kalan":len (adlar )-len (ikiz )-error },
    io .open ("results/d20_geo_ikiz.json","w",encoding ="utf-8"),indent =1 )


if __name__ =="__main__":
    main ()
