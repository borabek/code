# -*- coding: utf-8 -*-
"""GERI AL -- kontrol noktasina TAM donus. `kontrol_noktasi.py`'nin ikizi.

    python rollback.py                 # most new kontrol noktasina TAM don
    python rollback.py 2026-08-11      # belirli a noktaya don
    python rollback.py --kontrol       # HICBIR SEYI DEGISTIRME, only deviation raporla
    python rollback.py --kuru          # ne yapacagini yaz, yapma

TAM DONUS ucu birden yapar:
  1. KOD    : `git checkout <commit> -- .` + `git clean -fd`
              (`clean` only TAKIP EDILMEYEN and YOKSAYILMAYAN dosyalari deletes --
               data dizinleri .gitignore'da oldugu for ELLENMEZ.)
  2. MODEL  : `_KN_<label>/dosyalar/` icindekiler yerlerine geri kopyalanir.
  3. DOGRULA: kopya + damga SHA-256'lari and directory parmak izleri yeniden okunur.

GERI ALINAMAYAN cluster: manifestodaki `damga` (4.5 GB pkl, diske sigmiyor) and
`parmak_izi` dizinleri. Bunlar KOPYALANMADI, only DAMGALANDI. Degismislerse
this betik BAGIRIR and that dosyayi yeniden uretmek is required. Bu yuzden kampanya kurali:
MEVCUT DOSYANIN UZERINE YAZMA, new isim ver.
"""
import glob 
import hashlib 
import json 
import os 
import shutil 
import subprocess 
import sys 

KOK =os .path .dirname (os .path .abspath (__file__ ))


def sha (yol ,blok =1 <<20 ):
    h =hashlib .sha256 ()
    with open (yol ,"rb")as f :
        while True :
            b =f .read (blok )
            if not b :
                break 
            h .update (b )
    return h .hexdigest ()


def dizin_parmak_izi (d ):
    if not os .path .isdir (d ):
        return None 
    ad ,top =[],0 
    for f in sorted (os .listdir (d )):
        p =os .path .join (d ,f )
        if os .path .isfile (p ):
            n =os .path .getsize (p )
            top +=n 
            ad .append (f"{f }:{n }")
    return {"n":len (ad ),"bayt":top ,
    "ozet":hashlib .sha256 ("|".join (ad ).encode ()).hexdigest ()[:32 ]}


def en_yeni_kn ():
    d =sorted (glob .glob (os .path .join (KOK ,"_KN_*")))
    if not d :
        sys .exit ("KONTROL NOKTASI YOK. Once: python kontrol_noktasi.py")
    return d [-1 ]


def dogrula (man ,sessiz =False ):
    """Doner: (kopya_sapma, damga_sapma, parmak_sapma) listeleri."""
    ks ,ds ,ps =[],[],[]
    for rel ,m in man ["kopya"].items ():
        p =os .path .join (KOK ,rel .replace ("/",os .sep ))
        if not os .path .exists (p ):
            ks .append ((rel ,"YOK"))
        elif sha (p )!=m ["sha256"]:
            ks .append ((rel ,"DEGISMIS"))
    for rel ,m in man ["damga"].items ():
        p =os .path .join (KOK ,rel .replace ("/",os .sep ))
        if not os .path .exists (p ):
            ds .append ((rel ,"YOK"))
        elif os .path .getsize (p )!=m ["bayt"]or sha (p )!=m ["sha256"]:
            ds .append ((rel ,"DEGISMIS"))
    for d ,m in man ["parmak_izi"].items ():
        f =dizin_parmak_izi (os .path .join (KOK ,d ))
        if f is None :
            ps .append ((d ,"YOK"))
        elif f !=m :
            ps .append ((d ,f"n {m ['n']}->{f ['n']}"))
    if not sessiz :
        print (f"  kopya   : {len (man ['kopya'])} dosya, {len (ks )} deviation")
        for r ,n in ks [:10 ]:
            print (f"      ! {r }  {n }")
        print (f"  damga   : {len (man ['damga'])} dosya, {len (ds )} deviation"
        "   (GERI ALINAMAZ cluster)")
        for r ,n in ds [:10 ]:
            print (f"      ! {r }  {n }")
        print (f"  parmakiz: {len (man ['parmak_izi'])} dizin, {len (ps )} deviation")
        for r ,n in ps [:10 ]:
            print (f"      ! {r }  {n }")
    return ks ,ds ,ps 


def main ():
    arg =[a for a in sys .argv [1 :]if not a .startswith ("--")]
    yalniz_kontrol ="--kontrol"in sys .argv 
    kuru ="--kuru"in sys .argv 
    kn =(os .path .join (KOK ,f"_KN_{arg [0 ]}")if arg else en_yeni_kn ())
    man =json .load (open (os .path .join (kn ,"manifest.json")))
    print (f"KONTROL NOKTASI {os .path .basename (kn )}  ({man ['zaman']})")
    print (f"  git commit: {man ['git']}")

    if yalniz_kontrol :
        print ("\n-- SAPMA RAPORU (hicbir sey degistirilmedi) --")
        ks ,ds ,ps =dogrula (man )
        n =len (ks )+len (ds )+len (ps )
        print (f"\n{'TEMIZ -- durum kontrol noktasiyla AYNI'if not n else f'{n } SAPMA VAR'}")
        return 

        # --- 1) KOD ---------------------------------------------------------
    kirli =subprocess .check_output (
    ["git","status","--porcelain"],cwd =KOK ,text =True ).strip ()
    # porcelain: "XY PATH". Durum kodu two karakter, after a bosluk. Ama git
    # some kabuklarda basi clips; safe path ILK bosluktan sonrasini almak.
    yol =lambda l :l .split (" ",1 )[1 ].strip ().strip ('"')# noqa: E731
    yeni =[yol (l )for l in kirli .splitlines ()if l .lstrip ().startswith ("??")]
    degisen =[yol (l )for l in kirli .splitlines ()
    if not l .lstrip ().startswith ("??")]
    print (f"\n-- 1) KOD --  {len (degisen )} degismis, {len (yeni )} yeni dosya")
    for f in (degisen +yeni )[:20 ]:
        print (f"      {f }")
    if len (degisen )+len (yeni )>20 :
        print (f"      ... +{len (degisen )+len (yeni )-20 }")
    silinecek =subprocess .check_output (
    ["git","clean","-nd"],cwd =KOK ,text =True ).strip ()
    if silinecek :
        print ("   silinecek TAKIPSIZ dosyalar (veri dizinleri .gitignore'da, "
        "ELLENMEZ):")
        for l in silinecek .splitlines ()[:15 ]:
            print (f"      {l }")
    if not kuru :
    # `master` HER ZAMAN kontrol noktasindadir; kampanya isi `kampanya_*`
    # dalinda durur. Geri donus = master'a gecmek. Boylece kampanya
    # commit'leri KAYBOLMAZ, only calisma agacindan cikar.
        subprocess .run (["git","checkout","-qf","master"],cwd =KOK ,
        check =False )
        subprocess .run (["git","reset","-q","--hard",man ["git"]],cwd =KOK ,
        check =True )
        subprocess .run (["git","clean","-fdq"],cwd =KOK ,check =False )
        print ("      -> master'a donuldu (veri dizinleri .gitignore'da, ELLENMEDI;"
        " kampanya dali duruyor)")

        # --- 2) MODEL -------------------------------------------------------
    print (f"\n-- 2) MODEL --  {len (man ['kopya'])} dosya")
    for rel ,m in man ["kopya"].items ():
        src =os .path .join (kn ,"dosyalar",rel .replace ("/",os .sep ))
        dst =os .path .join (KOK ,rel .replace ("/",os .sep ))
        var =os .path .exists (dst )and sha (dst )==m ["sha256"]
        if var :
            continue 
        print (f"      geri kopyalanacak: {rel }")
        if not kuru :
            os .makedirs (os .path .dirname (dst ),exist_ok =True )
            shutil .copy2 (src ,dst )

    if kuru :
        print ("\n(KURU KOSU -- hicbir sey degistirilmedi)")
        return 

        # --- 3) DOGRULA -----------------------------------------------------
    print ("\n-- 3) DOGRULAMA --")
    ks ,ds ,ps =dogrula (man )
    if ks :
        print ("\nHATA: kopya kumesi geri alinamadi.")
        sys .exit (1 )
    if ds or ps :
        print ("\nUYARI: GERI ALINAMAZ kumede deviation var (yukarida). Kod ve model")
        print ("geri alindi, ama bu dosyalar kontrol noktasindaki hallerinde DEGIL.")
        print ("Yeniden uretilmeleri gerekir.")
        sys .exit (2 )
    print ("\nTAM DONUS BASARILI -- her sey kontrol noktasindaki gibi.")
    print ("Kaniti icin:  python probe_dagitim_dogrula.py   (beklenen robot 0.2980)")


if __name__ =="__main__":
    main ()
