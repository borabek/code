# -*- coding: utf-8 -*-
"""WSCAD OTURUMU AC + INDIRME AKISINI KESFET (faz 1: kesif, toplu indirme YOK).

WHY SEPARATE ADIM: wscaduniverse.com giris istiyor and 3D indirme also Cadenas kaydi
istiyor. Kullanicinin KENDI tarayicisindaki oturum Playwright'a GECMEZ -- Playwright
own profilini acar. Bu yuzden KALICI PROFIL kullaniyoruz: pencere a times acilir,
kullanici ICINDE giris yapar, oturum diske yazilir and sonraki kosularda korunur.

BU BETIK INDIRMEZ. Yaptigi:
  1. Kalici profille Edge acar (headed -- kullanici gormeli)
  2. Girisi BEKLER (sayfada oturum belirteci cikana up to yoklar)
  3. TEK a parcanin sayfasina gider and 3D/indirme akisinin DOM yapisini dokumler
Amac: toplu indirme dongusunu KORU KORUNE yazmamak. Akisi gordukten after loop yazilir.

Cikti: results/wscad_kesif.json + results/wscad_kesif.html (sayfa iskeleti)
"""
import io 
import json 
import os 
import sys 
import time 

PROFIL =os .path .join (os .environ .get ("TEMP","."),"wscad_pw_profil")
HEDEF ="8WH2040-4LF00"# SIE, GT'si elimizde, STEP'i absent
ANA ="https://www.wscaduniverse.com/"


def giris_var_mi (page ):
    """Oturum open mi? Login baglantisi KAYBOLDUYSA and a hesap/cikis izi VARSA."""
    try :
        u =page .url .lower ()
        if "auth/login"in u or "signin"in u :
            return False 
        icerik =page .content ().lower ()
        cikis =any (k in icerik for k in ("logout","abmelden","sign out","my account",
        "mein konto","profile"))
        login =("auth/login"in icerik )or ("anmelden"in icerik and not cikis )
        return cikis and not login 
    except Exception :
        return False 


def main ():
    from playwright .sync_api import sync_playwright 
    os .makedirs (PROFIL ,exist_ok =True )
    rapor ={"profil":PROFIL ,"hedef":HEDEF }
    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFIL ,channel ="msedge",headless =False ,
        accept_downloads =True ,viewport ={"width":1500 ,"height":950 },
        args =["--disable-blink-features=AutomationControlled"])
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .set_default_timeout (60000 )
        print (f"pencere acildi -- profil: {PROFIL }",flush =True )
        page .goto (ANA ,wait_until ="domcontentloaded")
        time .sleep (3 )

        if giris_var_mi (page ):
            print ("OTURUM ZATEN OPEN",flush =True )
        else :
            print ("\n>>> ACILAN PENCEREDE GIRIS YAP. Bekliyorum (en fazla 10 dk)...",flush =True )
            t0 =time .time ()
            while time .time ()-t0 <600 :
                time .sleep (5 )
                if giris_var_mi (page ):
                    print (f"GIRIS ALGILANDI ({time .time ()-t0 :.0f}s)",flush =True )
                    break 
            else :
                print ("GIRIS ALGILANMADI -- yine de kesfe devam ediyorum",flush =True )
        rapor ["giris"]=giris_var_mi (page )
        rapor ["url_ana"]=page .url 

        # --- part arama
        print (f"\n'{HEDEF }' araniyor...",flush =True )
        bulundu =False 
        for sel in ("input[type=search]","input[placeholder*='uch']","input[placeholder*='earch']",
        "input[name*='search']","#searchInput","input[type=text]"):
            try :
                el =page .query_selector (sel )
                if el and el .is_visible ():
                    el .click ();el .fill (HEDEF );page .keyboard .press ("Enter")
                    page .wait_for_load_state ("networkidle",timeout =45000 )
                    bulundu =True 
                    print (f"  arama kutusu: {sel }",flush =True )
                    break 
            except Exception :
                continue 
        rapor ["arama_yapildi"]=bulundu 
        rapor ["url_sonuc"]=page .url 
        time .sleep (3 )

        # --- 3D / indirme kontrolu present mi
        izler ={}
        for ad ,sel in (("3d_metin","text=/3D/i"),("download_metin","text=/download|herunterladen|indir/i"),
        ("cad_metin","text=/CAD/i"),("iframe","iframe"),
        ("part_detail_link","a[href*='part-detail']")):
            try :
                els =page .query_selector_all (sel )
                izler [ad ]=len (els )
            except Exception :
                izler [ad ]=-1 
        rapor ["izler"]=izler 
        try :
            rapor ["iframe_src"]=[f .url for f in page .frames ][:10 ]
        except Exception :
            pass 
        print (f"  sayfa izleri: {izler }",flush =True )

        with io .open ("results/wscad_kesif.html","w",encoding ="utf-8")as f :
            f .write (page .content ()[:400000 ])
        page .screenshot (path ="results/wscad_kesif.png",full_page =False )
        with io .open ("results/wscad_kesif.json","w",encoding ="utf-8")as f :
            json .dump (rapor ,f ,indent =1 ,ensure_ascii =False )
        print ("\nmakbuz -> results/wscad_kesif.json (+ .html, .png)",flush =True )
        print ("PENCERE OPEN BIRAKILIYOR (60s) -- after kapanir, oturum profilde KALIR",flush =True )
        time .sleep (60 )
        ctx .close ()


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
