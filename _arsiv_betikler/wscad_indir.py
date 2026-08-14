# -*- coding: utf-8 -*-
"""WSCAD UNIVERSE STEP INDIRICI -- korpusta STEP'i olmayan parts for.

AKIS (2026-08-04'te single single kesfedildi, korukorune yazilmadi):
  1. dogrudan arama URL'si: /search?searchText=<no>&norm=IEC&format=WSCAD&limit=96&offset=0
  2. sonuc TABLO satiri (`a[href*=part-detail]` YOK, URL /home'da kalabilir)
  3. satirdaki `mat-icon` metni "download" which is BUTON -> menu acilir
  4. menude UC secenek: "Download WSCAD" / "Download DWG" / **"Download STEP"**
  5. STEP secilir -> tarayici indirme olayi

CHECK GRUBU DERSI: first tarayicim HER SEYE "bulunamadi" dedi, korpusta STEP'i OLAN
PXC/WEI parcalari dahil. Arac bozuktu, data not. Bu yuzden this betik de each kosuda
CHECK parcalari indirir and onlar basarisiz olursa "bulunamadi" sonuclarini INVALID
sayar (sessizce sifir file indirip "bitti" demez).

KULLANIM
    python wscad_indir.py --n 40            # katmanli ornek, before bunu kos
    python wscad_indir.py --manufacturer A-B     # single manufacturer
    python wscad_indir.py --all of them           # STEP'i olmayan HER part (uzun surer)
"""
import argparse 
import glob 
import io 
import json 
import os 
import random 
import re 
import shutil 
import time 
import zipfile 

ANA ="https://www.wscaduniverse.com"
ARAMA =(ANA +"/search?searchText={}&norm=IEC&format=WSCAD"
"&manufacturerIds=&symbolsInTechnologies=&category=&limit=96&offset=0")
PROFIL =os .path .join (os .environ .get ("TEMP","."),"wscad_pw_profil")
GECICI =os .path .join (os .getcwd (),"_wscad_indirilen")
HEDEF_DIZIN ="all_wscad_stp"
KAYIT ="results/wscad_indirme_kaydi.json"
# CHECK: korpusta STEP'i OLAN parts -- portalda KESINLIKLE bulunmalilar
KONTROL =["3031759","2881490000"]


def _kayit_yukle ():
    if os .path .exists (KAYIT ):
        with io .open (KAYIT ,encoding ="utf-8")as f :
            return json .load (f )
    return {}


def _var_olan_step ():
    return {os .path .basename (s ).split ("_")[1 ]for s in glob .glob (HEDEF_DIZIN +"/*.stp")}


def hedefleri_sec (n ,manufacturer ,hepsi ):
    man =json .load (io .open ("results/manifest_korpus.json",encoding ="utf-8"))
    var =_var_olan_step ()
    candidate =[m for m in man if m ["cp"]>0 and m ["part"]and m ["part"]not in var ]
    if manufacturer :
        candidate =[m for m in candidate if m ["manufacturer"]==manufacturer ]
    if hepsi :
        return candidate 
        # KATMANLI: manufacturer x CP kovasi dengeli
    rng =random .Random (0 )
    gruplar ={}
    for m in candidate :
        gruplar .setdefault ((m ["manufacturer"],m ["kova"]),[]).append (m )
    out ,anahtarlar =[],sorted (gruplar )
    for k in anahtarlar :
        rng .shuffle (gruplar [k ])
    i =0 
    while len (out )<n and any (gruplar [k ]for k in anahtarlar ):
        k =anahtarlar [i %len (anahtarlar )]
        if gruplar [k ]:
            out .append (gruplar [k ].pop ())
        i +=1 
    return out 


def _step_kaydet (yol ,part ):
    """Inen dosyayi corpus adlandirmasiyla instead of koy. ZIP whereas icinden .stp cikarilir."""
    zaman =time .strftime ("%Y-%m-%d-%H-%M-%S")
    hedef =os .path .join (HEDEF_DIZIN ,f"wscaduniverse_{part }_{zaman }.stp")
    if zipfile .is_zipfile (yol ):
        with zipfile .ZipFile (yol )as z :
            adlar =[a for a in z .namelist ()if a .lower ().endswith ((".stp",".step"))]
            if not adlar :
                return None ,f"ZIP icinde .stp yok ({z .namelist ()[:3 ]})"
            with z .open (adlar [0 ])as src ,io .open (hedef ,"wb")as dst :
                shutil .copyfileobj (src ,dst )
    else :
        shutil .copy2 (yol ,hedef )
    if os .path .getsize (hedef )<2000 :
        os .remove (hedef )
        return None ,"file very small (empty/error sayfasi)"
    return hedef ,None 


def indir (pg ,ctx ,part ,bekle =25000 ):
    """(state, ayrinti). state: 'ok' | 'bulunamadi' | 'step_yok' | 'error'"""
    pg .goto (ARAMA .format (part ),wait_until ="domcontentloaded")
    try :
        pg .wait_for_load_state ("networkidle",timeout =25000 )
    except Exception :
        pass 
    time .sleep (2.0 )
    if "no results"in pg .content ().lower ():
        return "bulunamadi",""
    btn =None 
    for b in pg .query_selector_all ("tr button"):
        try :
            if (b .inner_text ()or "").strip ()=="download"and b .is_visible ():
                btn =b 
                break 
        except Exception :
            pass 
    if btn is None :
        return "bulunamadi","indirme dugmesi absent"
    btn .scroll_into_view_if_needed ()
    btn .click ()
    time .sleep (1.5 )
    ogeler =pg .query_selector_all (".cdk-overlay-pane button, .cdk-overlay-pane [role=menuitem]")
    step =None 
    for o in ogeler :
        if "step"in (o .inner_text ()or "").lower ():
            step =o 
            break 
    if step is None :
        pg .keyboard .press ("Escape")
        return "step_yok",f"menu: {[(o .inner_text ()or '').strip ()[:20 ]for o in ogeler ]}"
    try :
        with pg .expect_download (timeout =bekle )as bilgi :
            step .click ()
        d =bilgi .value 
        gec =os .path .join (GECICI ,d .suggested_filename or f"{part }.dat")
        d .save_as (gec )
    except Exception as e :
        pg .keyboard .press ("Escape")
        return "error",f"{type (e ).__name__ }: {str (e )[:70 ]}"
    yol ,error =_step_kaydet (gec ,part )
    try :
        os .remove (gec )
    except Exception :
        pass 
    return ("ok",os .path .basename (yol ))if yol else ("error",error )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--n",type =int ,default =40 )
    ap .add_argument ("--manufacturer",default ="")
    ap .add_argument ("--all of them",action ="store_true")
    ap .add_argument ("--gecikme",type =float ,default =1.5 ,help ="parts arasi saniye")
    a =ap .parse_args ()

    os .makedirs (GECICI ,exist_ok =True )
    rec_ =_kayit_yukle ()
    hedef =hedefleri_sec (a .n ,a .manufacturer ,a .hepsi )
    hedef =[m for m in hedef if rec_ .get (m ["part"],{}).get ("state")!="ok"]
    print (f"hedef: {len (hedef )} part | kayitta {len (rec_ )} onceki deneme",flush =True )

    from playwright .sync_api import sync_playwright 
    t0 =time .time ()
    sayac ={}
    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFIL ,channel ="msedge",headless =False ,accept_downloads =True ,
        downloads_path =GECICI ,viewport ={"width":1500 ,"height":950 })
        pg =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        pg .set_default_timeout (45000 )

        # --- CHECK: arac saglam mi
        print ("CHECK parcalari (arac saglamlik sinavi):",flush =True )
        k_ok =0 
        for c in KONTROL :
            d ,ay =indir (pg ,ctx ,c )
            print (f"   {c :<14}{d }  {ay [:60 ]}",flush =True )
            k_ok +=(d =="ok")
        if k_ok ==0 :
            print ("\n!! CHECK GRUBU TAMAMEN BASARISIZ -- arac bozuk, results INVALID.")
            print ("   Indirmeye devam EDILMIYOR (silent sifir riski).")
            ctx .close ()
            return 
        print (f"   -> {k_ok }/{len (KONTROL )} kontrol indi, arac saglam\n",flush =True )

        for i ,m in enumerate (hedef ,1 ):
            part =m ["part"]
            try :
                d ,ay =indir (pg ,ctx ,part )
            except Exception as e :
                d ,ay ="error",f"{type (e ).__name__ }: {str (e )[:60 ]}"
            sayac [d ]=sayac .get (d ,0 )+1 
            rec_ [part ]={"state":d ,"ayrinti":ay ,"manufacturer":m ["manufacturer"],
            "zaman":time .strftime ("%Y-%m-%d %H:%M:%S")}
            print (f"  [{i }/{len (hedef )}] {m ['manufacturer']:<7}{part :<24}{d :<12}{ay [:52 ]}",flush =True )
            if i %10 ==0 :
                with io .open (KAYIT ,"w",encoding ="utf-8")as f :
                    json .dump (rec_ ,f ,indent =1 ,ensure_ascii =False )
            time .sleep (a .gecikme )
        ctx .close ()

    with io .open (KAYIT ,"w",encoding ="utf-8")as f :
        json .dump (rec_ ,f ,indent =1 ,ensure_ascii =False )
    sure =time .time ()-t0 
    print (f"\nSONUC: {sayac }")
    print (f"sure {sure :.0f}s | part basina {sure /max (len (hedef ),1 ):.1f}s")
    ok =sayac .get ("ok",0 )
    print (f"ISABET ORANI: {ok }/{len (hedef )} = {ok /max (len (hedef ),1 ):.1%}")
    print (f"kayit -> {KAYIT } | STEP -> {HEDEF_DIZIN }/")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
