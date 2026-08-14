# -*- coding: utf-8 -*-
"""RECON + session setup for wscaduniverse, so the bulk STEP fetch can be written against real calls.

WHY EDGE: `playwright install chromium` fails ten this network ("Download failure, code=1"); Edge is
installed and playwright drives it with channel="msedge", no download needed.
WHY A PERSISTENT PROFILE: the site is a login-gated SPA (no visible search form, links point at
auth.wscaduniverse.com). Logging in ONCE headed leaves the session in _wsprofile for later runs.
WHY NETWORK CAPTURE: the SPA's download flow is unknown. Watching ONE manual download reveals the
request that actually returns the STEP, which is all the bulk fetcher needs.

Run it, log in, then search ONE part and download its STEP as you normally would. Everything the
browser requested is written to results/wscad_net.json.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe wscad_recon.py [--minutes 8]
"""
import os ,json ,argparse 

PROFILE =os .path .abspath ("_wsprofile")
HOME ="https://www.wscaduniverse.com/en/"


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--minutes",type =int ,default =8 ,help ="how long the browser stays open for you")
    a =ap .parse_args ()
    from playwright .sync_api import sync_playwright 

    os .makedirs (PROFILE ,exist_ok =True );os .makedirs ("results",exist_ok =True )
    seen =[]

    def on_req (r ):
        u =r .url 
        if any (k in u .lower ()for k in ("step","stp","download","cad","export","file","api")):
            rec ={"method":r .method ,"url":u [:400 ],"type":r .resource_type }
            # CAPTURE THE BODY: the download is POST /api/download/part, and calling that API directly
            # is far more robust than clicking through an unknown SPA. Without the payload we cannot.
            if r .method =="POST":
                try :
                    rec ["post_data"]=(r .post_data or "")[:1500 ]
                except Exception :
                    pass 
                try :
                    h =r .headers 
                    rec ["headers"]={k :v [:120 ]for k ,v in h .items ()
                    if k .lower ()in ("content-type","authorization","x-api-key","accept")}
                except Exception :
                    pass 
            seen .append (rec )

    def on_dl (d ):
        seen .append ({"DOWNLOAD":d .suggested_filename ,"url":d .url [:400 ]})
        try :
            os .makedirs ("_wsdl",exist_ok =True )
            d .save_as (os .path .join ("_wsdl",d .suggested_filename ))
            print (f"  [indirme yakalandi] {d .suggested_filename }",flush =True )
        except Exception as e :
            print (f"  [indirme kaydedilemedi] {str (e )[:80 ]}",flush =True )

    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFILE ,channel ="msedge",headless =False ,accept_downloads =True ,
        args =["--start-maximized"],no_viewport =True )
        ctx .on ("request",on_req )
        ctx .on ("page",lambda pg :pg .on ("download",on_dl ))
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .on ("download",on_dl )
        page .goto (HOME ,wait_until ="domcontentloaded",timeout =60000 )

        print ("\n"+"="*70 )
        print ("  TARAYICI ACIK.  Sirasiyla:")
        print ("   1) wscaduniverse'e GIRIS YAP")
        print ("   2) Bir part ARA  (ornek WEI part no: 1010100000)")
        print ("   3) O parcanin STEP / 3D dosyasini NORMAL SEKILDE INDIR")
        print (f"  Pencere {a .minutes } dakika acik kalacak, sonra kendim kapatacagim.")
        print ("  Ben bu sirada butun ag isteklerini kaydediyorum.")
        print ("="*70 +"\n",flush =True )

        for m in range (a .minutes ):
            page .wait_for_timeout (60000 )
            print (f"  ... {m +1 }/{a .minutes } dk  ({len (seen )} ilgili istek yakalandi)",flush =True )

        json .dump ({"n":len (seen ),"requests":seen [-250 :]},open ("results/wscad_net.json","w"),indent =1 )
        print (f"\n{len (seen )} istek kaydedildi -> results/wscad_net.json",flush =True )
        dls =[s for s in seen if "DOWNLOAD"in s ]
        if dls :
            print ("  INDIRME ISTEKLERI:")
            for d in dls :print ("   ",d ["DOWNLOAD"],"<-",d ["url"][:120 ])
        else :
            print ("  (indirme yakalanmadi -- indirmeyi tamamlayabildin mi?)")
        ctx .close ()


if __name__ =="__main__":
    main ()
