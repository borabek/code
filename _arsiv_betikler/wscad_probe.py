"""Drive WSCAD Universe like a human ten ONE article number and dump what appears,
so the bulk downloader is written against the real DOM. Reuses the logged-in
profile (_wscad_profile).

Usage: python wscad_probe.py 3001035 [--headed]
"""
import os 
import sys 

from playwright .sync_api import sync_playwright 

PROFILE =os .path .abspath ("_wscad_profile")


def main ():
    art =sys .argv [1 ]if len (sys .argv )>1 else "3001035"
    headed ="--headed"in sys .argv 

    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFILE ,headless =not headed ,accept_downloads =True )
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .goto ("https://www.wscaduniverse.com/en/",wait_until ="networkidle",
        timeout =90000 )
        page .wait_for_timeout (2500 )

        box =page .locator ("input[placeholder*='search' i]").first 
        box .click ()
        box .fill (art )
        box .press ("Enter")
        print (f"[1] '{art }' arandi -- sonuclar bekleniyor")
        page .wait_for_timeout (6000 )
        try :
            page .wait_for_load_state ("networkidle",timeout =30000 )
        except Exception :
            pass 

        print ("[2] URL:",page .url )
        # sonuc kartlari / linkler
        links =page .eval_on_selector_all (
        "a[href]",
        "els => els.map(e => e.getAttribute('href')).filter(h => h && "
        "(h.includes('part') || h.includes('artikel') || h.includes('product')))")
        uniq =sorted (set (links ))[:10 ]
        print (f"[3] part linki adaylari ({len (set (links ))}):")
        for u in uniq :
            print ("     ",u )

        if uniq :
            target =uniq [0 ]
            if target .startswith ("/"):
                target ="https://www.wscaduniverse.com"+target 
            page .goto (target ,wait_until ="networkidle",timeout =90000 )
            page .wait_for_timeout (4000 )
            print ("[4] part sayfasi:",page .url )
            print ("    baslik:",page .title ())
            txt =page .inner_text ("body")[:400 ].replace ("\n"," | ")
            print ("    metin:",txt )

            # indirme kontrolleri
            for sel in ("button:has-text('Download')","a:has-text('Download')",
            "button:has-text('3D')","a:has-text('3D')",
            "*:has-text('STEP')","select","[class*=download i]"):
                n =page .locator (sel ).count ()
                if n :
                    print (f"    kontrol {sel }: {n }")
                    if n <=6 :
                        for i in range (n ):
                            try :
                                t =page .locator (sel ).nth (i ).inner_text ()[:60 ]
                                print (f"        [{i }] {t !r }")
                            except Exception :
                                pass 
        ctx .close ()


if __name__ =="__main__":
    main ()
