"""Click WSCAD's download button ten a part page and dump the dialog that opens,
so the bulk downloader knows how to pick the 3D/STEP format."""
import os 
import sys 

from playwright .sync_api import sync_playwright 

PROFILE =os .path .abspath ("_wscad_profile")
PART =("https://www.wscaduniverse.com/part?manufacturerId={mid}&partNumber={art}"
"&format=WSCAD&norm=IEC")


def main ():
    art =sys .argv [1 ]if len (sys .argv )>1 else "3001035"
    mid =sys .argv [2 ]if len (sys .argv )>2 else "63"
    os .makedirs ("_wscad_dl",exist_ok =True )

    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFILE ,headless =True ,accept_downloads =True ,
        downloads_path =os .path .abspath ("_wscad_dl"))
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .goto (PART .format (mid =mid ,art =art ),wait_until ="networkidle",
        timeout =90000 )
        page .wait_for_timeout (4000 )

        # the toolbar 'download' icon-button ten the part card
        btn =page .locator ("button:has-text('download')").first 
        print ("download butonu was found:",btn .count ()>0 )
        btn .click ()
        page .wait_for_timeout (4000 )
        print ("tiklandi -- opened icerik:\n")

        body =page .inner_text ("body")
        print (body [:1200 ].replace ("\n"," | "))

        print ("\n--- dialog kontrolleri ---")
        for sel in ("mat-dialog-container","[role=dialog]","mat-checkbox",
        "input[type=checkbox]","mat-select","select",
        "button:has-text('Download')","button:has-text('OK')",
        "mat-radio-button"):
            n =page .locator (sel ).count ()
            if n :
                print (f"  {sel }: {n }")
                for i in range (min (n ,12 )):
                    try :
                        t =(page .locator (sel ).nth (i ).inner_text ()or "").strip ()[:50 ]
                        print (f"     [{i }] {t !r }")
                    except Exception :
                        pass 
        ctx .close ()


if __name__ =="__main__":
    main ()
