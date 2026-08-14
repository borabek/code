# -*- coding: utf-8 -*-
"""WSCAD VARLIK TARAMASI: new ureticilerin parcalari portalda VAR MI? (indirme YOK)

WHY BU ADIM ONCE: first kesifte `8WH2040-4LF00` (SIE) for portal "NO RESULTS" dedi.
Yani elimizdeki JSON'lardaki part numaralari WSCAD Universe katalogunda BIREBIR
gecmiyor may be -- JSON'lar baska a kaynaktan gelmis. Indirme dongusu yazmadan
before "this parts orada present mi" sorusu yanitlanmali; otherwise mukemmel a otomasyon
empty kumeyi indirir.

Kontrol grubu as korpusta ZATEN STEP'i which is PXC/WEI parcalari da aranir: onlar
bulunuyorsa arama akisi saglamdir and "bulunamadi" sonucu GERCEKTEN yokluk demektir.

Cikti: results/wscad_varlik.json  (manufacturer basina was found/bulunamadi + ornek URL)
"""
import io 
import re 
import json 
import os 
import random 
import sys 
import time 

PROFIL =os .path .join (os .environ .get ("TEMP","."),"wscad_pw_profil")
ANA ="https://www.wscaduniverse.com/"
N_PER =3 


def ornekle ():
    """Manifestten manufacturer basina N part sec + kontrol grubu (STEP'i OLAN PXC/WEI)."""
    man =json .load (io .open ("results/manifest_korpus.json",encoding ="utf-8"))
    ok =[m for m in man if m ["cp"]>0 ]
    hedef ={}
    for u in ("CWT","A-B","A-B_N","KLM","SIE","CCD","ELMEX","ABB","AL","DIN"):
        alt =[m for m in ok if m ["manufacturer"]==u and not m ["step"]]
        if alt :
            random .Random (0 ).shuffle (alt )
            hedef [u ]=[m ["part"]for m in alt [:N_PER ]]
            # CHECK: STEP'i which is, i.e. portalda kesinlikle bulunmasi gereken parts
    for u in ("PXC","WEI"):
        alt =[m for m in ok if m ["manufacturer"]==u and m ["step"]]
        random .Random (0 ).shuffle (alt )
        hedef ["KONTROL_"+u ]=[m ["part"]for m in alt [:2 ]]
    return hedef 


def ara (page ,no ):
    """Parca numarasini ara. (bulundu_mu, sonuc_url, kac_kart)"""
    page .goto (ANA ,wait_until ="domcontentloaded")
    time .sleep (1.5 )
    kutu =None 
    for sel in ("input[placeholder*='earch']","input[type=search]","input[type=text]"):
        el =page .query_selector (sel )
        if el and el .is_visible ():
            kutu =el 
            break 
    if kutu is None :
        return (None ,page .url ,-1 )
    kutu .click ()
    kutu .fill (no )
    page .keyboard .press ("Enter")
    try :
        page .wait_for_load_state ("networkidle",timeout =40000 )
    except Exception :
        pass 
    time .sleep (3.0 )
    icerik =page .content ()
    # DUZELTME (first version HER SEYE 'absent' dedi, CHECK grubu dahil -- arac bozuktu):
    # results `a[href*='part-detail']` DEGIL, a TABLO satiri as geliyor and URL
    # '/home' as KALIYOR. Dogru gosterge: "Search Results" basligi altindaki
    # "N part(s)" sayaci. Kontrol grubu (STEP'i which is PXC/WEI) this duzeltmeyi dogrular.
    yok ="no results"in icerik .lower ()
    n =-1 
    m =re .search (r">\s*(\d+)\s+parts?\s*<",icerik ,re .I )
    if m :
        n =int (m .group (1 ))
    if n <0 :# backup: part numarasini iceren tablo satiri say
        n =len ([r for r in page .query_selector_all ("tr, [role=row]")
        if no .lower ()in (r .inner_text ()or "").lower ()])
    return ((not yok )and n >0 ,page .url ,n )


def main ():
    from playwright .sync_api import sync_playwright 
    hedef =ornekle ()
    print ("aranacak:",{k :len (v )for k ,v in hedef .items ()},flush =True )
    rapor ={}
    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        PROFIL ,channel ="msedge",headless =False ,
        accept_downloads =True ,viewport ={"width":1500 ,"height":950 })
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        page .set_default_timeout (45000 )
        for u ,nolar in hedef .items ():
            rapor [u ]=[]
            for no in nolar :
                try :
                    b ,url ,k =ara (page ,no )
                except Exception as e :
                    b ,url ,k =(None ,f"HATA {type (e ).__name__ }",-1 )
                rapor [u ].append ({"part":no ,"bulundu":b ,"kart":k ,"url":url })
                print (f"  {u :<12}{no :<24}{'BULUNDU'if b else ('YOK'if b is False else 'HATA')}"
                f"  (kart {k })",flush =True )
        with io .open ("results/wscad_varlik.json","w",encoding ="utf-8")as f :
            json .dump (rapor ,f ,indent =1 ,ensure_ascii =False )
        print ("\nOZET")
        for u ,rs in rapor .items ():
            v =sum (1 for r in rs if r ["bulundu"])
            print (f"  {u :<12}{v }/{len (rs )} bulundu")
        print ("\nmakbuz -> results/wscad_varlik.json")
        ctx .close ()


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
