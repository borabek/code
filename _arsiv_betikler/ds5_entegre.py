# -*- coding: utf-8 -*-
"""DataSet 5 ENTEGRASYONU: 4788 JSON + 4788 STP (birebir eslesmis).

ONEMI: korpusta STEP'i EKSIK 2120 klemens vardi and WSCAD'den indirmek KESIN KAPALIYDI
([[wscad-indirme-conclusive-closed]]: 24 denemede 0 basari, reason portalda 3D VERI YOKLUGU).
Bu paket that eksigin **2391 parcasini / 8410 CP'sini** kapatiyor + 403 tamamen new part
(IKI YENI URETICI: WIE 280, WEG 123) getiriyor.

DOGRULANDI (entegrasyondan ONCE): 8 different ureticiden rastgele part -- STEP'ler
ayristi, JSON mesh'i with STEP mesh'inin boundary kutusu <=0.11mm ortusuyor, i.e. same part.

ADLANDIRMA: `eligible()` STEP'i `corpus_identity.step_kimlik` with reads and
`wscaduniverse_<pid>_<zaman>.stp` kalibini bekler. DataSet5 dosyalari JSON with AYNI ada
sahip (`A-B.1492-H4_ElectricalTerminal_ElectricalTerminal.stp`) -- oldugu like kopyalanirsa
kimlik "ElectricalTerminal_ElectricalTerminal" cikar. Bu yuzden KOPYALARKEN yeniden adlandirilir.

SAFETY: STEP'i ZATEN OLAN part ATLANIR (kopya olusturmaz). Olcum kumesinin (194 part)
degismedigi entegrasyondan after DOGRULANIR.
"""
import io 
import json 
import os 
import shutil 
import sys 
import time 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from corpus_identity import kimlik ,step_kimlik 

KAYNAK ="_ds5_stage/DataSet"
STEP_DIZIN ="all_wscad_stp"
JSON_DIZIN ="_ds1/DataSet"
ZAMAN =time .strftime ("%Y-%m-%d-%H-%M-%S")


def main ():
    import glob 
    term =[f for f in glob .glob (os .path .join (KAYNAK ,"*.json"))
    if "ElectricalTerminal"in os .path .basename (f )]
    var_step ={step_kimlik (s )for s in glob .glob (os .path .join (STEP_DIZIN ,"*.stp"))}
    var_json ={os .path .basename (f )for f in glob .glob (os .path .join (JSON_DIZIN ,"*.json"))}
    print (f"kaynak: {len (term )} ElectricalTerminal JSON")
    print (f"mevcut STEP kimligi {len (var_step )} | mevcut JSON {len (var_json )}")

    n_step =n_json =atlanan =stp_yok =0 
    rapor ={"step_eklenen":[],"json_eklenen":[]}
    for f in term :
        b =os .path .basename (f )
        mfg ,pid =kimlik (b )
        if not pid :
            continue 
        s =f [:-5 ]+".stp"
        if not os .path .exists (s ):
            stp_yok +=1 
            continue 
            # 1) JSON: korpusta otherwise ekle
        if b not in var_json :
            shutil .copy2 (f ,os .path .join (JSON_DIZIN ,b ))
            n_json +=1 
            rapor ["json_eklenen"].append (b )
            # 2) STEP: kimligi already varsa ATLA (kopya olusturma)
        if pid in var_step :
            atlanan +=1 
            continue 
        hedef =os .path .join (STEP_DIZIN ,f"wscaduniverse_{pid }_{ZAMAN }.stp")
        shutil .copy2 (s ,hedef )
        var_step .add (pid )
        n_step +=1 
        rapor ["step_eklenen"].append (pid )
    print (f"\nEKLENEN JSON : {n_json }")
    print (f"EKLENEN STEP : {n_step }")
    print (f"ATLANAN STEP (kimligi zaten vardi): {atlanan }")
    print (f"STP dosyasi bulunamayan: {stp_yok }")
    with io .open ("results/ds5_entegre.json","w",encoding ="utf-8")as fh :
        json .dump ({"json_eklenen":n_json ,"step_eklenen":n_step ,"atlanan":atlanan ,
        "zaman":ZAMAN ,"ornek_step":rapor ["step_eklenen"][:20 ]},
        fh ,indent =1 ,ensure_ascii =False )
    print ("receipt -> results/ds5_entegre.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
