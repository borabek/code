# -*- coding: utf-8 -*-
"""KORPUS KIMLIGI: part numarasini DOGRU ayristir + manifest uret (F2-9).

BULUNAN HATA (2026-08-04, dataset-4 entegrasyonunda): `big_arbiter.eligible()` part
kimligini soyle cikariyor:

    head = os.path.basename(f).split("_")[0]
    mfg, pid = (head.split(".", 1) + [""])[:2]

Bu IKI places kiriliyor:
  * URETICI kodunda lower line varsa (`A-B_N.1492-H4_...`) -> head = "A-B", point absent,
    **pid BOS** -> part korpusa ASLA giremez. Yeni veride 233 file.
  * PARCA NUMARASINDA lower line varsa (`ELMEX.KUT16_GY_...`) -> pid = "KUT16",
    "_GY" DUSER. Yeni veride 397 file; 30 ayri kimlik CAKISIYOR (for example 14 different
    part "CX2.5" kimligine dusuyor).

STEP dosyasi tarafinda same kesme present (`wscaduniverse_<pid>_<ts>.stp` -> split("_")[1]),
i.e. kesilme TUTARLI and eslesme most zaman yine calisiyor. TEHLIKE cakismada: two different
part same kesik kimlige duserse, a STEP dosyasi YANLIS JSON'a baglanabilir -- and this
sessizce becomes, error vermez.

Bu modul correct ayristirmayi TEK YERDE tanimlar. `big_arbiter`'i DEGISTIRMEZ: kimlik
degisimi measurement kumesini oynatabilecegi for this a PROTOKOL degisikligidir (F0-1) and
etkisi olculmeden yapilmaz. Once DIAGNOSIS.
"""
import collections 
import glob 
import io 
import json 
import os 
import re 

DS ="_ds1/DataSet"
STP ="all_wscad_stp"


def kimlik (dosya_adi ):
    """(manufacturer, parca_no) -- lower cizgili manufacturer VE part numaralarini correct separates.

    `A-B_N.1492-H4_ElectricalTerminal_ElectricalTerminal.json` -> ("A-B_N", "1492-H4")
    `ELMEX.KUT16_GY_ElectricalTerminal_...`                    -> ("ELMEX", "KUT16_GY")
    `PXC.3271055_ElectricalTerminal_...`                       -> ("PXC", "3271055")
    """
    b =os .path .basename (dosya_adi )
    if "."not in b :
        return ("","")
    mfg ,remaining =b .split (".",1 )
    # part numarasi, first "_ElectricalTerminal" (ya da baska a "_Electrical*") oncesine up to
    m =re .split (r"_Electrical|_Mechanics|_Common",remaining ,maxsplit =1 )
    return (mfg ,m [0 ])


def kimlik_eski (dosya_adi ):
    """big_arbiter'in SU ANDA kullandigi (hatali) ayristirma -- comparison for."""
    h =os .path .basename (dosya_adi ).split ("_")[0 ]
    return tuple ((h .split (".",1 )+[""])[:2 ])


def step_kimlik (path ):
    """STEP file adindan kimlik -- IKI adlandirma sozlesmesini de anlar.

    A) INDIRME sozlesmesi : `wscaduniverse_<pid>_<ts>.stp`      -> pid
    B) JSON sozlesmesi    : `<MFG>.<pid>_ElectricalTerminal_... -> pid

    (B) 2026-08-05'te DataSet 6 with geldi and SESSIZ a cokuse path acti: old kod
    `b.split("_")[1:]`i birlestiriyordu, i.e. 1656 STEP'in HEPSI same anahtara
    ("ElectricalTerminal_ElectricalTerminal") dusuyordu. `eligible()` STEP sozlugunu
    anahtarla kurdugu for 1655 file sessizce EZILIYOR, corpus 4405'te KALIYORDU --
    no error verilmeden. Sozlesme before ayirt edilir, after ayristirilir.
    """
    b =os .path .splitext (os .path .basename (path ))[0 ]
    # (B): JSON sozlesmesi -- bilesen eki present VE ondan ONCE point present.
    # Nokta ILK lower cizgiden before olmak ZORUNDA DEGIL: `A-B_N.1492-H4_Electrical...`
    # like ALT CIZGILI manufacturer kodlarinda point ikinci parcadadir.
    m =re .search (r"_Electrical|_Mechanics|_Common",b )
    if m and "."in b [:m .start ()]:
        return kimlik (b )[1 ]
    p =b .split ("_")
    if len (p )<2 :
        return b 
        # last part zaman damgasi whereas (2026-...) onu at, ortadakileri birlestir
    if re .match (r"^\d{4}-\d{2}-\d{2}",p [-1 ]):
        return "_".join (p [1 :-1 ])
    return "_".join (p [1 :])


def main ():
    js =sorted (glob .glob (os .path .join (DS ,"*.json")))
    stp =glob .glob (os .path .join (STP ,"*.stp"))
    print (f"JSON {len (js )} | STEP {len (stp )}")

    # --- 1) ESKI vs YENI ayristirma farki
    diff =[f for f in js if kimlik (f )[1 ]!=kimlik_eski (f )[1 ]]
    empty_ =[f for f in js if not kimlik_eski (f )[1 ]]
    print (f"\nayristirma FARKI which file: {len (diff )}  (bunlarin {len (empty_ )}'i eski yontemde BOS)")

    # --- 2) CAKISMA: two different part same ESKI kimlige dusuyor mu?
    esk =collections .defaultdict (set )
    for f in js :
        e =kimlik_eski (f )[1 ]
        if e :
            esk [e ].add (kimlik (f )[1 ])
    cak ={k :v for k ,v in esk .items ()if len (v )>1 }
    print (f"ESKI kimlikte CAKISMA: {len (cak )} kimlik, {sum (len (v )for v in cak .values ())} gercek part")

    # --- 3) GERCEK RISK: cakisan kimligin STEP'i present mi? (wrong eslesme may be)
    sk_eski =collections .defaultdict (list )
    for s in stp :
        sk_eski [os .path .basename (s ).split ("_")[1 ]].append (s )
    riskli ={k :v for k ,v in cak .items ()if k in sk_eski }
    print (f"** CAKISAN VE STEP'i OLAN (wrong eslesme RISKI): {len (riskli )} kimlik **")
    for k ,v in sorted (riskli .items ())[:10 ]:
        print (f"   '{k }' -> {sorted (v )}  | STEP {len (sk_eski [k ])} file")

        # --- 4) MANIFEST (F2-9)
    man =[]
    for f in js :
        mfg ,pid =kimlik (f )
        try :
            j =json .load (io .open (f ,encoding ="utf-8-sig"))
        except Exception :
            continue 
        cps =j .get ("ConnectionPoints")or []
        pts =(j .get ("Graphic3d")or {}).get ("Points")or []
        n =len (cps )
        man .append ({"file":os .path .basename (f ),"manufacturer":mfg ,"part":pid ,
        "cp":n ,"g3d_tepe":len (pts ),
        "bucket":"1-3"if n <=3 else ("4-7"if n <=7 else "8+"),
        "step":bool (kimlik_eski (f )[1 ]in sk_eski )})
    with io .open ("results/manifest_corpus.json","w",encoding ="utf-8")as fh :
        json .dump (man ,fh ,indent =0 )
    print (f"\nMANIFEST -> results/manifest_corpus.json ({len (man )} part)")

    ok =[m for m in man if m ["cp"]>0 and m ["g3d_tepe"]>0 ]
    print (f"\n{'manufacturer':<9}{'part':>7}{'CP':>8}{'STEP':>7}{'1-3':>6}{'4-7':>6}{'8+':>6}")
    for u ,g in sorted (collections .Counter (m ["manufacturer"]for m in ok ).most_common ()):
        alt =[m for m in ok if m ["manufacturer"]==u ]
        kv =collections .Counter (m ["bucket"]for m in alt )
        print (f"{u :<9}{len (alt ):>7}{sum (m ['cp']for m in alt ):>8}"
        f"{sum (m ['step']for m in alt ):>7}{kv ['1-3']:>6}{kv ['4-7']:>6}{kv ['8+']:>6}")
    print (f"{'TOPLAM':<9}{len (ok ):>7}{sum (m ['cp']for m in ok ):>8}{sum (m ['step']for m in ok ):>7}")


if __name__ =="__main__":
    main ()
