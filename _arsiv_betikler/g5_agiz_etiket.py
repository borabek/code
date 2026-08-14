# -*- coding: utf-8 -*-
"""G5-a: AGIZ-CEVRIMI OTO-ETIKET -- seg korpusunu 71'den ~1500'e cikar.

WHY (G1 olcumu, 2026-08-04): gorulmemis manufacturer sinavinda FN'lerin **%68.1'i ADAY_YOK**
-- i.e. candidate HIC uretilmemis. Kotu ureticilerde this pay %76.2, iyi olanlarda %57.6; gate
reddi whereas two yarida da AYNI (%12.9 vs %14.1). Yani ureticiler arasi yayilimin sebebi gate
DEGIL TEMSIL. Tavan hesabi: gate and cozunurluk MUKEMMEL olsa (sifir FP!) bile F1 0.7038 --
0.70 for ADAY_YOK kovasindan pay alinmak ZORUNDA. Ona dokunan single arm budur.

ROOT CAUSE: network 71 parcayla egitildi (`EXPECTED_COUNTS = {'train': 71, ...}`), turetme
korpusu 4405. Bu betik manufacturer GT'sinden zayif label produces.

h3 WHY COKTU VE BU WHY FARKLI ([[h3-highcp-finetune-dead]]):
  h3, CP'nin cevresine **r=2mm SABIT DISK** boyadi. Disk body yuzeyini de boyuyordu ->
  network "duz surface de CableEntry" ogrendi -> atesleme %78'den %24'e COKTU.
  Burada two difference present:
    1. YARICAP OLCULUYOR, sabit not: `mouth_width` (A0'da kalibre edildi, %1 deviation)
       each CP'de acikligin GERCEK genisligini gives.
    2. BOYANAN SEY KANAL, duz surface not: axis along [-1mm, +depth] bandinda and
       eksene dik mesafesi olculen yaricapin under kalan vertices.

OZ-TUTARLILIK KAPISI (asil koruma): label uretildikten after URUNUN KENDI
`cp_openings.connection_points` turetmesi that etikete uygulanir. Cikan `v_o` GT CP'sine
toleransta oturmuyorsa **label ATILIR**. Yani "agi wrong seye egiten" label korpusa
giremez -- h3'un cokusu this kapidan gecemezdi.

TEZ DEGISMEZ: 5 sinif same, `v_o` turetmesi same, ~6000 uniform remesh same. Degisen TEK
sey ETIKETLI PARCA SAYISI.
"""
import argparse 
import collections 
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI_VARSAYILAN ="_label_auto"# each part: <pid>.npz (V, F, labels)
MAKBUZ ="results/g5_agiz_etiket.json"

# BOYAMA DERINLIGI (2026-08-07'de MEASURED and DUZELTILDI).
# Eski degerler 1.0/6.0 idi. 6mm channel ICINE boyamak, urunun turetmesinin mouth
# merkezini TUPUN DERININE kaydirmasina path aciyordu; small parcalarda tolerans 3mm
# oldugu for oz-tutarlilik kapisi that parcalari ELIYORDU. Reddedilen 29 parcada
# taranarak measured: 1.0/6.0 -> 0/29 gecer, 0.5/2.0 -> **8/29** gecer.
# Yani gate etiketin KALITESINI not boyamanin DERINLIGINI cezalandiriyordu.
# Ayrica more sig boyama TEZE DAHA SADIK: tezin `v_o`'this AGIZ SINIRI merkezidir,
# channel ici not. Eski value `_ESKI_DERINLIK` under duruyor.
_ESKI_DERINLIK =(1.0 ,6.0 )
EKSEN_ONCE =0.5 # agzin DISINDA boyanacak band (mm) -- cevrim/rim
EKSEN_SONRA =2.0 # channel ICINE correct boyanacak depth (mm)
YARICAP_PAYI =1.15 # olculen yaricapa emniyet payi


def agiz_etiketle (V ,F ,mesh ,G ,Gd ,CE ,HOUSING =0 ):
    """GT CP'lerinden vertex etiketi uret. Doner: (labels, bilgi)."""
    from cp_geometry import mouth_width ,seat_to_mouth 
    lab =np .full (len (V ),HOUSING ,np .int64 )
    bilgi =[]
    for p ,d in zip (G ,Gd ):
        nd =float (np .linalg .norm (d ))
        if nd <1e-9 :
            bilgi .append (None );continue 
        d =d /nd 
        # seat_to_mouth (agiz_noktasi, isaretli_offset_mm) DONDURUR -- single point not.
        # Ilk surumde dogrudan np.asarray'e verildi and 30/30 parcada
        # "setting an array element with a sequence" with dustu.
        try :
            m ,_off =seat_to_mouth (mesh ,p ,d )
        except Exception :
            m =p 
        m =np .asarray (m ,float ).reshape (3 )
        try :
            ic ,ort =mouth_width (mesh ,m ,d )
        except Exception :
            ic =ort =0.0 
        r =0.5 *float (ort if ort >0 else ic )
        if r <=1e-6 :
            bilgi .append (None );continue # olculemedi -> UYDURMA, atla
        rel =V -m 
        eks =rel @d # axis along konum
        dik =np .linalg .norm (rel -eks [:,None ]*d [None ,:],axis =1 )
        msk =(eks >=-EKSEN_ONCE )&(eks <=EKSEN_SONRA )&(dik <=r *YARICAP_PAYI )
        lab [msk ]=CE 
        bilgi .append ({"mouth":m .tolist (),"r":r ,"n_tepe":int (msk .sum ())})
    return lab ,bilgi 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sinir",type =int ,default =0 )
    # VARDIYA: turetmedeki AYNI desen. Betik already devam edebilir (present which is .npz atlanir)
    # but vardiyasiz four kopya AYNI parcadan baslar and is bosa gider.
    ap .add_argument ("--vardiya",type =int ,default =0 )
    ap .add_argument ("--toplam",type =int ,default =1 )
    ap .add_argument ("--pids-file",default ="",
    help ="only this dosyadaki pid'leri etiketle (kontrollu parti for)")
    ap .add_argument ("--oz-tut-threshold",type =float ,default =0.6 ,
    help ="oz-tutarlilik kapisi: GT'nin bu orani geri gelmeli (0.6 varsayilan)")
    ap .add_argument ("--tol-carpan",type =float ,default =1.0 ,
    help ="oz-tutarlilik kapisi tolerans carpani")
    # FEW-SHOT (K6.5-b): each kosum KENDI k parcasini AYRI dizine boyamali. Sabit
    # `_label_auto` kullanilirsa onceki kosumlarin dosyalari birikir and fine-tune
    # k part instead of YUZLERCE parcayla egitilir -- error vermez, sayilari sisirir.
    ap .add_argument ("--cikti",default =CIKTI_VARSAYILAN ,
    help ="boyama cikti dizini (varsayilan _label_auto)")
    a =ap .parse_args ()

    import protocol 
    protocol .tez_dogrula ()
    import trimesh 
    import thesis_remesh 
    import cp_openings 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE 
    from infer_step_cp import step_to_mesh 
    import cad_eval 

    CIKTI =a .out_ 
    os .makedirs (CIKTI ,exist_ok =True )
    var ={os .path .splitext (x )[0 ]for x in os .listdir (CIKTI )if x .endswith (".npz")}
    E =[t for t in eligible ()if t [1 ]not in var ]
    # SINAV DISLAMASI -- PARCA VE URETICI DUZEYINDE (2026-08-05'te bulunan kirlilik).
    # Bu betik `eligible()` with calisiyordu and exam kumelerini HIC dislamiyordu. Sonuc:
    # `_label_auto_obj` old exam kumesinin 250 parcasindan **169'unu** and that ureticilerden
    # 958 parcayi iceriyordu; seg agi sinavin ucte ikisini EGITIMDE gormustu and orada
    # olculen "gorulmemis manufacturer" sayilari SISIKTI. Kaynak buydu, here kapatiliyor.
    import io as _io 
    _SP ,_SM =set (),set ()
    # HANGI SINAVLAR DISLANACAK -- ayarlanabilir (2026-08-06).
    # WHY: two sinavi BIRDEN dislamak label korpusunu 8 ureticiye dusurdu and
    # g6 agi, g5'ten KOTU output (D6 candidate kahini 0.8808 -> 0.8101, candidate %33 few).
    # [[cesitlilik-and-fn-profili]]: same N'de KARISIK data single ureticiyi yener.
    # Dogru tasarim SINAV BASINA BIR AG: a sinavi olcerken only O sinavin
    # ureticileri dislanir, digerleri egitime GIRER and cesitliligi korur.
    _sinavlar =tuple (x for x in os .environ .get (
    "ETIKET_DISLA",
    "results/d5_4_sinav_kumesi.json,results/d6_sinav_kumesi.json").split (",")if x )
    for _sf in _sinavlar :
        if os .path .exists (_sf ):
            with _io .open (_sf ,encoding ="utf-8")as _f :
                _sv =json .load (_f )
            _SP |=set (_sv .get ("pidler")or [])
            _SM |=set (_sv .get ("manufacturer")or {})
    _once =len (E )
    E =[t for t in E if t [1 ]not in _SP and t [0 ]not in _SM ]
    if a .pids_file :
        _sec ={x .strip ()for x in _io .open (a .pids_file ,encoding ="utf-8")if x .strip ()}
        E =[t for t in E if t [1 ]in _sec ]
        print (f"PID FILTRESI: {len (_sec )} istendi -> {len (E )} bulundu",flush =True )
    print (f"SINAV DISLAMASI: {_once } -> {len (E )} part "
    f"({len (_SM )} manufacturer, {len (_SP )} part disarida)",flush =True )

    # URETICI-DENGELI SIRA: yarida kesilse bile corpus dengeli olsun (turetmedeki same fikir)
    from d5_turet_yeni import dengeli_sira 
    E =dengeli_sira (E )
    if a .total_ >1 :
        E =[t for i ,t in enumerate (E )if i %a .total_ ==a .vardiya ]
        print (f"VARDIYA {a .vardiya }/{a .total_ }",flush =True )
    if a .bound_ :
        E =E [:a .bound_ ]
    print (f"etiketlenecek {len (E )} part (zaten var {len (var )})",flush =True )

    say =collections .Counter ();t0 =time .time ();UR =collections .Counter ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (E ,1 ):
        if k %25 ==0 :
            h =(time .time ()-t0 )/k 
            print (f"  {k }/{len (E )}  {h :.1f}s/part  kalan ~{h *(len (E )-k )/60 :.0f} dk  "
            f"{dict (say )}",flush =True )
        try :
            j =json .load (io .open (jf ,encoding ="utf-8-sig"))
            cps =j .get ("ConnectionPoints")or []
            G0 =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in cps ],float )
            D0 =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in cps ],float )
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (stp )
            R ,t_ ,res =cad_eval .align_frames (Vr ,Vj )
            if res >2.0 :
                say ["hizalama_zayif"]+=1 ;continue # GT guvenilmez -> etiketleme
            G =(G0 -t_ )@R ;Gd =D0 @R 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            mesh =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
            lab ,bilgi =agiz_etiketle (V ,F ,mesh ,G ,Gd ,int (CE ))
            if not (lab ==int (CE )).any ():
                say ["etiket_bos"]+=1 ;continue 

                # --- OZ-TUTARLILIK KAPISI: urunun own turetmesi GT'yi geri veriyor mu?
            cp =cp_openings .connection_points (V ,F ,lab ,min_v =1 )
            if not cp :
                say ["vo_uretmedi"]+=1 ;continue 
            P =np .array ([c ["point"]for c in cp ],float )
            tol =a .tol_carpan *max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            d =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =-1 )
            kapsanan =int ((d .min (0 )<=tol ).sum ())
            # OZ-TUTARLILIK ESIGI ARTIK PARAMETRE (2026-08-06).
            # MEASURED (59 very-CP parcasi): medyan geri-kazanim **%33** and %60'lik kapidan
            # **0/59** geciyor. Yani network bugune up to TEK BIR etiketli very-CP parcasi
            # gormedi -- very-CP robotunun 0.0481 olmasinin dogrudan aciklamasi budur.
            #
            # WHY GEVSETMEK GUVENLI: labels ZATEN GT'den boyaniyor (`agiz_etiketle`
            # each GT CP'sini olculen yaricapla boyar). Kapi, etiketin dogrulugunu not
            # URUNUN KENDI TURETMESININ onlari geri bulup bulmadigini olcer. Yani gate,
            # urunun zayifligini olcup TAM DA O ZAYIFLIGI duzeltecek veriyi atiyordu --
            # dongusel. h3 cokusu farkliydi: orada ETIKETIN KENDISI yanlisti (sabit 2mm
            # disk duz yuzeyi boyuyordu), here label GT'den geliyor.
            #
            # Yine de BAGLAYICI KILL gecerli: atesleme %78 altina duserse ya da
            # gorulmemis F1 gerilerse GERI ALINIR.
            if kapsanan <max (1 ,int (a .oz_tut_esik *len (G ))):
                say ["oz_tutarlilik_dustu"]+=1 ;continue 

            np .savez_compressed (os .path .join (CIKTI ,pid +".npz"),
            V =V .astype (np .float32 ),F =F .astype (np .int32 ),
            labels =lab .astype (np .int8 ))
            say ["yazildi"]+=1 ;UR [mfg ]+=1 
        except Exception as e :
            say ["error"]+=1 
            if say ["error"]<=5 :
                print (f"    {pid }: {type (e ).__name__ }: {str (e )[:60 ]}",flush =True )

    print (f"\n{dict (say )}  |  {time .time ()-t0 :.0f}s")
    print (f"  manufacturer: {dict (UR .most_common (15 ))}")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"say":dict (say ),"manufacturer":dict (UR ),
        "eksen_once":EKSEN_ONCE ,"eksen_sonra":EKSEN_SONRA ,
        "yaricap_payi":YARICAP_PAYI },f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
