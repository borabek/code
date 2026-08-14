# -*- coding: utf-8 -*-
"""D5-2: YENI KORPUSU TURET + GEOMETRI ANAHTARLARINI AYNI GECISTE URET.

DataSet5 with uygun corpus 1926 -> 4720 part / 23 manufacturer became. Gate egitimi for new
parcalarin TURETILMESI is required (network cikarimi + candidate uretimi + X/XR oznitelikleri).

TEK GECIS: turetme already each parcanin STEP'ini yukleyip remesh'liyor; geometri anahtari
(D5-1) AYNI ANDA is computed. Ayri gecis ~1.5 saat tasarruf edilir.

SIRALAMA URETICI-DENGELI: each turda each ureticiden a part alinir. Boylece YARIDA
KESILSE BILE elde 23 ureticiden DENGELI a lower cluster becomes and ARA OLCUM anlamli becomes.
(Alfabetik sirayla yapilsaydi first 800 part only A-B + ABB olurdu.)

DEVAM EDILEBILIR: each 25 parcada diske writes; yeniden calistirilinca kaldigi yerden devam.

TEZ DEGISMEZ: same network (4 checkpoint), same uniform ~6000 remesh, same `v_o` candidate
ureticisi. Degisen single sey KORPUS BUYUKLUGU -- this, "only data" deneyinin sarti.
"""
import argparse 
import collections 
import io 
import json 
import os 
import pickle 
import shutil 
import sys 
import tempfile 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/_der_yeni.pkl"# vardiyada _der_yeni_<k>.pkl becomes
GEO ="results/_geo_yeni.json"
ISARET ="results/_d5_su_an.txt"# this an islenen part -- takilma teshisi for
ATLA_DOSYA ="results/_d5_atla.txt"# kilitlendigi bilinen parts
OPS_TMP ="results/_ops_gecici"# part-yerel operator onbellegi (each parcada silinir)
DISK_ESIK_GB =3.0 # bunun under TEMIZ dur (gmsh sessizce surunmeye baslar)


def dengeli_sira (E ):
    """Uretici-dengeli order: each turda each ureticiden a part."""
    g =collections .defaultdict (list )
    for t in E :
        g [t [0 ]].append (t )
    for k in g :
        g [k ].sort (key =lambda x :x [1 ])
        # SONSUZ DONGU HATASI (2026-08-04, own hatam): before `while any(g.values())` yaziyordum
        # but listelerden eleman CIKARMIYORUM, indeksle geziyorum -> listeler never bosalmiyor,
        # condition hep True kaliyor and `i` tum uzunluklari astiktan after loop SONSUZA up to
        # donuyordu. Turetme parcalara HIC baslamadi, 20 dakika empty dongude CPU yakti; ben de
        # two kosuyu "part kilitlendi" sanip oldurdum. Dogrusu EN UZUN listeye up to donmek.
    enb =max (len (v )for v in g .values ())if g else 0 
    out =[]
    for i in range (enb ):
        for k in sorted (g ):
            if i <len (g [k ]):
                out .append (g [k ][i ])
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--sinir",type =int ,default =0 )
    # VARDIYA (2026-08-04): measured -- 16 cekirdegin only %12'si, GPU %24 is used.
    # Darbogaz spektral operator kurulumu and SERI calisiyor. Is N vardiyaya bolununce
    # neredeyse dogrusal hizlaniyor: 13 saat -> ~3.5 saat (4 vardiya, 8 cekirdek, %50 pay durur).
    # CKPT OVERRIDE (G6): new seg agiyla YENIDEN turetme for. Gate'in new candidate
    # dagilimiyla egitilmesi SART -- old gate'e new adaylari vermek [[gate-refit-minv4]]
    # dersinin ihlali becomes (two gate AYNI dagilimda egitilmeli).
    ap .add_argument ("--ckpt",nargs ="+",default =[],help ="cp_config yerine bu ckpt'leri kullan")
    # G6: TUM corpus YENI agla yeniden turetilir -- old turetme "already present" SAYILMAZ,
    # because old candidates old agin ciktisi. Karistirmak gate'i two different dagilimla egitir.
    ap .add_argument ("--hepsi",action ="store_true",help ="eski turetmeyi 'var' sayma")
    # PID FILTRESI (R4a): TOPLULUK olcumu for TUM korpusu turetmeye gerek absent --
    # only two OLCUM kumesi (250 exam + 194) yeter, ~444 part.
    ap .add_argument ("--pids-file",default ="",help ="yalniz bu dosyadaki pid'leri turet")
    ap .add_argument ("--cikti-eki",default ="",help ="cikti dosya adina ek (G6 icin _g6)")
    ap .add_argument ("--vardiya",type =int ,default =0 )
    ap .add_argument ("--toplam",type =int ,default =1 )
    a =ap .parse_args ()

    import protocol 
    protocol .tez_dogrula ()
    import torch 
    import cad_eval 
    import diffusionnet as D_ 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from build_zengin_parite import _normaller ,zengin 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from geometri_anahtar import anahtar 
    from infer_step_cp import load_any ,step_to_mesh 

    os .makedirs (OPS_TMP ,exist_ok =True )
    for _e in os .listdir (OPS_TMP ):# onceki kosudan residual kalmasin
        shutil .rmtree (os .path .join (OPS_TMP ,_e ),ignore_errors =True )
    global CIKTI ,GEO ,ISARET 
    ek =a .cikti_eki 
    if a .toplam >1 :
        CIKTI =f"results/_der_yeni{ek }_{a .vardiya }.pkl"
        GEO =f"results/_geo_yeni{ek }_{a .vardiya }.json"
        ISARET =f"results/_d5_su_an{ek }_{a .vardiya }.txt"
    elif ek :
        CIKTI =f"results/_der_yeni{ek }.pkl";GEO =f"results/_geo_yeni{ek }.json"
    ATLA =set ()
    if os .path .exists (ATLA_DOSYA ):
        ATLA ={x .strip ()for x in io .open (ATLA_DOSYA ,encoding ="utf-8")if x .strip ()}
        print (f"atlanacak (kilitlenen) part: {len (ATLA )}",flush =True )
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    E =eligible ()
    # already turetilmis olanlar (gate korpusu + measurement kumesi)
    var =set ()
    if not a .hepsi :
        d =np .load ("results/zengin_parite_w2.npz",allow_pickle =True )
        var |={str (x )for x in d ["pids"]}
        for y in ("results/_der_tam.pkl","results/_der_kontrol.pkl"):
            if os .path .exists (y ):
                with open (y ,"rb")as f :
                    var |={r ["pid"]for r in pickle .load (f )}
                    # DEVAM ETMENIN BOLME HATASI (2026-08-04, own hatam):
                    # Her vardiya YALNIZ KENDI pkl'ini "already present" sayiyordu. Devam edildiginde four
                    # vardiyanin `hedef` listeleri FARKLILASIYOR and `i % 4 == k` residual AYNI listeyi
                    # bolmuyor: some parts IKI KEZ turetiliyor, bazilari HIC turetilmiyor.
                    # (Belirti: hedef toplami 2484, oysa 2690-823 = 1867 olmaliydi.)
                    # Dogrusu: bolmeyi belirleyen `present` kumesi TUM vardiyalarin ciktisini icermeli;
                    # `OUT` whereas only this vardiyanin kayitlarini tutmaya devam eder.
    for _k in range (a .toplam if a .toplam >1 else 1 ):
        _f =f"results/_der_yeni{ek }_{_k }.pkl"if a .toplam >1 else CIKTI 
        if os .path .exists (_f ):
            with open (_f ,"rb")as _h :
                var |={r ["pid"]for r in pickle .load (_h )}
    OUT ,GEOD =[],{}
    if os .path .exists (CIKTI ):
        with open (CIKTI ,"rb")as f :
            OUT =pickle .load (f )
        var |={r ["pid"]for r in OUT }
        if os .path .exists (GEO ):
            GEOD =json .load (io .open (GEO ,encoding ="utf-8"))
        print (f"devam: {len (OUT )} part zaten turetilmis",flush =True )

    if a .pids_file :
        _sec ={x .strip ()for x in io .open (a .pids_file ,encoding ="utf-8")if x .strip ()}
        E =[t for t in E if t [1 ]in _sec ]
        print (f"PID FILTRESI: {len (_sec )} istendi -> {len (E )} bulundu",flush =True )
    hedef =dengeli_sira ([t for t in E if t [1 ]not in var ])
    if a .toplam >1 :
        hedef =[t for i ,t in enumerate (hedef )if i %a .toplam ==a .vardiya ]
        print (f"VARDIYA {a .vardiya }/{a .toplam }",flush =True )
    if a .sinir :
        hedef =hedef [:a .sinir ]
    print (f"uygun corpus {len (E )} | zaten var {len (var )} | TURETILECEK {len (hedef )}")
    print ("  ilk 12 (manufacturer-dengeli):",[t [0 ]for t in hedef [:12 ]],flush =True )

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =a .ckpt or cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    print (f"  ckpt: {[c .split ('/')[-1 ]for c in cks ]}",flush =True )
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    t0 =time .time ();error =0 
    for k ,(mfg ,pid ,jf ,stp )in enumerate (hedef ,1 ):
        if k %5 ==0 :
            hiz =(time .time ()-t0 )/k 
            kalan =hiz *(len (hedef )-k )/60 
            print (f"  {k }/{len (hedef )}  {time .time ()-t0 :.0f}s  error={error }  "
            f"({hiz :.1f}s/part, kalan ~{kalan :.0f} dk)",flush =True )
            with open (CIKTI ,"wb")as f :
                pickle .dump (OUT ,f )
            with io .open (GEO ,"w",encoding ="utf-8")as f :
                json .dump (GEOD ,f )
                # DISK BEKCISI (2026-08-05, IKINCI times vurdu): disk dolunca gmsh HATA VERMEZ,
                # 9s/part -> 6-12 DK/part'ya duser ([[disk-dolunca-gmsh-donuyor]]). Belirti
                # "yavasladi" oldugu for saatler suclunun pesinde harcaniyor. Burada TEMIZ
                # duruyoruz: kayitlar already yazildi, yeniden calistirinca kaldigi yerden devam.
            _bos =shutil .disk_usage (".").free /1e9 
            if _bos <DISK_ESIK_GB :
                print (f"\n** DISK {_bos :.1f} GB < {DISK_ESIK_GB } GB -- TEMIZ DURULUYOR. "
                f"{len (OUT )} kayit yazildi; yer acip yeniden baslatin. **",flush =True )
                break 
                # HANGI PARCADA TAKILDI: each parcadan ONCE imza atilir. 2026-08-04'te turetme
                # first 5 parcanin birinde asili kaldi and 5'te-a yazdigim for SUCLUYU bulamadim
                # (hafizada kayitli tuzak: some parts turetmeyi kilitliyor). Artik imza present.
        with io .open (ISARET ,"w",encoding ="utf-8")as _f :
            _f .write (f"{k }	{mfg }	{pid }	{stp }")
        if pid in ATLA :
            print (f"    {pid }: ATLANDI (kilitlenen part listesinde)",flush =True )
            continue 
        try :
            Vr ,Fr =step_to_mesh (stp )
            GEOD [pid ]=anahtar (Vr ,Fr )# D5-1 same gecisde
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            # OPERATOR ONBELLEGI PARCA-YEREL VE GECICI (2026-08-04, DISK ACIL DURUMU):
            # Eskiden paylasilan `results/step_infer/ops_k*` dizinine yaziyordu. Ama this
            # gecis each parcayi BIR KEZ turetiyor; cache a more HIC okunmuyor, buna
            # karsilik part basina ~9 MB disk yiyordu. 2690 part = ~25 GB, and diskte
            # 0.9 GB kalmisti -- vardiyalar that is why surunuyordu.
            # Parca-yerel gecici directory: AYNI parcada same k_eig'i paylasan models
            # operatoru yine yeniden kullanir (4 model -> 2 hesap), part bitince silinir.
            tmp =tempfile .mkdtemp (prefix ="ops_",dir =OPS_TMP )
            try :
                pbs =[]
                for model ,meta in models :
                    _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                    op_cache_dir =os .path .join (
                    tmp ,f"k{int (meta .get ('k_eig',64 ))}"),
                    return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
            finally :
                shutil .rmtree (tmp ,ignore_errors =True )
            cps ,probs ,_ ,uyeler =RC .derive_candidates (V ,F ,pbs ,stp ,cfg =cfg )
            if not cps :
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ));X =None ;XR =None 
            else :
                P =np .array ([c ["point"]for c in cps ],float )
                Pd =np .array ([c ["direction"]for c in cps ],float )
                X =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp )
                try :
                    XR =zengin (V ,F ,probs ,cps ,_normaller (V ,F ))
                except Exception :
                    XR =None 
            j =json .load (io .open (jf ,encoding ="utf-8-sig"))
            g =j .get ("ConnectionPoints")or []
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in g ],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in g ],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            R ,t_ ,_ =cad_eval .align_frames (Vr ,Vj )
            G =(G -t_ )@R ;Gd =Gd @R 
            OUT .append ({"pid":pid ,"mfg":mfg ,"geo":GEOD [pid ],"cluster":None ,
            "diag":float (np .linalg .norm (V .max (0 )-V .min (0 ))),
            "n":len (G ),"P":P ,"Pd":Pd ,"X":X ,"XR":XR ,
            "G":G ,"Gd":Gd ,"UYE":uyeler })
        except Exception as e :
            error +=1 
            if error <=5 :
                print (f"    {pid }: {type (e ).__name__ }: {str (e )[:70 ]}",flush =True )
    with open (CIKTI ,"wb")as f :
        pickle .dump (OUT ,f )
    with io .open (GEO ,"w",encoding ="utf-8")as f :
        json .dump (GEOD ,f )
    c =collections .Counter (r ["mfg"]for r in OUT )
    print (f"\n{len (OUT )} kayit -> {CIKTI } | error {error } | {time .time ()-t0 :.0f}s")
    print (f"  geometri anahtari -> {GEO } ({len (GEOD )} part)")
    print (f"  manufacturer: {dict (c .most_common (12 ))}")
    print (f"  GT toplam {sum (r ['n']for r in OUT )} | candidate {sum (len (r ['P'])for r in OUT )}")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
