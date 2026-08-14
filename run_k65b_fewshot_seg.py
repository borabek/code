"""K6.5-b: FEW-SHOT SEGMENTASYON FINE-TUNE -- listenin most high beklenen degerli kolu.

SENARYO (kullanicinin real is akisi): gorulmemis a markadan k part gelir, INSAN
onlarin CP'lerini isaretler, sistem adapte becomes, AYNI markanin kalan parcalarinda
olculur. k=0 = bugunku sifir-atis (robot 0.2344).

WHY SEG, WHY GATE DEGIL: gate ayagi measured and TAMAMEN NULL output (3 markada
+-0.005). Sebep gorulmemis markada FN'lerin %76.2'sinin ADAY_YOK olmasi -- gate'e ne
ogretirsen ogret URETILMEMIS adayi geciremez. Adaptasyon TEMSIL katmaninda must be.

ZINCIR:
  1. k parcayi GT CP'lerinden boya      g5_agiz_etiket.py --pids-file
  2. that boyamayla fine-tune              train_seg_extra.py --init-from
  3. olasilik onbellegi                 p1_olasilik_onbellek.py --ckpt <ft> --ek
  4. candidate turet + gate + TAM zincir     this betik

IKI TRAP (ikisi de bilerek ele alindi):
  * OZ-TUTARLILIK KAPISI KAPATILIR (--oz-tut-threshold 0). Kapi, urunun already beceremedigi
    parcalari eler; few-shot'ta this DONGUSELDIR -- full da ogrenmek istedigimiz hard
    parcalari atar. Gercek senaryoda etiketi INSAN koyar, oz-tutarlilik aranmaz.
  * ADAPTASYON PARCASI OLCUME GIRMEZ. Girerse numbers ornekleme-ici becomes.
"""
import argparse 
import json 
import os 
import subprocess 
import sys 

import numpy as np 

PY ="./.venv/Scripts/python.exe"
URUN_CKPT ="results/seg_g10/g10_s0.pt"
CIKTI ="results/k65b_fewshot_seg.json"


def kos (cmd ,log ,ek_env =None ):
    """Alt sureci kos; BASARISIZ OLURSA PATLA (silent devam = fake sonuc)."""
    print (f"  $ {' '.join (cmd )}",flush =True )
    env =dict (os .environ )
    if ek_env :
        env .update (ek_env )
    with open (log ,"w")as f :
        r =subprocess .run (cmd ,stdout =f ,stderr =subprocess .STDOUT ,env =env )
    if r .returncode !=0 :
        raise RuntimeError (f"ADIM BASARISIZ (kod {r .returncode }): {log }")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--brand",default ="SUPU")
    ap .add_argument ("--klar",type =int ,nargs ="+",default =[1 ,3 ,5 ])
    ap .add_argument ("--cekilis",type =int ,default =2 )
    ap .add_argument ("--epochs",type =int ,default =40 )
    ap .add_argument ("--lr",type =float ,default =2e-4 ,
    help ="fine-tune for DUSUK lr; 1e-3 with network onceki bilgisini unutur")
    ap .add_argument ("--kip",choices =["only","tekrar"],default ="tekrar",
    help ="only = SADECE k parcayla adapte (agresif, unutma riski); "
    "tekrar = corpus korunur, k part N kez tekrarlanir (gerceksi)")
    ap .add_argument ("--tekrar",type =int ,default =30 ,
    help ="--kip tekrar for: k part kac kez tekrarlanacak")
    a =ap .parse_args ()

    os .environ .setdefault ("BA_ALLOW_SEEN","1")
    sys .path .insert (0 ,".")
    import d6_record 

    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    pidler =sorted (p for p ,r in rec_ .items ()if r ["mfg"]==a .brand )
    if len (pidler )<max (a .klar )+10 :
        raise SystemExit (f"{a .brand }: yalniz {len (pidler )} part, yetersiz")
    kume_yolu =f"results/_fs_kume_{a .brand }.json"
    if not os .path .exists (kume_yolu ):
        raise SystemExit (f"{kume_yolu } yok -- once brand alt kumesini uret")
    with open (kume_yolu )as _f :
        n_beklenen =len (json .load (_f )["pidler"])
    print (f"=== {a .brand }: {len (pidler )} part | cache beklentisi "
    f"{n_beklenen } ===",flush =True )

    rng =np .random .RandomState (0 )
    res_ ={}
    for k in a .klar :
        res_ [k ]=[]
        for c in range (a .cekilis ):
            label_ =f"{a .brand }_k{k }_c{c }"
            adapt =sorted (rng .choice (pidler ,k ,replace =False ))
            olc =[p for p in pidler if p not in adapt ]
            print (f"\n--- {label_ }: adapt {adapt } | measurement {len (olc )} part ---",
            flush =True )

            # DEVAM EDILEBILIRLIK -- SIKI CHECK.
            # ILK SURUMUM GEVSEKTI (">=10 npz varsa atla") and BAYAT a onbellegi
            # gecerli saydi: that cache (a) `--cluster` eklenmeden before uretilmisti,
            # i.e. 468 parcalik TUM exam kumesini kapsiyordu, (b) "most iyi" ckpt'ten
            # geliyordu, "last"dan not, (c) yarida kesilmisti (327/468).
            # Olcum onunla kosulsaydi SESSIZCE wrong number verirdi.
            # Simdi: part count TAM eslesmeli VE cache ckpt'ten YENI must be.
            _ob =f"results/_p1_olasilik_fs_{label_ }"
            _ck_son =f"results/seg_fs/{label_ }_last.pt"
            _tamam =False 
            if os .path .exists (_ck_son )and os .path .isdir (_ob ):
                _n =len ([x for x in os .listdir (_ob )if x .endswith (".npz")])
                _yeni =os .path .getmtime (_ob )>=os .path .getmtime (_ck_son )
                _tamam =(_n ==n_beklenen )and _yeni 
                if not _tamam :
                    print (f"  YENIDEN KOSULACAK {label_ }: cache {_n }/"
                    f"{n_beklenen } part, ckpt'ten yeni={_yeni }",flush =True )
            if _tamam :
                _nb =len ([x for x in os .listdir (f"results/_fs_boya_{label_ }")
                if x .endswith (".npz")])
                print (f"  ATLANDI (dogrulandi): {label_ }",flush =True )
                res_ [k ].append ({"cekilis":c ,"adapt":adapt ,"n_olc":len (olc ),
                "k_gercek":_nb ,"k_istenen":len (adapt ),
                "ckpt":f"results/seg_fs/{label_ }.pt",
                "ckpt_kullanilan":_ck_son ,"cache":_ob })
                continue 

            pf =f"results/_fs_{label_ }_pids.txt"
            with open (pf ,"w")as f :
                f .write ("\n".join (adapt ))
            boya_dir =f"results/_fs_boya_{label_ }"
            ck =f"results/seg_fs/{label_ }.pt"
            os .makedirs ("results/seg_fs",exist_ok =True )

            # 1) BOYA -- oz-tutarlilik kapisi KAPALI (dongusellik onlemi)
            # SINAV DISLAMASI KAPATILIR (ETIKET_DISLA=""). Guvenli, because:
            #  * only --pids-file'daki k part boyanir (baska no exam parcasi not)
            #  * that k part OLCUMDEN CIKARILIR (`olc` listesinde absent)
            #  * uretilen ckpt ATILIKTIR, urune girmez
            # Senaryo already "this k parcayi sisteme VERIYORUZ" demek; dislama bunu bloke eder.
            kos ([PY ,"-u","g5_agiz_etiket.py","--pids-file",pf ,
            "--oz-tut-threshold","0.0","--cikti",boya_dir ],
            f"results/_fs_{label_ }_boya.log",ek_env ={"ETIKET_DISLA":""})
            # 1b) NPZ -> OBJ+labels.txt. train_seg_extra.load_extra YALNIZ directory
            # bicimini reads; this step atlanirsa "0 part yuklendi" becomes and
            # --only-kismi olmasa training SESSIZCE korpusla kosardi.
            obj_dir =f"results/_fs_obj_{label_ }"
            kos ([PY ,"-u","g5b_etiket_donustur.py","--source",boya_dir ,
            "--hedef",obj_dir ],f"results/_fs_{label_ }_donus.log")
            n_boya =len ([x for x in os .listdir (boya_dir )if x .endswith (".npz")])
            # BOYANAN > ISTENEN olursa SESSIZ SISME demektir -> DUR.
            # BOYANAN < ISTENEN whereas: oto-boyayici INSAN ETIKETININ VEKILI and some
            # parcalarda vekil does not work (GT'nin none of them algilanan a acikliga
            # dusmuyor -> `oz_tutarlilik_dustu`). Insan that parcayi etiketleyebilirdi.
            # Bu yuzden kosumu DURDURMUYORUZ, GERCEK k'yi KAYDEDIYORUZ; olculen
            # egri so real insan etiketine according to a ALT SINIR becomes.
            if n_boya >len (adapt ):
                raise RuntimeError (
                f"{label_ }: {len (adapt )} istendi, {n_boya } boyandi -- SISME, durduruldu.")
            if n_boya ==0 :
                raise RuntimeError (f"{label_ }: hicbir part boyanamadi")
            if n_boya <len (adapt ):
                print (f"  UYARI: {len (adapt )} istendi, {n_boya } boyandi "
                f"(oto-boyayici vekil; gercek k={n_boya })",flush =True )

                # 2) FINE-TUNE
            egit =[PY ,"-u","train_seg_extra.py","--init-from",URUN_CKPT ,
            "--partial-dir",obj_dir ,"--epochs",str (a .epochs ),
            "--lr",str (a .lr ),"--val-partial","--checkpoint-out",ck ]
            egit +=["--only-kismi"]if a .kip =="only"else ["--kismi-tekrar",str (a .tekrar )]
            kos (egit ,f"results/_fs_{label_ }_train.log")

            # 3) OLASILIK ONBELLEGI
            # ONBELLEK YALNIZ BU MARKANIN PARCALARI ICIN: full exam kumesi 468
            # part (~25 dk); measurement already only this markada yapiliyor (~4 dk).
            cluster =f"results/_fs_kume_{a .brand }.json"
            if not os .path .exists (cluster ):
                raise SystemExit (f"{cluster } yok -- once brand alt kumesini uret")
                # SON EPOCH KULLANILIR, "most iyi" DEGIL. Egitim betigi most iyiyi GENEL
                # dogrulamaya according to seciyor; adaptasyon genel dogrulamayi DUSURDUGU for
                # that secim EN AZ ADAPTE OLMUS modeli kaydeder -- i.e. few-shot'i olcmek
                # isterken few-shot'i engeller. Few-shot'ta secim absent: sabit epoch butcesi,
                # last ckpt. (Hedef markadan ayri a secim kumesi ayirmak measurement kumesini
                # kucultecegi for simdilik yapilmiyor; kayit altina alindi.)
            ck_son =ck .replace (".pt","_last.pt")
            if not os .path .exists (ck_son ):
                raise RuntimeError (f"{ck_son } yok -- training son ckpt yazmadi")
            kos ([PY ,"-u","p1_olasilik_onbellek.py","--ckpt",ck_son ,
            "--cluster",cluster ,"--ek",f"_fs_{label_ }"],
            f"results/_fs_{label_ }_p1.log")

            res_ [k ].append ({"cekilis":c ,"adapt":adapt ,"n_olc":len (olc ),
            "k_gercek":n_boya ,"k_istenen":len (adapt ),
            "ckpt":ck ,"ckpt_kullanilan":ck .replace (".pt","_last.pt"),"cache":f"results/_p1_olasilik_fs_{label_ }"})
            print (f"  HAZIR -> {ck }",flush =True )

    with open (CIKTI ,"w")as f :
        json .dump ({"brand":a .brand ,"klar":a .klar ,"cekilis":a .cekilis ,
        "epochs":a .epochs ,"lr":a .lr ,"taban_ckpt":URUN_CKPT ,
        "kosumlar":res_ ,
        "uyari":"k_gercek < k_istenen olabilir: oto-boyayici INSAN "
        "etiketinin VEKILIDIR ve bazi parcalarda GT'nin hicbiri "
        "algilanan a acikliga dusmez. Olculen egri, gercek "
        "insan etiketine according to a ALT SINIRDIR.",
        "not":"Bu betik ADAPTASYON+ONBELLEK uretir. UCTAN UCA OLCUM "
        "ayri adimdir (probe_k65b_olc.py) -- thus training a kez "
        "kosar, measurement tekrar tekrar kosulabilir."},f ,indent =1 )
    print (f"\nmakbuz -> {CIKTI }")
    print ("SIRADAKI: probe_k65b_olc.py (uctan uca measurement, k=0 tabaniyla birlikte)")


if __name__ =="__main__":
    main ()
