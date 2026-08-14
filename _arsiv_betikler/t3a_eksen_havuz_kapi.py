# -*- coding: utf-8 -*-
"""T3a: EKSEN-FARKINDALIKLI HAVUZLAMA -- PAHALI ISTEN ONCEKI UCUZ KAPI.

WHY BU KOL: `votes`, gate'in GENELLESEN TEK ozelligi (manufacturer-disi AUC dususu 0.018;
digerleri 0.12-0.18 -- [[gate-memorizes-not-learns]]). Ve `r1_oy_parcalanmasi` olctu:
manufacturer CP'lerinin **%52'sinde** uye noktalari 5mm'den genis yayiliyor, i.e. same acikligi
bulan uyeler AYRI adaylara bolunuyor and genellesen single sinyal BOZULUYOR.

MESAFEYLE COZULEMEZ: this gece olctum, GT'lerin most-yakin-komsu mesafesi medyan 5.15mm and
**%66'sinin 8mm inside komsusu present**. Parcalanma yayilimi (medyan 5.27mm) with real CP
ADIMI AYNI OLCEKTE. Kumeleme yaricapini buyutmek "same acikligi two uye gordu" with "two
komsu CP" ayrimini imkansiz kilar. Cozum radius DEGIL, AYNI ACIKLIGA AIT OLMA must be:
axis-farkindalikli havuzlama (dik distance <= 3mm VE eksenler <=20 derece hizali, DERINLIK
serbest).

WHY YENIDEN OLCULUYOR: this bayrak more before was tried and elendi -- but 8 UYELI havuza
GECISLE BIRLIKTE (G3). Iki degisiklik same kosuda oldugu for axis havuzunun KENDI etkisi
never ayrisdirilmadi. Dagitilan 4 uyeli toplulukla YALNIZ BASINA never olculmedi.

BU BETIK PAHALI ISI YAPMAZ. Adil a uctan uca measurement, EGITIM korpusunun da new dagilimla
yeniden turetilmesini gerektirir (~2-4 saat GPU; aksi halde gate with candidate dagilimi
uyusmaz -- G3'te full this error yapilmisti). Once ucuz gate:

    (1) candidate havuzu GT'nin ne kadarini iceriyor (recall) -- DUSMEMELI
    (2) GT basina OY count -- YUKSELMELI (kolun tum gerekcesi this)
    (3) candidate count -- asiri dusmemeli

KAPI: oy/GT >= +0.15 VE recall dususu <= 0.005. Gecerse pahali is HAK EDILIR.

cp_config.json GECICI as yamalanir and `finally` with GERI ALINIR (SHA with dogrulanir).
"""
import hashlib ,io ,json ,os ,shutil ,subprocess ,sys ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CFG ="cp_config.json"
YEDEK ="cp_config.json.t3a_yedek"
CIKTI ="results/_der_eksen.pkl"


def sha (p ):
    with open (p ,"rb")as f :
        return hashlib .sha256 (f .read ()).hexdigest ()[:16 ]


def istat (DER ):
    """(aday_recall, oy_per_GT, aday_sayisi) -- GT'ye TESPIT toleransiyla bakar."""
    ul =0 ;n =0 ;na =0 
    for r in DER :
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        n +=len (G )
        P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
        na +=len (P )
        if len (P )and len (G ):
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            ul +=int ((pe .min (0 )<=max (3.0 ,0.06 *float (r ["diag"]))).sum ())
    return ul /max (n ,1 ),na ,n 


OY_SUTUN =11 # FEAT_NAMES_13 = [..., "aspect", "flat", "chan_conn", "votes", "conf"]


def oy_istat (DER ):
    """GT with eslesen adaylarin ORTALAMA oy count -- URUNUN KENDI `_votes` degeri.

    NOTE (first surumde tuzaga dusuyordum): oyu "birlesmis noktanin 5mm kuresi icindeki
    uye noktalarini say" diye hesaplamak this deneyi YANLI yapar. Eksen-farkindalikli
    havuzlama DERINLIK farkini bilerek serbest birakir; that uyeler 5mm kurenin DISINDA
    kalir and new arm own kazandigi oylari KAYBETMIS gorunurdu. `wire_gate.feats_for`
    oyu X'in 11. sutununa yaziyor (`float(c.get("_votes", 1))`) -- two turetme de
    same sutunu tasiyor, karsilastirma so elmayla elma becomes.
    """
    oy =[]
    for r in DER :
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
        X =r .get ("X")
        if not len (P )or not len (G )or X is None or np .asarray (X ).shape [1 ]<=OY_SUTUN :
            continue 
        X =np .asarray (X ,float )
        for b in range (len (G )):
            d =P -G [b ]
            a =d @Gd [b ]
            pe =np .linalg .norm (d -a [:,None ]*Gd [b ],axis =1 )
            pe =np .where (np .abs (a )>40 ,np .inf ,pe )
            if not np .isfinite (pe ).any ()or pe .min ()>max (3.0 ,0.06 *float (r ["diag"])):
                continue 
            oy .append (float (X [int (np .argmin (pe )),OY_SUTUN ]))
    return float (np .mean (oy ))if oy else 0.0 ,len (oy )


def main ():
    import pickle 
    import measure_set 
    # CHECK, 2 GUNLUK ONBELLEK DEGIL, AYNI GECE AYNI ORTAMDA URETILMIS TURETMEDIR.
    # Sebep: `diffusion_net` this gece yeniden kuruldu and 8 parcalik dogrulamada konumlarin
    # process-arasi ~0.4mm'ye up to oynadigi measured ([[robot-nondeterminism]] pymeshlab).
    # Taze a "axis havuzu OPEN" turetmesini 2 gun onceki onbellekle karsilastirmak,
    # havuzlama etkisini that oynamayla and aradaki kod kaymasiyla KARISTIRIRDI.
    KONTROL ="results/_der_kontrol.pkl"
    assert os .path .exists (KONTROL ),f"{KONTROL } yok -- once bayrak KAPALI turetme kosmali"
    ESKI ,rap =measure_set .cluster (KONTROL )
    measure_set .rapor_bas (rap )
    try :
        _E2 ,_ =measure_set .cluster ("results/_der_tam.pkl")
        _r2 ,_a2 ,_g2 =istat (_E2 )
        print (f"[kayma bilgisi] 2 gunluk cache: recall {_r2 :.4f} / {_a2 } candidate")
    except Exception :
        pass 

    if not os .path .exists (CIKTI ):
        s0 =sha (CFG )
        shutil .copy2 (CFG ,YEDEK )
        try :
            c =json .load (io .open (CFG ,encoding ="utf-8"))
            print (f"mevcut robot_eksen_havuz = {c .get ('robot_eksen_havuz')}")
            c ["robot_eksen_havuz"]=True 
            c ["_t3a_gecici"]=("EKSEN HAVUZ DENEYI -- this anahtar dosyada goruyorsan "
            "t3a yarida kesilmis demektir, cp_config.json.t3a_yedek'ten GERI AL")
            with io .open (CFG ,"w",encoding ="utf-8")as f :
                json .dump (c ,f ,indent =1 ,ensure_ascii =False )
            print ("cp_config YAMALANDI (robot_eksen_havuz=True); turetme basliyor...",flush =True )
            t0 =time .time ()
            r =subprocess .run ([sys .executable ,"turet.py","--output",CIKTI ],
            capture_output =True ,text =True )
            print (r .stdout [-2500 :])
            if r .returncode !=0 :
                print ("TURETME HATASI:",r .stderr [-1500 :])
            print (f"turetme {time .time ()-t0 :.0f}s",flush =True )
        finally :
            shutil .copy2 (YEDEK ,CFG )
            os .remove (YEDEK )
            s1 =sha (CFG )
            print (f"cp_config GERI ALINDI  sha {s0 } -> {s1 }  "
            f"{'AYNI (dogrulandi)'if s0 ==s1 else '!!! FARKLI -- ELLE CHECK ET'}")
            assert s0 ==s1 ,"cp_config geri alinamadi"
    else :
        print (f"{CIKTI } zaten var, turetme atlandi")

    with open (CIKTI ,"rb")as f :
        YENI_HAM =pickle .load (f )
    YENI ,rap2 =measure_set .cluster (CIKTI )
    print (f"\nolcum kumesi: eski {len (ESKI )} part | yeni {len (YENI )} part")
    r0 ,na0 ,ng0 =istat (ESKI );r1 ,na1 ,ng1 =istat (YENI )
    o0 ,n0 =oy_istat (ESKI );o1 ,n1 =oy_istat (YENI )
    print (f"\n{'':<22}{'ESKI (5mm kure)':>18}{'YENI (axis)':>15}{'difference':>10}")
    print (f"{'candidate recall':<22}{r0 :>18.4f}{r1 :>15.4f}{r1 -r0 :>+10.4f}")
    print (f"{'candidate count':<22}{na0 :>18}{na1 :>15}{na1 -na0 :>+10}")
    print (f"{'oy / eslesen GT':<22}{o0 :>18.3f}{o1 :>15.3f}{o1 -o0 :>+10.3f}")
    print (f"{'  (n)':<22}{n0 :>18}{n1 :>15}")
    gate =(o1 -o0 )>=0.15 and (r0 -r1 )<=0.005 
    print (f"\nKAPI: oy/GT >= +0.15 VE recall dususu <= 0.005 -> "
    f"{'GECTI -- pahali gate yeniden-turetmesi HAK EDILDI'if gate else 'GECMEDI -- arm kapanir, pahali is YAPILMAZ'}")
    with io .open ("results/t3a_eksen_havuz_kapi.json","w",encoding ="utf-8")as f :
        json .dump ({"old":{"recall":r0 ,"candidate":na0 ,"oy":o0 ,"n_oy":n0 ,"gt":ng0 },
        "new":{"recall":r1 ,"candidate":na1 ,"oy":o1 ,"n_oy":n1 ,"gt":ng1 },
        "d_oy":o1 -o0 ,"d_recall":r1 -r0 ,"gate":bool (gate )},f ,indent =1 )
    print ("receipt -> results/t3a_eksen_havuz_kapi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
