# -*- coding: utf-8 -*-
"""PROTOKOL: egitimde NE KULLANILAMAZ -- single source, zorunlu guard (F0-1).

WHY SEPARATE MODUL: `measure_set.sinav_egitim_maskesi()` already vardi but ISTEGE BAGLIYDI --
cagiran unutursa no sey uyarmiyordu. 2026-08-03'te full this became: dagitilan training
korpusu 95 grup-temiz LOCKED parcasinin 77'sini icermis, and this however elle arandiginda
was found. Olculdu (2026-08-04, this modul yazilirken):

    dagitilan `zengin_parite_w2.npz` : 1709 part
      LOCKED grubuna ait            :  128 part  <- IHLAL
      OLCUM kumesi grubuna ait      :  391 part

IKI AYRI YASAK VAR, KARISTIRILMAMALI:
  * LOCKED (single-atislik exam)  : HER egitimde yasak. Ihlal edilirse exam hakki YANAR.
  * OLCUM kumesi (194 part)    : that cluster on OLCULECEK gate'te yasak. F0-1'e according to
    194'un tamami "development"tir -- i.e. AYAR/VERDICT bolmelerinde kullanilabilir, but
    on puanlanacak modelin egitimine giremez.

YASAK PARCA DEGIL **GEOMETRI GRUBU** duzeyindedir: parcalarin %80'inin korpusta ikizi present
([[geometry-twin-leakage]]). Parca kimligiyle diskalamak ikizi disarida birakmaz.
"""
import io 
import json 

import numpy as np 

import measure_set as OK 

MAKBUZ ="results/protokol_dogrulama.json"


def yasak_gruplar (olcum_da =False ,der_yolu ="results/_der_tam.pkl"):
    """Egitimde kullanilamayacak GEOMETRI GRUPLARI."""
    g =set (OK .locked_gruplari ())
    if olcum_da :
        D ,_ =OK .cluster (der_yolu )
        g |={r ["geo"]for r in D }
    return g 


def egitim_maskesi (pidler ,olcum_da =False ,der_yolu ="results/_der_tam.pkl"):
    """Egitimde KULLANILABILIR olanlar True. Grup duzeyinde works."""
    gk =OK .geo_anahtarlari ()
    yg =yasak_gruplar (olcum_da ,der_yolu )
    g =np .array ([gk .get (str (p ),"none:"+str (p ))for p in pidler ])
    return ~np .isin (g ,list (yg ))


def dogrula (pidler ,ad ="training",olcum_da =False ,sert =True ,der_yolu ="results/_der_tam.pkl"):
    """Protokol bekcisi. Ihlal varsa `sert=True` whereas HATA FIRLATIR.

    Sessiz gecis YOKTUR: this fonksiyon each cagrida receipt writes, so "guard was run mu"
    sorusu sonradan yanitlanabilir.
    """
    pidler =[str (p )for p in pidler ]
    m =egitim_maskesi (pidler ,olcum_da ,der_yolu )
    ihlal =sorted (set (np .array (pidler )[~m ].tolist ()))
    rec_ ={"ad":ad ,"n_parca":len (set (pidler )),"n_ihlal_parca":len (ihlal ),
    "olcum_da_yasak":bool (olcum_da ),"ilk_ihlaller":ihlal [:20 ]}
    try :
        with io .open (MAKBUZ ,encoding ="utf-8")as f :
            hepsi =json .load (f )
    except Exception :
        hepsi =[]
    hepsi =[h for h in hepsi if h .get ("ad")!=ad ][-49 :]+[rec_ ]
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump (hepsi ,f ,indent =1 ,ensure_ascii =False )
    if ihlal and sert :
        raise AssertionError (
        f"PROTOKOL IHLALI [{ad }]: {len (ihlal )} part yasak geometri grubunda "
        f"(LOCKED{' + OLCUM'if olcum_da else ''}). Ilk 10: {ihlal [:10 ]}. "
        f"`protocol.egitim_maskesi()` ile filtreleyin.")
    return m 


    # --------------------------------------------------------------------------------------
    # F0-3 DAIMI OLCUM KURALLARI -- each arm for gecerli, istisnasiz.
KURALLAR ="""
1. KILL ONCEDEN YAZILIR. Kol kosmadan first "neyi gecerse yasar" yazili olmali; sonucu
   gorup threshold belirlemek yasak. (2026-08-04: r9c'de threshold tutmadi ve KAYDIRILMADI.)
2. AYAR = DEV + ATANMAMIS.  HUKUM = VAL.  Ayar kumesinde secilen TEK ayar VAL'de raporlanir.
3. HAVUZLANMIS SATIR SECIM ICIN KULLANILMAZ -- ayar kumesini de icerdigi for yaniltir.
   Olculdu (T1): DEV'de +0.0277 gosteren rule VAL'de -0.0162 cikti; havuzlanmis +0.0011
   diyordu. Havuzlanmis only BILGI satiridir.
4. GRUP BOOTSTRAP: part not GEOMETRI GRUBU. Parcalarin %80'inin ikizi present; part
   bootstrap'i confidence araligini SAHTE DARALTIR.
5. TEK REJIMLI ALT KUMEDE `f1w` CAGIRMA -- `f1_rejim` kullan (agirliklar present which rejimler
   uzerinden normalize edilir).
6. DAGITIM OLURSA: smoke_test.py + pytest tests/ + headline YENIDEN URETILIR ve config'e
   provenance yazilir.
7. ADAY-URETIMI kararlari (cluster_mm / min_vertices / dedupe / promote / havuzlama)
   ONBELLEKLE OLCULEMEZ -- yeniden turetme sart.
8. TEZ DEGISMEZLERI (asagidaki TEZ sozlugu) each kolun basinda dogrulanir.
9. BULUNAN HATA GOZ ARDI EDILMEZ. Bir madde biterken cikan error/uyari/tutarsizlik
   "then bakariz" diye gecilmez: LISTEYE H-maddesi as eklenir, ETKISI OLCULUR ve
   duzeltilir. Duzeltme measurement kumesini oynatabiliyorsa first ETKI raporlanir, then
   uygulanir. (Bu proje hatalari bulup ertelediginde each seferinde bedelini odedi:
   diffusion_net dususu, gate bayatlamasi, unsigned kahin, kimlik ayristirmasi.)
"""


def bolmeler (der_yolu ="results/_der_tam.pkl"):
    """(ayar_pidleri, hukum_pidleri) -- rule 2'nin TEK kaynagi."""
    s3 =json .load (io .open (OK .SPLIT ,encoding ="utf-8"))
    val ={str (p )for p in s3 ["val"]["parts"]}
    D ,_ =OK .cluster (der_yolu )
    hepsi ={r ["pid"]for r in D }
    return (hepsi -val ),(hepsi &val )


    # --------------------------------------------------------------------------------------
    # TEZ DEGISMEZLERI -- 34 maddelik listenin TAMAMI along dokunulmayacak three sey.
    # Soz as not TEST as tutulur: each arm kosarken this dogrulanir.
    #   1. AG      : Scheffler DiffusionNet, 5 sinif (Housing/Contact/SnapPoint/CableEntry/
    #                LabelSurface) -- dagitilan 4 checkpoint'lik ensemble
    #   2. ORGU    : UNIFORM IZOTROPIK remesh, hedef ~6000 vertex (tezin domain-gap cozumu)
    #   3. CP TANIMI: v_o = acikligin AGIZ boundary noktalarindan turetilen centre (Abb. 44)
    # Yardimci basliklar, gate, selector and fiziksel bayraklar SON ISLEMDIR; bunlari degistirmek
    # tezden deviation DEGILDIR. Yukaridaki ucunu degistirmek SAPMADIR.
TEZ ={"remesh_hedef":6000 ,"n_sinif":5 ,"n_checkpoint":4 ,
"cp_def":"cp-v3-thesis-connection-classes"}


def tez_dogrula (sert =True ):
    """Tez degismezleri yerinde mi? Liste along each kolun basinda cagrilir."""
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    deviation =[]
    cks =cfg ["current_product"].get ("checkpoints")or cfg .get ("robot_vote2_checkpoints")or []
    if len (cks )!=TEZ ["n_checkpoint"]:
        deviation .append (f"ensemble {len (cks )} uye (tez surumu {TEZ ['n_checkpoint']})")
    if cfg .get ("cp_def_version")!=TEZ ["cp_def"]:
        deviation .append (f"cp_def_version={cfg .get ('cp_def_version')} (beklenen {TEZ ['cp_def']})")
    try :
        import inspect 

        import thesis_remesh 
        src =inspect .getsource (thesis_remesh .remesh_uniform )
        if "target"not in src :
            deviation .append ("thesis_remesh.remesh_uniform imzasi degismis")
    except Exception as e :
        deviation .append (f"thesis_remesh okunamadi: {type (e ).__name__ }")
    if deviation and sert :
        raise AssertionError ("TEZDEN SAPMA: "+" | ".join (deviation ))
    return deviation 


def _selftest ():
    """Bekcinin GERCEKTEN yakaladigini dogrula -- 'silent gecis' testi."""
    lg =sorted (yasak_gruplar ())
    assert lg ,"LOCKED grubu bos -- protocol kurulamaz"
    gk =OK .geo_anahtarlari ()
    kirli =[p for p ,g in gk .items ()if g ==lg [0 ]]
    assert kirli ,"LOCKED grubuna ait part bulunamadi"
    m =egitim_maskesi (kirli )
    assert not m .any (),"guard LOCKED parcasini yakalamadi"
    try :
        dogrula (kirli ,ad ="_selftest",sert =True )
    except AssertionError :
        pass 
    else :
        raise AssertionError ("dogrula() ihlalde HATA FIRLATMADI -- sessiz gecis!")
    temiz =[p for p ,g in gk .items ()if g not in yasak_gruplar ()][:5 ]
    assert egitim_maskesi (temiz ).all (),"guard temiz parcayi yanlislikla eledi"
    return True 


if __name__ =="__main__":
    import sys 
    print (f"LOCKED yasak grubu: {len (yasak_gruplar ())}")
    print (f"LOCKED+OLCUM yasak grubu: {len (yasak_gruplar (olcum_da =True ))}")
    d =np .load ("results/zengin_parite_w2.npz",allow_pickle =True )
    pid =[str (x )for x in d ["pids"]]
    for ad ,oc in (("dagitilan_gate_LOCKED",False ),("dagitilan_gate_LOCKED+OLCUM",True )):
        m =egitim_maskesi (pid ,olcum_da =oc )
        u =set (pid );ui =set (np .array (pid )[~m ].tolist ())
        print (f"{ad :<30} {len (u )} part -> ihlal {len (ui )} | candidate {len (pid )} -> temiz {int (m .sum ())}")
        dogrula (pid ,ad =ad ,olcum_da =oc ,sert =False )
    print (f"\n_selftest: {'GECTI'if _selftest ()else 'KALDI'}")
    print (f"receipt -> {MAKBUZ }")
