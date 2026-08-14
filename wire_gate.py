# -*- coding: utf-8 -*-
"""WIRE/TOOL KAPISI: robotun buldugu acikliklardan TOOL/actuator agizlarini eleyip TEL-girislerini keeps.

Otopsi (OTOPSI_CP_TIPI.md): tez Contact sinifi = 'Kontaktierung bzw. Werkzeugeinschub' -> tool agzi
kontaktla same sinif, segmentasyon ayirmaz, geometri de ayirmaz (recon AUC 0.61). AMA yapisal/baglamsal
ozellikler AYIRIR (held-out AUC 0.867; outward + ce_frac basi ceker). Bu modul that ayiriciyi egitir,
kaydeder and cikarimda uygular.

Uctan-uca dogrulama (gorulmemis part, karisik manufacturer): precision 0.607->0.751 (+0.144) @ threshold 0.30,
recall only 0.684->0.651. WEI P+0.087 / PXC P+0.217. Tool agizlari atiliyor, real teller kaliyor.

feats_for(): CP basina yapisal feature. train_and_save(): npz'den egit+kaydet. apply(): CP listesini filtrele.
"""
import os ,json ,numpy as np 
import collections as _collections 

# SESSIZ NOTRLESME SAYACI. Bu projede a times yasandi: rtree kurulu olmadigi for trimesh
# contains()/ray HER cagride patliyordu, `except: continue` bunu yutuyordu and two fonksiyon
# SESSIZ NO-OP'a donusmustu -- kimse difference etmedi because output "makul" gorunuyordu.
# Buradaki feature bloklari da error halinde NOTR (sifir) donuyor; notr donmek correct davranis
# but SESSIZ olmasi not. Her notr path here sayilir; measurement betikleri bunu basar and
# kitlesel notrlesme varsa gorunur becomes.
FALLBACK =_collections .Counter ()


def fallback_ozet (sifirla =True ):
    """Notr-donus sayaclarini dondur (and istege bagli sifirla)."""
    d =dict (FALLBACK )
    if sifirla :
        FALLBACK .clear ()
    return d 

FEAT_NAMES_13 =["size","depth","nn_dist","n_close","outward","nverts","ce_frac","ct_frac",
"aspect","flat","chan_conn","votes","conf"]
# E (2026-07-30): EKSEN GUVENI. Adayin ekseni B-rep'ten OKUNABILDI mi, hangi yuzeyden?
# Olculdu (same split, same candidates, single degisken): axis BELIRLI adaylarda TP orani %93.8,
# BELIRSIZ olanlarda %85.0 (+8.8 score). Capraz dogrulamali gate karsilastirmasi: 13 feature
# CP-F1 0.5798 -> 13+4 feature 0.6005 (+0.0208, kill esigi +0.005).
# TAM KORPUS gate'inde de gecti (+0.0068) AMA UCTAN UCA KAYBETTI and REVERTED:
# same candidates, two gate, single difference feature kumesi -> tespit 0.7790 -> 0.7665 (-0.0125),
# robot-hazir +0.0001 (absent). Yani gate DUZEYINDE kazanan a feature real hatta zarar
# verebiliyor; measurement havuzu (npz adaylari) with calisma havuzu (I with turetilen candidates) same
# not. VARSAYILAN KAPALI. Model results/wire_gate_axis17.pkl as duruyor.
FEAT_NAMES_AXIS =["ax_cyl","ax_plane","ax_none","ax_radius"]
USE_AXIS_FEATS =os .environ .get ("WG_AXIS_FEATS","0")not in ("0","false","False")

# FIZ (2026-07-31): B-rep FIZIKSEL ozellikler. E'den farki, E "axis OKUNABILDI mi" diye
# soruyordu (a GUVEN olcusu); bunlar acikligin FIZIGINI olcuyor.
#
# WHY: FP otopsisi yanlislarin %86.3'unu three fiziksel imzaya bagliyor (boydan-boya hole %31,
# radius<1mm %21, axis-dik %35) and 13 ozelligin HICBIRI bunlari tasimiyordu.
#
# MEASURED (q4/q5, 3277 candidate, 200 AYRIK part, grup-capraz OOF):
#   single feature AUC: brep_r 0.707 (null p95 0.519) -- mevcut `size` 0.583, `depth` 0.583
#   brep_r with size KORELASYONU -0.009 -> kopya DEGIL, GERCEKTEN YENI BILGI
#   13 -> 13+5: OOF AUC 0.8576 -> 0.8996 (+0.042), candidate-duzeyi F1 0.6997 -> 0.7554 (+0.0556)
#
# brep_r IKI sinyal tasiyor, ikisi de real (ayristirildi):
#   (a) "B-rep silindiri eslesti mi": TP %85.5 vs FP %54.1  -> single basina AUC 0.657
#   (b) eslesenlerde YARICAPIN KENDISI: TP medyan 2.00mm vs FP 1.05mm -> AUC 0.607
# Yani salt a "eslesme bayragi" not; olcunun kendisi de ayiriyor.
#
# Bu measurement radius hatasi duzeltilmeden YAPILAMAZDI (see. brep_axes._fit_circle: radius
# yay-centroid'inden hesaplanirken 3.5 fold small cikiyordu).
#
# VARSAYILAN KAPALI: dagitilan gate 13 sutunlu. Acmadan ONCE full corpus gate verisi this 5
# sutunla yeniden uretilip gate yeniden egitilmeli, after UCTAN UCA olculmeli (E'nin dersi:
# gate duzeyinde kazanan feature real hatta kaybedebilir).
FEAT_NAMES_FIZ =["brep_r","esesenli","r_orani","bos_derinlik","gecen"]


def _fiz_default ():
    """Bayragin TEK DOGRULUK KAYNAGI cp_config; ortam degiskeni only deney for ezer.

    Yalniz ortam degiskenine bakmak URETIMDE SESSIZCE KAPALI kalmasi demekti: robot hattinda
    kimse WG_FIZ_FEATS set etmiyor, that zaman feats_for 13 column produces and 18 sutunlu gate
    calismaz. Ayni sinifta a error E maddesinde yasandi (step_path gecmeyince 4 column sifir).
    """
    v =os .environ .get ("WG_FIZ_FEATS")
    if v is not None :
        return v not in ("0","false","False")
    try :
        with open ("cp_config.json",encoding ="utf-8")as f :# see. [[file-tanitici-sizintisi]]
            return bool (json .load (f ).get ("gate_fiz_feats",False ))
    except Exception :
        return False 


USE_FIZ_FEATS =_fiz_default ()

# EK (2026-07-31): two olculmus open kalem, single blokta. FIZ blogunun ARKASINA eklenir ki
# dagitilan 18-sutunlu gate'in column anlamlari degismesin (apply first n_feat sutunu takes).
#
#  ic_derinlik   : ISIN ICERI (-d). `bos_derinlik` yazilirken isin DISARI gonderilmisti; olcunce
#                  onun "disarisi gercekten empty mu" oldugu anlasildi (TP'lerin %75.1'inde isin never
#                  carpmiyor, FP'lerin %35.6'sinda). Kanal DERINLIGI never olculmemisti: tel girisi
#                  ~5-15mm'de metal kelepcede biter, montaj deligi bitmez.
#  c_govde       : adayin oturdugu silindirin rengi body (metal disi) mi   -- AUC 0.622
#  kanalda_metal : same axis cizgisindeki silindirlerden biri metal mi     -- AUC 0.493 (zayif)
#  metal_mesafe  : most yakin metal yuze uzaklik (mm)                         -- AUC ters 0.639
#                  (TP ort 42.5mm, FP ort 62.0mm)
#
# RENK NEREDEN: renk STEP metninden (OVER_RIDING_STYLED_ITEM -> ADVANCED_FACE), geometri OCC'den
# (kuresel koordinat). Eslesme SILINDIR ALT-DIZISI with is done and PARCA BASINA dogrulanir
# (two taraf da radius veriyor; 13 parcada 12'si eleman eleman equal). Dogrulamayan parcada
# renk sutunlari NOTR kalir -- korpusun ~%65'inde renk cozuluyor.
FEAT_NAMES_EK =["ic_derinlik","c_govde","kanalda_metal","metal_mesafe"]


def _ek_default ():
    v =os .environ .get ("WG_EK_FEATS")
    if v is not None :
        return v not in ("0","false","False")
    try :
        with open ("cp_config.json",encoding ="utf-8")as f :# see. [[file-tanitici-sizintisi]]
            return bool (json .load (f ).get ("gate_ek_feats",False ))
    except Exception :
        return False 


USE_EK_FEATS =_ek_default ()


def _cfg_get (key_ ,ortam ,varsayilan ):
    v =os .environ .get (ortam )
    if v is not None :
        return type (varsayilan )(v )
    try :
        with open ("cp_config.json",encoding ="utf-8")as f :# see. [[file-tanitici-sizintisi]]
            return type (varsayilan )(json .load (f ).get (key_ ,varsayilan ))
    except Exception :
        return varsayilan 


        # GORELI ESIK: see. apply() icindeki not. Varsayilan KAPALI -- uctan uca olculmeden acilmaz.
GORELI_ESIK =str (_cfg_get ("gate_goreli_esik","WG_GORELI","0")).lower ()not in ("0","false","")
GORELI_ORAN =float (_cfg_get ("gate_goreli_oran","WG_GORELI_ORAN",0.5 ))
GORELI_TABAN =float (_cfg_get ("gate_goreli_taban","WG_GORELI_TABAN",0.20 ))


# TOPO (2026-08-01): ICBUKEY KENAR TOPOLOJISI -- hole-tanima alaninin birinci sinyali.
# Bir opening, ICBUKEY kenarlarla cevrili face kumesidir; disa cikinti DISBUKEY kenarlarla.
# Gate'in 18 sutununun HICBIRI topolojik degildi (all of them olasilik istatistigi ya da point-geometrisi).
#
# MEASURED (t13, 3277 candidate / 197 ayrik part, bag duzeltmeli Mann-Whitney + permutasyon null):
#   kon_cevre 0.709 (TP medyan 1.000 = TAM TUR icbukey halka, FP 0.833) | kon_sayi 0.703
#   kon_oran  0.666 | kon_aci 0.520 (OLU, but blokta tutuluyor -- model karar versin)
# ARTIMLI (t14, same candidates): tanidik +0.0167, GORULMEMIS URETICIDE EN KOTU +0.0176.
# Ikisinde birden kazanmasi beklenendi: hole each ureticide deliktir, istatistik not GEOMETRI.
#
# MESH tabanli -- B-rep/renk yollarindaki kapsama kaybi here YOK, each parcada is computed.
FEAT_NAMES_TOPO =["kon_oran","kon_sayi","kon_aci","kon_cevre"]
USE_TOPO_FEATS =str (_cfg_get ("gate_topo_feats","WG_TOPO","0")).lower ()not in ("0","false","")
# YARICAP: 6.0 mm ILK SURUMDE KEYFI secilmisti; taranmasi for ayarlanabilir olmasi sart.
# EGITIM and CIKARIM AYNI R'yi kullanmali -- gate'in ogrendigi sutunlar R'ye baglidir. Bu yuzden
# value cp_config'te tutuluyor and modelin makbuzuna yaziliyor (gate_topo_r).
TOPO_R =float (_cfg_get ("gate_topo_r","WG_TOPO_R",6.0 ))


def _topo_feats (V ,F ,cps ):
    """TOPO: candidate basina 4 icbukey-edge ozelligi. Kenar yapisi PARCA BASINA BIR KEZ is computed."""
    import numpy as _np 
    n =len (cps )
    try :
        import topo_feats as _T 
        onb =_T .icbukey_kenarlar (V ,F )
    except Exception as _e :
        FALLBACK [f"topo:error:{type (_e ).__name__ }"]+=n 
        return _np .zeros ((n ,4 ),float )
    out =_np .zeros ((n ,4 ),float )
    for i ,c in enumerate (cps ):
        try :
            out [i ]=_T .topo_ozellik (V ,F ,_np .asarray (c ["point"],float ),
            _np .asarray (c ["direction"],float ),R =TOPO_R ,cache =onb )
        except Exception :
            FALLBACK ["topo:aday_hata"]+=1 
    return out 


    # ZENGIN (2026-08-02): KONUM + COK-YARICAP + NORMAL-DEGISIM. Denetimin P1 tavsiyesi.
    #
    # WHY: 22 sutunun tamami ADAYIN KENDI cevresini anlatiyor -- "this opening neye benziyor".
    # Hicbiri "this opening PARCANIN NERESINDE" ya da "cevresindeki 3/6/10/15mm kurelerde
    # SEGMENTASYON ne diyor" sorusunu tasimiyordu. Klemenste tel girisleri belirli yuzeylerde
    # siralanir; konum gercekten ayirt edici a bilgi.
    #
    # MEASURED (p2 candidate duzeyi, 19631 candidate / 1599 part, grup-capraz OOF):
    #   +konum9 +0.0114 | +cokyaricap24 +0.0169 | +all of them(33) +0.0231  (ucu de HER IKI ureticide +)
    # UCTAN UCA (p4, kilitli cluster 194 part / 171 grup, GRUP bootstrap, karar_olcutu):
    #   tanidik 0.7301 -> 0.7534 (+0.0233) | WEI-disi 0.5593 -> 0.5946 | PXC-disi 0.6619 -> 0.6743
    #   GA(WEI) [+0.0005, +0.0697] = KANITLI. Bes sartin BESI de saglandi -> DEPLOYED.
    # TAPER (huni profili) BILEREK YOK: more before measured and OLU output.
FEAT_NAMES_ZENGIN =(["kon_x","kon_y","kon_z"]
+[f"yuz_{a }{b }"for b in "xyz"for a in ("lo","hi")]
+[f"r{int (r )}_{k }"for r in (3 ,6 ,10 ,15 )
for k in ("c0","c1","c2","c3","c4","yog")]
+[f"nstd_{int (r )}"for r in (3 ,6 ,10 )])
USE_ZENGIN_FEATS =str (_cfg_get ("gate_zengin_feats","WG_ZENGIN","0")).lower ()not in ("0","false","")


def _zengin_feats (V ,F ,probs ,cps ):
    """36 column: konum(3) + bbox-face uzakligi(6) + very-radius sinif profili(24) + normal-std(3).

    `build_zengin_parite.zengin` with AYNI hesap -- training and inference single kaynaktan olsun diye
    oradan cagriliyor (kopyalanmiyor)."""
    import numpy as _np 
    n =len (cps )
    try :
        from build_zengin_parite import _normaller ,zengin as _z 
        return _z (_np .asarray (V ,float ),_np .asarray (F ),_np .asarray (probs ,float ),cps ,
        _normaller (V ,F ))
    except Exception as _e :
        FALLBACK [f"zengin:error:{type (_e ).__name__ }"]+=n 
        return _np .zeros ((n ,len (FEAT_NAMES_ZENGIN )),float )


FEAT_NAMES =(FEAT_NAMES_13 +(FEAT_NAMES_AXIS if USE_AXIS_FEATS else [])
+(FEAT_NAMES_FIZ if USE_FIZ_FEATS else [])
+(FEAT_NAMES_EK if USE_EK_FEATS else [])
+(FEAT_NAMES_TOPO if USE_TOPO_FEATS else [])
+(FEAT_NAMES_ZENGIN if USE_ZENGIN_FEATS else []))


def _ek_feats (V ,F ,cps ,step_path ):
    """EK: iceri channel derinligi + renk. step_path otherwise/renk cozulmezse NOTR returns."""
    import numpy as _np 
    n =len (cps )
    out =_np .zeros ((n ,4 ),float )
    out [:,0 ]=-1.0 # ic_derinlik: carpma otherwise -1
    out [:,3 ]=99.0 # metal_mesafe: metal otherwise 99mm
    if not step_path :
        FALLBACK ["ek:step_path_yok"]+=n 
        return out 
    try :
        import cp_geometry as _G 
        import trimesh as _tm 
        mesh =_tm .Trimesh (vertices =_np .asarray (V ,float ),faces =_np .asarray (F ),process =False )
    except Exception as _e :
        FALLBACK [f"ek:mesh_hata:{type (_e ).__name__ }"]+=n 
        return out 
    for i ,c in enumerate (cps ):
        p =_np .asarray (c ["point"],float );d =_np .asarray (c ["direction"],float )
        try :
            h =_np .asarray (_G .ray_hits (mesh ,p -0.05 *d ,-d ,max_mm =200.0 ),float ).ravel ()
            if len (h ):
                out [i ,0 ]=float (h [0 ])# agizdan first tabana distance = channel derinligi
        except Exception :
            pass 
    try :
        import step_face_colors as _SC 
        faces ,ok =_SC .read_all_by_order (step_path )
    except Exception as _e :
        FALLBACK [f"ek:renk_hata:{type (_e ).__name__ }"]+=n 
        return out 
    if not ok :
        FALLBACK ["ek:sira_dogrulanmadi"]+=n # renk sutunlari NOTR kalir
        return out 
        # METAL MESAFESI TUM yuzlerden: tel kelepcesi silindirik not, DUZ metal yuzeydir.
        # Olculdu (q11, 70 part / 809 candidate): only metal silindirlerle AUC ters 0.658
        # (TP 52.3mm / FP 76.3mm); TUM metal yuzlerle ters 0.699 (TP 36.8mm / FP 66.2mm)
        # -> ayirt edicilik +0.0415. Kapsama da artiyor (28 -> 32 part), because mutlak face
        # sirasindaki SABIT KAYMA residual cozuluyor, part atilmiyor.
    mcom =_np .array ([f ["centroid"]for f in faces if f ["is_metal"]],float )
    cyl =[f for f in faces if f ["axis"]is not None and f ["rgb"]is not None ]
    if len (cyl )<2 :
        if len (mcom ):# renk present but silindir absent: mesafeyi yine yaz
            for i ,c in enumerate (cps ):
                p =_np .asarray (c ["point"],float )
                out [i ,3 ]=float (_np .min (_np .linalg .norm (mcom -p ,axis =1 )))
        return out 
    C =_np .array ([f ["com"]for f in cyl ],float )
    A =_np .array ([f ["axis"]/(_np .linalg .norm (f ["axis"])+1e-12 )for f in cyl ],float )
    MET =_np .array ([f ["is_metal"]for f in cyl ],bool )
    for i ,c in enumerate (cps ):
        p =_np .asarray (c ["point"],float );d =_np .asarray (c ["direction"],float )
        rel =p -C 
        al =(rel *A ).sum (1 )
        off =_np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
        near =off <=3.0 
        if near .any ():
            cand =_np .where (near )[0 ]
            j =int (cand [int (_np .argmax (_np .abs (A [cand ]@d )))])
            out [i ,1 ]=float (not MET [j ])
            par =_np .abs (A @A [j ])>0.99 
            dd =C -C [j ]
            coax =par &(_np .linalg .norm (dd -(dd @A [j ])[:,None ]*A [j ],axis =1 )<1.5 )
            out [i ,2 ]=float (bool ((MET &coax ).any ()))
        if len (mcom ):
            out [i ,3 ]=float (_np .min (_np .linalg .norm (mcom -p ,axis =1 )))
    return out 

MODEL_PATH ="results/wire_gate.pkl"


def _fiz_feats (V ,F ,cps ,step_path ):
    """FIZ: B-rep fiziksel ozellikleri. step_path otherwise notr returns -- old cagrilar bozulmaz."""
    import numpy as _np 
    n =len (cps )
    if not step_path :
        FALLBACK ["fiz:step_path_yok"]+=n 
        return _np .zeros ((n ,5 ),float )
    try :
        import brep_axes as _bx 
        import cp_geometry as _G 
        import trimesh as _tm 
        C ,A ,R =_bx .cylinders (step_path )
        mesh =_tm .Trimesh (vertices =_np .asarray (V ,float ),faces =_np .asarray (F ),process =False )
    except Exception as _e :
        FALLBACK [f"fiz:brep_hata:{type (_e ).__name__ }"]+=n 
        return _np .zeros ((n ,5 ),float )
    out =[]
    for c in cps :
        p =_np .asarray (c ["point"],float );d =_np .asarray (c ["direction"],float )
        brep_r ,coax_n ,r_or =0.0 ,0 ,1.0 
        if len (C ):
            rel =p -C 
            al =(rel *A ).sum (1 )
            off =_np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
            ok =off <=3.0 
            if ok .any ():
                cand =_np .where (ok )[0 ]
                j =cand [int (_np .argmax (_np .abs (A [cand ]@d )))]
                brep_r =float (R [j ])
                par =_np .abs (A @A [j ])>0.995 
                dd =C -C [j ]
                coax =par &(_np .linalg .norm (dd -(dd @A [j ])[:,None ]*A [j ],axis =1 )<0.5 )
                coax_n =int (coax .sum ())
                rr =R [coax ]
                if len (rr )>1 :
                    r_or =float (rr .max ()/max (rr .min (),1e-6 ))
                    # `bos_derinlik` ADI YANILTICI: isin DISARI (+d) gidiyor, i.e. kanalin dibine not
                    # aciklikin ONUNE. Olctugu sey "disarisi gercekten empty mu": measured (3277 candidate) TP'lerin
                    # %75.1'inde isin never carpmiyor (opening bosluga bakiyor), FP'lerin whereas only %35.6'sinda.
                    # Yani gecerli a opening-dogrulamasi, depth olcusu DEGIL. ICERI (-d) bakan version
                    # (real channel derinligi = kelepceye distance) HENUZ DENENMEDI -- listede open madde.
        derin ,gecen =-1.0 ,0 
        try :
            h =_np .asarray (_G .ray_hits (mesh ,p +0.05 *d ,d ,max_mm =200.0 ),float ).ravel ()
            if len (h ):
                derin =float (h [0 ]);gecen =int (len (h )<=1 )
            else :
                gecen =1 # HIC carpmadi: depth -1 kalir, 0.0'dan AYRI
        except Exception :
            pass 
        out .append ([brep_r ,coax_n ,r_or ,derin ,gecen ])
    return _np .array (out ,float ).reshape (n ,5 )


def _axis_feats (cps ,step_path ):
    """E: axis guveni ozellikleri. step_path otherwise notr (sifir) returns -- old cagrilar bozulmaz."""
    import numpy as _np 
    if not step_path :
        return _np .zeros ((len (cps ),4 ),float )
    try :
        import brep_axes as _bx 
        cyl =_bx .cylinders (step_path );pl =_bx .planes (step_path )
    except Exception :
        return _np .zeros ((len (cps ),4 ),float )
    out =[]
    for c in cps :
        p =_np .asarray (c ["point"],float );d =_np .asarray (c ["direction"],float )
        a ,rad =_bx .axis_at (p ,d ,cyl ,max_off_mm =5.0 ,max_turn_deg =60.0 ,want_radius =True )
        c_ok =a is not None 
        p_ok =(not c_ok )and (_bx .axis_from_planes (p ,d ,pl ,max_dist_mm =6.0 ,min_faces =4 ,
        flat_ratio =0.20 ,max_turn_deg =45.0 )is not None )
        out .append ([float (c_ok ),float (p_ok ),float (not (c_ok or p_ok )),
        float (rad )if (c_ok and rad )else 0.0 ])
    return _np .array (out ,float )


def feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =None ):
    """each CP for yapisal feature (mesh frame). probs = (N,5) olasilik; cps = connection_points ciktisi.

    step_path verilir and WG_AXIS_FEATS=1 whereas 4 EKSEN GUVENI ozelligi eklenir (see FEAT_NAMES_AXIS).
    Varsayilan KAPALI: dagitilan gate 13 ozellikle egitildi, shape uyusmazligi onu bozardi.
    Yeni gate egitilince bayrak acilir."""
    conn =probs [:,CE ]+probs [:,CT ]
    ctr =0.5 *(V .min (0 )+V .max (0 ))
    P =np .array ([np .asarray (c ["point"],float )for c in cps ])
    Dv =np .array ([np .asarray (c ["direction"],float )for c in cps ])
    Dv =Dv /(np .linalg .norm (Dv ,axis =1 ,keepdims =True )+1e-9 )
    X =[]
    for i ,c in enumerate (cps ):
        p =P [i ];d =Dv [i ]
        area =float (c .get ("area",0.0 ));size =2.0 *(area /np .pi )**0.5 if area >0 else 0.0 
        depth =float (c .get ("insertion_depth_mm",0.0 ))
        if len (P )>1 :
            dd =np .linalg .norm (P -p ,axis =1 );dd [i ]=1e9 
            nn =float (dd .min ());nclose =int ((dd <=12.0 ).sum ())
        else :
            nn =50.0 ;nclose =0 
        mo =np .abs ((V -ctr )@d ).max ();outward =float (((p -ctr )@d )/(mo +1e-9 ))
        near =np .linalg .norm (V -p ,axis =1 )<=6.0 
        nverts =int (near .sum ())
        ce_frac =float (probs [near ,CE ].mean ())if near .any ()else 0.0 
        ct_frac =float (probs [near ,CT ].mean ())if near .any ()else 0.0 
        if near .sum ()>=6 :
            Q =V [near ]-V [near ].mean (0 );sv =np .linalg .svd (Q ,compute_uv =False )
            aspect =float (sv [0 ]/(sv [1 ]+1e-6 ));flat =float (sv [2 ]/(sv [0 ]+1e-6 ))
        else :
            aspect =1.0 ;flat =0.0 
        rel =V -p ;al =rel @d ;perp =np .linalg .norm (rel -al [:,None ]*d [None ,:],axis =1 )
        chan =(al >=0 )&(al <=15 )&(perp <=4 );chan_conn =float (conn [chan ].mean ())if chan .any ()else 0.0 
        X .append ([size ,depth ,nn ,nclose ,outward ,nverts ,ce_frac ,ct_frac ,aspect ,flat ,chan_conn ,
        float (c .get ("_votes",1 )),float (c .get ("confidence",0.0 ))])
    X =np .array (X ,float )
    if USE_AXIS_FEATS :
        X =np .hstack ([X ,_axis_feats (cps ,step_path )])
    if USE_FIZ_FEATS :
        X =np .hstack ([X ,_fiz_feats (V ,F ,cps ,step_path )])
    if USE_EK_FEATS :
        X =np .hstack ([X ,_ek_feats (V ,F ,cps ,step_path )])
    if USE_TOPO_FEATS :
        X =np .hstack ([X ,_topo_feats (V ,F ,cps )])
    if USE_ZENGIN_FEATS :
        X =np .hstack ([X ,_zengin_feats (V ,F ,probs ,cps )])
    return X 


def train_and_save (npz ="results/f1_sweep_data.npz",out =MODEL_PATH ):
# RandomForest beats GradientBoosting ten the wire/tool gate (leakage-free GroupKFold OOF, robust across
# hyperparams/seeds): base F1 0.693->0.750, top-N/metadata 0.742->0.775, WEI 0.641->0.690. Same candidate
# data, same folds -- a genuinely stronger classifier, not a fold artifact.
    from sklearn .ensemble import RandomForestClassifier 
    import pickle 
    d =np .load (npz ,allow_pickle =True )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
    clf .fit (d ["X"],d ["y"])
    pickle .dump ({"clf":clf ,"feat_names":FEAT_NAMES },open (out ,"wb"))
    print (f"wire-gate egitildi RF ({len (d ['y'])} CP) -> {out }")
    return out 


_CACHE ={}
def _load (path =MODEL_PATH ):
    if path not in _CACHE :
        import pickle 
        if os .path .exists (path ):
            with open (path ,"rb")as f :# see. [[file-tanitici-sizintisi]]
                _CACHE [path ]=pickle .load (f )
        else :
            _CACHE [path ]=None 
    return _CACHE [path ]


def _cokus_yonlendir (m ,X_ham ,s_ham ):
    """COKUS YONLENDIRME: gate this parcada kararsizsa SAGLAM (part-ici) modele gec.

    WHY (measured 2026-08-01, u6/u7/u8): part-ici z-skor GENEL a iyilestirme DEGIL, a
    KURTARMA. Uctan uca 9 bolmede desen tekduze -- TABAN ZAYIFKEN kazaniyor, SAGLIKLIYKEN
    kaybediyor:
        WEI disarida baseline 0.4832 (COKUS)     -> +0.0869
        seri 25      baseline 0.5654             -> +0.0639
        seri 10      baseline 0.5980             -> +0.0086
        seri 17/15/30/32/16 baseline 0.66-0.78   -> -0.008 .. -0.060
        PXC disarida baseline 0.7203 (SAGLIKLI)  -> -0.0375
    Her parcaya uygulamak, cokmeyen parcalarda vergi odemek demekti.

    SEZGI metadata GEREKTIRMEZ: ham gate'in KENDI skor dagilimi cokusu gosteriyor. Olculmus
    teshis -- cokus halinde model adaylarin %10.3'une pozitif diyor, real %24.1. Yani parcanin
    EN YUKSEK ham skoru dusukse gate that parcada kararsizdir.

    ESIK training korpusunun 10'uncu yuzdeligi; modelin inside saklanir (`esik_cokus`), calisma
    aninda no sey hesaplanmaz and test tarafina BAKILMAZ.

    KAZANC (uctan uca, D = each parcaya z-skor):
        WEI disarida  D 0.5702 -> R 0.5682  (-0.0020, kazancin %98'i korunur)
        PXC disarida  D 0.6828 -> R 0.7029  (+0.0201, bedelin YARISI geri)
        tanidik       D 0.7390 -> R 0.7439  (+0.0049, ham gate'ten bile iyi)
        7 seri-disi   D -0.0075 -> R -0.0029 (vergi %60 azaldi)
    """
    clf_z =m .get ("clf_z")
    threshold =m .get ("esik_cokus")
    if clf_z is None or threshold is None or not len (s_ham ):
        return s_ham 
    if float (np .max (s_ham ))>=float (threshold ):
        return s_ham 
    return clf_z .predict_proba (within_part (X_ham ,m .get ("donusum_z","zskor")))[:,1 ]


def decision_score (m ,X ):
    """URUNUN DECISION SKORU -- single source. Model + ham feature matrisi -> candidate basina skor.

    WHY SEPARATE BIR FONKSIYON (2026-08-01): measurement betiklerim gate'i `clf.predict_proba(...)` diye
    ELDE yeniden kuruyordu. Urunun `apply()` yolu whereas arada yonlendirme does. Iki path
    AYRISIRSA olculen sey urunun YAPTIGI sey olmaz -- and difference small oldugu for tabloda
    difference edilmez. Artik ikisi de BURAYA cagiriyor.
    """
    X =np .asarray (X ,float )
    # DONUSUM DE BURADA: eskiden `apply()` yapiyordu, i.e. measurement betikleri decision_score'yu
    # cagirinca urunun YARISINI taklit etmis oluyordu (2026-08-01). Ham ozellikten skora
    # giden TUM path single fonksiyonda.
    Xd =within_part (X ,m .get ("donusum"))
    nf =m .get ("n_feat")
    if nf is not None and Xd .shape [1 ]!=nf :
        if Xd .shape [1 ]<nf :
            raise ValueError (f"gate {nf } sutun bekliyor, {Xd .shape [1 ]} uretildi "
            f"(WG_FIZ_FEATS acik mi?)")
        Xd =Xd [:,:nf ]
    s =m ["clf"].predict_proba (Xd )[:,1 ]
    # Yonlendirme HAM matrisle works: z modeli own donusumunu own uygular.
    return _cokus_yonlendir (m ,X ,s )


def decision_mask (s ,threshold =None ):
    """URUNUN KABUL KURALI -- single source (goreli threshold + mutlak baseline, ya da sabit threshold).

    `apply()` bunu CP sozlukleri uzerinden, measurement betikleri skor dizisi uzerinden uygular;
    kuralin KENDISI single places tanimli olsun diye ayrildi.
    """
    s =np .asarray (s ,float )
    if not len (s ):
        return np .zeros (0 ,bool )
    if GORELI_ESIK :
        return (s >=GORELI_ORAN *max (float (s .max ()),1e-9 ))&(s >=GORELI_TABAN )
    return s >=(float (threshold )if threshold is not None 
    else float (_cfg_get ("robot_wire_gate_threshold","WG_ESIK",0.35 )))


def _dogrula_uyum (m ):
    """Modelin egitildigi AYARLARLA calisma anindaki ayarlarin AYNI oldugunu dogrula.

    WHY REQUIRED (2026-08-01'de acilan mayin): topoloji yaricapi `TOPO_R` ayarlanabilir yapildi
    but model onu tasimiyordu. Biri `cp_config.gate_topo_r`'yi degistirse, R=6'da egitilmis gate
    R=8 ozellikleriyle beslenirdi -- SUTUN SAYISI AYNI KALDIGI ICIN HATA DA VERMEZDI. Gate sessizce
    kotulesir and bunu a sonraki full olcume up to no sey gostermez.

    Genislik uyumu (`n_feat`) BU HATAYI YAKALAMAZ: sutunlarin SAYISI not ANLAMI degisir.
    Bu yuzden ayar degerinin kendisi modelde saklanir and here karsilastirilir.
    """
    if m is None :
        return 
    r =m .get ("topo_r")
    if r is not None and USE_TOPO_FEATS and abs (float (r )-float (TOPO_R ))>1e-9 :
        raise ValueError (
        f"gate topoloji yaricapi {r } mm ile EGITILDI, calisma aninda {TOPO_R } mm uretiliyor. "
        f"Sutun sayisi ayni oldugu icin bu sessizce yanlis skor verirdi. "
        f"cp_config.gate_topo_r'yi {r } yap ya da gate'i {TOPO_R } mm ile yeniden egit.")


POSE_PATH ="results/pose_head.pkl"
# POSE KIRPMA TESHISI -- only `CP_POSE_KANCA` set edilince dolar.
# Kirpma ONCESI yer degistirme vektorunu keeps; so single kosudan each
# `maks_mm` degeri cevrimdisi yeniden kurulabilir. Urun yolunu DEGISTIRMEZ.
POSE_KANCA =[]


def _yerel_cerceve (d ):
    """Eksen + two dik unit vektor. Duzeltme YEREL cercevede ifade edilir ki part donunce
    de gecerli olsun (dunya koordinatlarina bagimli a correction donmus parcada anlamsizdir)."""
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    return d ,u ,np .cross (d ,u )


def pose_correct (X ,cps ,model_path =None ):
    """POST-GATE POSE DUZELTMESI -- kabul edilmis CP'lerin YANAL sapmasini duzelt.

    WHY (ceiling olcumu, results/t_tavan.json): kahin gate robot-haziri only +0.059 tasiyor;
    KONUM +0.325, YON +0.103. Yani gate ne up to iyilesirse iyilessin robota few yansiyor, is
    konumda. Bu, gate KARARINDAN SONRA calisan ayri a kafadir.

    WHY LEARNED, RULE DEGIL: 2026-08-01'de RULE tabanli four konum/direction kolu was tried and
    DORDU DE became (analitik eksene izdusum uctan uca -0.045, part-ici axis uzlasisi net -110
    point, yarik yonu net -187, B-rep gate taramasi 0.000). Hepsinin kusuru ayniydi: correction
    HERKESE same sekilde uygulaniyordu. Ogrenilmis kafa "ne up to and hangi yone" sorusunu ADAY
    BASINA cevaplar.

    MEASURED (uctan uca, kilitli cluster 194 part / 171 grup, GRUP bootstrap):
        robot-hazir 0.4407 -> 0.4835  (+0.0428, GA [+0.0220, +0.0680] -> KANITLI)
        tespit      0.7534 -> 0.7535  (+0.0001, bedelsiz)
    YON duzeltmesi NOT DEPLOYED: already correct which is yonleri bozuyordu (medyan 0.00 -> 2.36 derece).

    Duzeltme `maks_mm` with sinirli. Model otherwise CP'ler DEGISMEDEN returns.

    MODEL YOLU cevreden ezilebilir (`CP_POSE_MODEL`): candidate a pose head'i,
    DAGITILANA dokunmadan full zincirde olcebilmek for. Varsayilan = dagitilan.
    """
    if model_path is None :
        model_path =os .environ .get ("CP_POSE_MODEL",POSE_PATH )
    m =_load (model_path )
    if m is None or not len (cps ):
        return cps 
    X =np .asarray (X ,float )
    nf =m .get ("n_feat")
    if nf is not None and X .shape [1 ]!=nf :
        FALLBACK [f"pose:genislik:{X .shape [1 ]}!={nf }"]+=len (cps )
        return cps 
    try :
        pr =m ["model"].predict (X )
    except Exception as _e :
        FALLBACK [f"pose:error:{type (_e ).__name__ }"]+=len (cps )
        return cps 
        # KIRPMA SINIRI. Cevreden ezilebilir: `maks_mm` a HIPERPARAMETRE and
        # tarandigina dair kayit absent. 2026-08-14 teshisi, GT'lerin %83'u for
        # havuzda 10 mm yakinda correct yonlu a candidate olduğunu showed -- 3 mm
        # kirpma this bandin ucte birini bile kapsamiyor.
    mx =float (os .environ .get ("CP_POSE_MAKS_MM",m .get ("maks_mm",3.0 )))
    _kanca =bool (os .environ .get ("CP_POSE_KANCA"))
    for c ,p in zip (cps ,pr ):
        d ,u ,v =_yerel_cerceve (c ["direction"])
        dw =float (p [0 ])*u +float (p [1 ])*v 
        n =float (np .linalg .norm (dw ))
        if _kanca :
        # KIRPMA ONCESI state: single kosudan HERHANGI a mx degeri for
        # cevrimdisi yeniden kurulabilsin diye.
            POSE_KANCA .append ({
            "p0":list (map (float ,np .asarray (c ["point"],float ))),
            "dw":list (map (float ,dw )),
            "n":n ,
            "dir":list (map (float ,np .asarray (c ["direction"],float ))),
            })
        if n >1e-9 :
            c ["point"]=np .asarray (c ["point"],float )+dw *(min (n ,mx )/n )
            c ["_pose_mm"]=float (min (n ,mx ))
    return cps 


ACI_PATH ="results/aci_secici.pkl"


def angle_correct (X ,cps ,model_path =ACI_PATH ):
    """SECICI ACI DUZELTMESI -- only "yonu wrong" denen adaylarda.

    WHY SECICI: yonlerin most ZATEN TAM DOGRU (medyan 0.00 derece). Duzeltmeyi herkese
    uygulamak medyani 0.00'dan 2.36 dereceye cikariyordu -- this gecenin four olu kolunun ortak
    kusuru. Once a siniflandirici "this adayin acisi 10 dereceden extra wrong mi" der; only
    oyleyse regresor duzeltmeyi uygular.

    MEASURED (kilitli cluster 194 part / 171 grup, GRUP bootstrap):
        robot-hazir 0.5280 -> 0.5406  (+0.0126, GA [+0.0002, +0.0325] KANITLI)
        tespit      DEGISMEZ -- and this YAPISAL: tespit olcutu aciya BAKMAZ (am=180), i.e. this
        correction tespiti etkileyemez. Kazanc +0.02 barinin under but KANITLI and BEDELSIZ.
    """
    m =_load (model_path )
    if m is None or not len (cps ):
        return cps 
    X =np .asarray (X ,float )
    nf =m .get ("n_feat")
    if nf is not None and X .shape [1 ]!=nf :
        FALLBACK [f"aci:genislik:{X .shape [1 ]}!={nf }"]+=len (cps )
        return cps 
    try :
        ps =m ["sec"].predict_proba (X )[:,1 ]
        pr =m ["reg"].predict (X )
    except Exception as _e :
        FALLBACK [f"aci:error:{type (_e ).__name__ }"]+=len (cps )
        return cps 
    threshold =float (m .get ("threshold",0.10 ))
    for c ,p ,q in zip (cps ,ps ,pr ):
        if p <threshold :
            continue 
        d ,u ,v =_yerel_cerceve (c ["direction"])
        g =d +float (q [2 ])*u +float (q [3 ])*v 
        n =float (np .linalg .norm (g ))
        if n >1e-9 :
            c ["direction"]=g /n 
            c ["_aci_duzeltildi"]=float (p )
    return cps 


UYE_PATH ="results/uye_secici.pkl"


def pick_member_direction (X ,cps ,uye_listeleri ,model_path =UYE_PATH ,yakin_mm =5.0 ):
    """UYE YON SECIMI -- `_vote2` birlestirmede ATILAN uye yonlerinden most iyisini sec.

    WHY (R7 kahin olcumu): four modelin yonlerinde robot-haziri 0.5523 -> 0.6059 yapacak
    bilgi VAR (+0.0536). Konum tarafinda YOK (-0.0022; pose head already almis). `_vote2`
    konumu confidence-agirlikli ortaliyor but YONU only temsilciden aliyor -- digerleri ATILIYOR.

    Bu, "yapisal wall" sandigim seyi de kismen actı: angle>45 derece sapan 217 noktanin
    %21.7'sinde uyeler between GT'ye 10 derece inside a direction VAR. Analitik B-rep ekseni
    same grubun %0'ini kurtariyordu.

    MEASURED (kilitli cluster 194 part / 171 grup, GRUP bootstrap):
        robot-hazir 0.5523 -> 0.5801  (+0.0278, GA [+0.0105, +0.0490] KANITLI)
        tespit      DEGISMEZ (yapisal: tespit olcutu aciya bakmaz)
    Kahinin ~%52'si yakalandi.

    uye_listeleri: model basina CP listesi (birlestirme ONCESI).
    """
    m =_load (model_path )
    if m is None or not len (cps )or not uye_listeleri :
        return cps 
    X =np .asarray (X ,float )
    for i ,c in enumerate (cps ):
        p0 =np .asarray (c ["point"],float )
        d0 =np .asarray (c ["direction"],float )
        uy =[(d0 ,1.0 ,0.0 )]
        for lst in uye_listeleri :
            for mm in lst :
                q =np .asarray (mm ["point"],float )
                dd =float (np .linalg .norm (q -p0 ))
                if dd <=yakin_mm :
                    uy .append ((np .asarray (mm ["direction"],float ),
                    float (mm .get ("confidence",1.0 )),dd ))
        if len (uy )<2 :
            continue 
        DIR =np .array ([u [0 ]for u in uy ])
        DIR =DIR /(np .linalg .norm (DIR ,axis =1 ,keepdims =True )+1e-9 )
        CONF =np .array ([u [1 ]for u in uy ]);MES =np .array ([u [2 ]for u in uy ])
        ort =DIR .mean (0 );ort /=np .linalg .norm (ort )+1e-9 
        a_ort =np .degrees (np .arccos (np .clip (np .abs (DIR @ort ),0 ,1 )))
        a_bir =np .degrees (np .arccos (np .clip (np .abs (DIR @d0 ),0 ,1 )))
        rank_ =np .argsort (np .argsort (-CONF ))
        F =np .array ([[CONF [u_ ],MES [u_ ],a_ort [u_ ],a_bir [u_ ],float (len (DIR )),
        float (np .mean (a_ort )),float (rank_ [u_ ])]+X [i ].tolist ()
        for u_ in range (len (DIR ))],float )
        if m .get ("n_feat")is not None and F .shape [1 ]!=m ["n_feat"]:
            FALLBACK [f"uye:genislik:{F .shape [1 ]}!={m ['n_feat']}"]+=1 
            continue 
        try :
            j =int (np .argmax (m ["sec"].predict_proba (F )[:,1 ]))
        except Exception as _e :
            FALLBACK [f"uye:error:{type (_e ).__name__ }"]+=1 
            continue 
        if j :
            c ["direction"]=DIR [j ]
            c ["_uye_secildi"]=int (j )
    return cps 


def within_part (X ,donusum ):
    """PARCA-ICI BAGLAM: each sutunun, BU PARCANIN adaylari icindeki goreli konumunu ekle.

    WHY (measured 2026-08-01, u3/u4): gate GORULMEMIS a ureticide cokuyor. Cokusun a yarisi
    kalibrasyondu and GORELI ESIK with cozuldu -- i.e. SKORU part inside karsilastirmak. Bu, same
    fikrin OZELLIK duzeyindeki hali: "this candidate this parcadaki most derin 2. hole" ifadesi ureticiden
    bagimsizdir, "depth 4.2 mm" degildir.

    HAM sutunlar KORUNUR, uzerine goreli olanlar EKLENIR. Yalniz goreli kullanmak measured and
    KOTU (tanidik -0.090): mutlak buyuklukler real bilgi tasiyor. Model hangisini nerede
    kullanacagina own karar versin.

    UCTAN UCA (manufacturer-disi, 200 part):
        gorulmemis manufacturer EN KOTU  0.4832 -> 0.5702 (+0.0869, GA [+0.048,+0.126])
        DIGER manufacturer-disi split    0.7203 -> 0.6828 (-0.0375, GA [-0.065,-0.009])  <- GERCEK BEDEL
        tanidik                     0.7410 -> 0.7390 (-0.0023, GURULTU)
    Yani this a RISK TAKASI: ureticiler arasi YAYILIM 0.237 -> 0.113. Takasi kaldirmak for three
    path was tried (only ezberleyen sutunlar / only fiziksel sutunlar / ensemble) and UCU DE
    basarisiz -- kazanc with bedel same mekanizmadan geliyor (u5_takas.json).

    CALISMA ANINDA UYGULANABILIR: only parcanin KENDI adaylarini kullanir; corpus istatistigi,
    komsu part ya da manufacturer kimligi gerektirmez.
    """
    if not donusum :
        return X 
    X =np .asarray (X ,float )
    if donusum =="zskor":
        sd =X .std (0 )
        Z =np .where (sd >1e-12 ,(X -X .mean (0 ))/np .where (sd >1e-12 ,sd ,1.0 ),0.0 )
    elif donusum =="sira":
        Z =(np .full_like (X ,0.5 )if len (X )<2 else 
        np .argsort (np .argsort (X ,axis =0 ),axis =0 ).astype (float )/(len (X )-1 ))
    else :
        raise ValueError (f"bilinmeyen part-ici donusum: {donusum !r }")
    return np .hstack ([X ,Z ])


    # --- KALABALIK BASTIRMA (NMS) -----------------------------------------------
    # 2026-08-10. Ayni fiziksel agza birden very candidate dusuyor; Macar eslesme BIRE BIR
    # oldugu for fazlasi ZORUNLU FP. Gate skoru most high which is tutulur.
    #
    # MEASURED -- TAM URUN ZINCIRI (poz kafasi dahil), D7 brand-disi 835 part, MIKRO
    # (`probe_product_nms_uctan_uca.py`, `results/urun_nms_uctan_uca*.json`):
    #   r     robot            tespit           makro   artan brand   yikilan
    #   0   0.1956           0.4314           0.2051      -
    #   3   0.1974 (+0.0018) 0.4399 (+0.0085) 0.2062     8/12         CEM
    #   4   0.2008 (+0.0052) 0.4471 (+0.0157) 0.2118     9/12         -
    #   5   0.2029 (+0.0072) 0.4523 (+0.0210) 0.2146    10/12         -   <-- SECILEN
    #   6   0.2041 (+0.0085) 0.4556 (+0.0242) 0.2143    10/12         CEM
    #
    # YARICAP SECIMI r=5.0. RULE: "no markayi YIKMAYAN most large radius".
    # BILEREK ARGMAX DEGIL: D7 a DEV kumesi, argmax'ini almak ona ayar yapmaktir
    # (uclu-split-and-fake-kazanclar). r=6 more high robot gives but CEM'i
    # 0.0164 -> 0.0000 yikar; r=3'te de yikiliyor, r=4/5'te YUKSELIYOR -- i.e. CEM'in
    # skoru single a eslesmeye dayanan noise, but rule kurala uyulur.
    # BAGIMSIZ VERIFICATION (D6 468 part, gate yolu, `results/nms_tarama.json`):
    # r=5'te robot +0.0005, 6/8 brand artida, most kotu brand 0.0155 -> 0.0157 (yikim YOK).
NMS_MM =float (_cfg_get ("robot_cp_nms_mm","WG_NMS_MM",5.0 ))


def crowd_mask (P ,skor ,r_mm =None ):
    """NMS'in TEK GERCEKLEMESI. Doner: tutulacaklarin bool maskesi.

    Hem urun yolu (`apply` -> `suppress_crowd`) hem measurement yolu (`canonical_d7.
    urun_poz`) BU fonksiyonu cagirir. Iki yolun ayrisip different number uretmesi this
    projede more before measured (see. measurement-yolu-and-secim-kusurlari); that yuzden
    rule TEK YERDE durur.
    """
    r =NMS_MM if r_mm is None else float (r_mm )
    P =np .asarray (P ,float );skor =np .asarray (skor ,float )
    tut =np .ones (len (P ),bool )
    if r <=0 or len (P )<2 :
        return tut 
    rank_ =np .argsort (-skor )
    for a ,i in enumerate (rank_ ):
        if not tut [i ]:
            continue 
        for j in rank_ [a +1 :]:
            if tut [j ]and np .linalg .norm (P [i ]-P [j ])<r :
                tut [j ]=False 
    return tut 


def suppress_crowd (cps ,r_mm =None ):
    """Birbirine `r_mm` mm'den yakin CP'lerden only most high wire_score'lu kalir.

    r_mm<=0 whereas list DEGISMEDEN returns (arm kapatilabilir). Girdi order korunur.
    """
    if len (cps )<2 :
        return cps 
    m =crowd_mask ([c ["point"]for c in cps ],
    [c .get ("wire_score",0.0 )for c in cps ],r_mm )
    return [c for c ,k in zip (cps ,m )if k ]


def apply (V ,F ,probs ,cps ,CE ,CT ,threshold =0.30 ,model_path =MODEL_PATH ,top_n =None ,
step_path =None ):
    """CP listesini wire-gate with filtrele. Her CP'ye wire_score yazilir.
    top_n=None (BASE urun): skor>=threshold olanlari tut (tool agizlari atilir).
    top_n=N (METADATA-ASSISTED mod): manufacturer CP count N biliniyorsa, most high wire_score N tanesini tut
      (threshold instead of). Olculdu (leakage-free OOF): ALL F1 0.775 vs base 0.750 (RF gate). AYRI mod as
      raporla, base urun F1'i with karistirma.
    Model otherwise VEYA <2 CP varsa listeyi degistirmeden returns."""
    m =_load (model_path )
    _dogrula_uyum (m )
    if m is None or len (cps )<2 :
        for c in cps :c ["wire_score"]=1.0 
        return cps if top_n is None else sorted (cps ,key =lambda c :-c .get ("confidence",0.0 ))[:top_n ]
    X =feats_for (V ,F ,np .asarray (probs ,float ),cps ,CE ,CT ,step_path =step_path )
    # SUTUN SECIMI: model hangi ozellikleri kullandigini KENDI carries. Boylece feats_for
    # degismeden gate'in feature kumesi degistirilebilir and old models calismaya devam eder.
    cols =m .get ("cols")
    if cols is not None :
        X =X [:,list (cols )]
        # PARCA-ICI BAGLAM: model own `donusum` alaninda carries (cols/n_feat with same tasarim),
        # so old models degismeden works. n_feat KONTROLUNDEN ONCE uygulanmali:
        # donusum column sayisini ikiye katlar.
        # DONUSUM and GENISLIK KONTROLU decision_score inside (single source).
    s =decision_score (m ,X )
    for c ,sc in zip (cps ,s ):
        c ["wire_score"]=float (sc )
    if top_n is not None :# metadata-assisted: most high wire_score N tanesi
        return sorted (cps ,key =lambda c :-c ["wire_score"])[:max (int (top_n ),0 )]
    if GORELI_ESIK :
    # GORELI ESIK (2026-07-31): sabit threshold instead of PARCA-ICI goreli karar + low mutlak baseline.
    #
    # WHY: sabit threshold new a URETICIDE cokuyor. Olculdu (manufacturer-disi split, 18 column):
    # gate F1 0.7422 -> 0.6399 (manufacturer 0 disarida) / 0.2799 (manufacturer 1 disarida). Ayristirma:
    # kaybin ~%42'si KALIBRASYON. Kanit: manufacturer 1'de model adaylarin %10.3'une pozitif diyor,
    # real %24.1 -- skor dagilimi kayinca sabit 0.40 very high kaliyor (orada most iyi 0.15).
    #
    # RULE: candidate, own PARCASINDAKI most high skorun `GORELI_ORAN` katini gecmeli VE mutlak
    # `GORELI_TABAN`'i asmali. Goreli kisim distribution kaymasini emer; TABAN whereas saf goreli
    # kuralin acigini kapatir -- baseline olmadan rule, no real CP olmayan parcada bile
    # "most yuksegin yarisi"ni kabul edip garanti wrong uretirdi (korpusta sifir-CP part present).
    #
    # MEASURED (goreli 0.5 + baseline 0.20): tanidik veride -0.0074, most kotu manufacturer-disi bolmede
    # 0.2799 -> 0.4402 (+0.160). Makbuz: results/t6_goreli_taban.json
        _m =decision_mask ([c ["wire_score"]for c in cps ])
        return suppress_crowd ([c for c ,k in zip (cps ,_m )if k ])
    _m =decision_mask ([c ["wire_score"]for c in cps ],threshold =threshold )# base: sabit threshold
    return suppress_crowd ([c for c ,k in zip (cps ,_m )if k ])


SP_NAMES =["sp_mir","sp_nnws","sp_cons","sp_dens","sp_cen","sp_nnd"]
SPATIAL_MODEL_PATH ="results/wire_gate_spatial.pkl"


def spatial_feats (Pp ,ws ):
    """6 within-part spatial sinyal (frame-invariant): mirror-partner, nn-wire_score, row/col-consistency,
    local-density, centrality, nn-dist. op2_spatial with same (yan-etkisiz kopya, deploy for)."""
    Pp =np .asarray (Pp ,float );ws =np .asarray (ws ,float );n =len (Pp )
    if n <2 :return np .zeros ((n ,6 ))
    Q =Pp -Pp .mean (0 );_ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False );u ,v =Vt [0 ],Vt [1 ]
    pu =Q @u ;pv =Q @v ;span =max (pu .max ()-pu .min (),pv .max ()-pv .min (),1.0 )
    Dm =np .linalg .norm (Pp [:,None ]-Pp [None ],axis =-1 );np .fill_diagonal (Dm ,1e9 )
    mir =np .zeros (n )
    for i in range (n ):
        refl =np .abs (pu +pu [i ])+np .abs (pv -pv [i ])
        j =int (np .argmin (refl +np .where (np .arange (n )==i ,1e9 ,0 )));mir [i ]=float (refl [j ]<0.08 *span )
    nn_ws =ws [np .argmin (Dm ,1 )]
    cons =np .zeros (n );tol =0.05 *span 
    for i in range (n ):
        al =((np .abs (pv -pv [i ])<tol )|(np .abs (pu -pu [i ])<tol ));al [i ]=False ;cons [i ]=float (ws [al ].sum ())
    cons =cons /max (cons .max (),1e-9 )
    dens =np .array ([float (ws [(Dm [i ]<0.10 *span )].sum ())for i in range (n )]);dens =dens /max (dens .max (),1e-9 )
    cen =np .linalg .norm (Q ,axis =1 )/max (np .linalg .norm (Q ,axis =1 ).max (),1e-9 )
    nnd =Dm .min (1 )/span 
    return np .stack ([mir ,nn_ws ,cons ,dens ,cen ,nnd ],1 )


def apply_spatial (V ,F ,probs ,cps ,CE ,CT ,top_n ,model_path =MODEL_PATH ,
spatial_path =SPATIAL_MODEL_PATH ,step_path =None ):
    """SPATIAL-RERANK (opt-in, top-N modu): base gate skorunu spatial feature'larla zenginlestir -> top-N.
    5/5 seed-robust (+0.002 ALL, WEI +0.01-0.02). base gate otherwise VEYA <2 CP -> normal apply(top_n).

    IKI MODEL, IKI GENISLIK: base gate 18 sutunlu may be (B-rep fiziksel ozellikler), spatial
    model whereas 13+spatial with egitildi. Bu yuzden base own `n_feat`'i up to column takes, spatial
    model always ILK 13'u gorur. Karistirilirsa sessizce wrong genislikte prediction yapilirdi.
    """
    base =_load (model_path );sp =_load (spatial_path )
    _dogrula_uyum (base )
    if base is None or sp is None or len (cps )<=max (int (top_n ),0 ):
        return apply (V ,F ,probs ,cps ,CE ,CT ,top_n =top_n ,model_path =model_path ,
        step_path =step_path )
    X =feats_for (V ,F ,np .asarray (probs ,float ),cps ,CE ,CT ,step_path =step_path )
    # base model part-ici donusum tasiyor may be (n_feat that zaman 2 katidir); spatial model
    # HER ZAMAN ham first 13 sutunu gorur -- ikisi ayri genislikte, karistirilmamali.
    Xb =within_part (X ,base .get ("donusum"))
    nb =base .get ("n_feat")or Xb .shape [1 ]
    ws0 =base ["clf"].predict_proba (Xb [:,:nb ])[:,1 ]# first-pass base skor
    X13 =X [:,:13 ]# spatial model 13 sutunla egitildi
    P =np .array ([np .asarray (c ["point"],float )for c in cps ])
    Xsp =np .hstack ([X13 ,spatial_feats (P ,ws0 )])
    s =sp ["clf"].predict_proba (Xsp )[:,1 ]
    for c ,sc in zip (cps ,s ):c ["wire_score"]=float (sc )
    return sorted (cps ,key =lambda c :-c ["wire_score"])[:max (int (top_n ),0 )]


if __name__ =="__main__":
    train_and_save ()


YON_SECICI_PATH ="results/yon_secici.pkl"


def pick_direction_from_dictionary (X ,cps ,V ,step_path =None ,uyeler =None ,model_path =YON_SECICI_PATH ):
    """AYRIK YON SECICI -- fiziksel a YON SOZLUGUNDEN most iyisini SECER (uretmez).

    WHY AYRIK, WHY REGRESYON DEGIL: surekli regresorle yonu yeniden kurmak DENENDI and
    each esikte DUSTU (2026-08-03, robot -0.0256). Sinifi TANIMAK (wrong-axis sinifi
    grup-capraz AUC 0.930) duzeltebilmek DEGILDIR. Burada model a direction URETMIYOR;
    fiziksel as turetilmis adaylardan BIRINI seciyor -- uyduramaz.

    SOZLUK (all of them calisma aninda turetilir):
        mevcut/ham    zincirin ciktisi and duzeltmesiz candidate yonu
        uye_k         birlestirmede atilan uye yonleri
        obb +/-       parcanin yonelimli boundary kutusu eksenleri
        uzlasi        parcadaki tum adaylarin direction uzlasisi
        yuz_uzlasi    same yuzdeki adaylarin uzlasisi
        dik +/-       adayin eksenine DIK duzlemde OBB eksenlerine most yakin yonler
        yuzn +/-      B-rep baskin duzlem normalleri
    MEASURED: dictionary, "angle mukemmel" tavaninin (0.6717) %90'ini kapsiyor.

    SAFETY: mevcut yonden however skor farki MARJI asarsa sapilir. "Herkese uygula"
    tuzagi this havuzda four kolu oldurmustu.

    MEASURED (194 part/174 grup, GRUP bootstrap; selector 1301 AYRI parcada egitildi,
    measurement kumesini and LOCKED'i HIC gormedi):
        robot-hazir 0.5893 -> 0.6213  (+0.0319, GA [+0.0120,+0.0527] KANITLI)
        tespit      0.7584 -> 0.7584  (YAPISAL: tespit olcutu aciya bakmaz)
    Model otherwise CP'ler DEGISMEDEN returns.
    """
    m =_load (model_path )
    if m is None or not len (cps )or V is None :
        return cps 
    try :
        from d1_direction_distribute import satirla ,sozluk_kur ,_birim 
        import numpy as _np 
        P =_np .array ([c ["point"]for c in cps ],float )
        Pd =_np .array ([c ["direction"]for c in cps ],float )
        YUZN =[]
        if step_path :
            try :
                import brep_axes as _ba 
                pl =_ba .planes (step_path )
                if pl is not None and len (pl ):
                    _n =_np .asarray (pl [1 ],float );_r =_np .asarray (pl [2 ],float )
                    for j in _np .argsort (-_r )[:6 ]:
                        b =_birim (_n [j ])
                        if b is not None :
                            YUZN .append (b )
            except Exception :
                pass 
                # PARITE: dictionary residual EGITIMDEKIYLE BIREBIR same (uye/ham girdileri
                # kaldirildi -- see. d1_direction_distribute.sozluk_kur docstring).
        SOZ ,UZ =sozluk_kur (V ,P ,Pd ,YUZN =YUZN )
        ax ,_ ,RJ =satirla (_np .asarray (X ,float ),SOZ ,UZ ,Pd ,None )
        if not ax :
            return cps 
        pr =m ["clf"].predict_proba (_np .array (ax ,float ))[:,1 ]
        SK ={}
        for n_ ,(i ,gi )in enumerate (RJ ):
            SK .setdefault (i ,{})[gi ]=pr [n_ ]
        marj =float (m .get ("marj",0.05 ))
        for i ,sc in SK .items ():
            gi =max (sc ,key =sc .get )
            if gi !=0 and sc [gi ]-sc .get (0 ,0.0 )>=marj :
                cps [i ]["direction"]=[float (x )for x in SOZ [i ][gi ][1 ]]
    except Exception as e :
        FALLBACK [f"yon_secici:{type (e ).__name__ }"]+=len (cps )
    return cps 
