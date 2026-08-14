# -*- coding: utf-8 -*-
"""E2 — SAHAYA INEN ZINCIR vs OLCULEN ZINCIR, ESLI KIYAS

SORUN. Ihracatcilar `robot_cp.extract` cagiriyor; kampanyada olculen
`product_p6`/`product_wide` sahaya HIC girmiyor. `export_robot_glb.py` icine
`cp_config.glb_kanonik_zincir` bayragi kondu but ACILMADI: saglamlik
denetimi gecti (40/40 part, cokme/NaN absent, yonler unit) fakat olculen
zincir **181 GT for 309 CP** uretiyordu. Fonksiyonel karsiligi robotun
olmayan yerlere gitmesidir; F1 same kalsa bile this a gerileme may be.

BU BETIK AYNI PARCALARDA IKI ZINCIRI YAN YANA runs and UC sayiyi birden
gives: robot F1, KESINLIK, and part basina uretilen CP count.

KESINLIK WHY SEPARATE RAPORLANIR. F1 precision with recall'i single sayiya
katlar; robot for bunlar same sey degildir. Yanlis a CP, robotun empty
yere hareket etmesi (and carpma riski) demektir. Bu yuzden bayrak
"F1 arttiysa ac" with DEGIL, "F1 artti VE precision gerilemedi" with acilir.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

N_PARCA =int (os .environ .get ("EZ_N","24"))
MARKALAR =set (os .environ .get ("EZ_MARKA","NIT,MOR,SUPU,UPUN").split (","))


_NRM ={}


def _graf_duzelt (pb ,F ,tur ,w ):
    """Y24: mesh KENARLARI on olasilik correction (mean-alan CRF).

    Her turda each tepenin olasilik vektoru, komsularinin ortalamasiyla
    `w` agirliginda harmanlanir: p <- (1-w)*p + w*komsu_ortalamasi.
    Bu, full a CRF'in mean-alan yaklasimidir and ek parametre/training
    GEREKTIRMEZ. `kind=0` or `w=0` -> bit-same baseline.
    """
    pb =np .asarray (pb ,float )
    if tur <=0 or w <=0 :
        return pb 
    F =np .asarray (F ,np .int64 )
    n =len (pb )
    # edge listesi (each ucgenin three kenari, two yonlu)
    e =np .vstack ([F [:,[0 ,1 ]],F [:,[1 ,2 ]],F [:,[2 ,0 ]]])
    e =np .vstack ([e ,e [:,::-1 ]])
    derece =np .bincount (e [:,0 ],minlength =n ).astype (float )
    derece [derece ==0 ]=1.0 
    for _ in range (int (tur )):
        top =np .zeros_like (pb )
        np .add .at (top ,e [:,0 ],pb [e [:,1 ]])
        kom =top /derece [:,None ]
        pb =(1.0 -w )*pb +w *kom 
        s =pb .sum (1 ,keepdims =True )
        pb =pb /np .maximum (s ,1e-12 )
    return pb 


def _yerel_normal (V ,F ,P ):
    """each prediction noktasi for YEREL DIS NORMAL (most yakin tepenin normali).

    "Disari bak" kurali for correct vekil budur: body merkezi kaba a
    yaklasimdir and klemens like uzun parcalarda yaniltir.
    """
    import numpy as _np 
    if not len (P ):
        return _np .zeros ((0 ,3 ))
    key_ =(len (V ),len (F ))
    try :
        import trimesh 
        ag =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
        N =_np .asarray (ag .vertex_normals ,float )
    except Exception :# noqa: BLE001
        return _np .zeros ((len (P ),3 ))
    from scipy .spatial import cKDTree 
    _ =key_ 
    return N [cKDTree (V ).query (_np .asarray (P ,float ))[1 ]]


def _halka_normal (V ,F ,P ,ic_r =3.0 ,dis_r =8.0 ):
    """AGIZ CEVRESINDEKI YUZUN normali -- "disari" for DOGRU referans.

    DIAGNOSIS (Bolum 21.32): most yakin tepenin normali aslinda deligin DUVAR
    normalidir and eksene DIKTIR (prediction yonuyle arasindaki angle median
    88.9 derece). O referansla sign atamak rastgeleye yakindir; this
    yuzden `disari_normal` kolu −0.0266 verdi.

    Dogrusu: CP'nin ETRAFINDAKI HALKADAN (ic_r with dis_r arasi) normal
    ortalamasi. Bu halka hole duvarinin DISINDA, duz yuzeydedir.
    """
    import numpy as _np 
    if not len (P ):
        return _np .zeros ((0 ,3 ))
    try :
        import trimesh 
        from scipy .spatial import cKDTree 
        ag =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
        N =_np .asarray (ag .vertex_normals ,float )
    except Exception :# noqa: BLE001
        return _np .zeros ((len (P ),3 ))
    agac =cKDTree (V )
    out =_np .zeros ((len (P ),3 ))
    for i ,p in enumerate (_np .asarray (P ,float )):
        dis_k =agac .query_ball_point (p ,dis_r )
        if not dis_k :
            continue 
        d =_np .linalg .norm (V [dis_k ]-p [None ,:],axis =1 )
        halka =_np .asarray (dis_k )[d >=ic_r ]
        if not len (halka ):
            halka =_np .asarray (dis_k )
        v =N [halka ].mean (0 )
        n =_np .linalg .norm (v )
        out [i ]=v /n if n >1e-9 else 0.0 
    return out 


def _pd (cps ):
    if not cps :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    P =np .asarray ([c ["point"]for c in cps ],float ).reshape (-1 ,3 )
    D =np .asarray ([c ["direction"]for c in cps ],float ).reshape (-1 ,3 )
    return P ,D 


def main ():
    t0 =time .time ()
    import torch 
    import canonical_chain 
    import robot_cp 
    import thesis_remesh 
    from infer_step_cp import load_any 
    import diffusionnet as D_ 
    import connector3d 

    cfg =json .load (open ("cp_config.json"))
    ca =float (cfg .get ("robot_conf_auto",0.5 ))
    mav =int (cfg .get ("robot_min_auto_votes",3 ))
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    # CHECK NOKTASI SECIMI (2026-08-13). `EZ_CKPT` virgulle ayrilmis path
    # listesi takes; verilmezse `cp_config.robot_vote2_checkpoints`. Boylece
    # new a segmentasyon kontrol noktasi (orn. Y1 augmentasyonlu) TAM
    # ZINCIRDE, single degiskenli olculebilir. Kanonik dersi: seg IoU kazanci
    # robot F1 kazanci DEGILDIR, dagitilacak yolda olculmelidir.
    _ck =os .environ .get ("EZ_CKPT","")
    ck_list =([x .strip ()for x in _ck .split (",")if x .strip ()]if _ck 
    else cfg ["robot_vote2_checkpoints"])
    print (f"kontrol noktalari ({len (ck_list )}): "
    f"{[os .path .basename (x )for x in ck_list ]}",flush =True )
    modeller =[load_any (c ,dev =dev )[:2 ]for c in ck_list ]
    CE ,CT =int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )
    STEP =K .step_map ()

    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    # PARCA LISTESI (2026-08-13). `EZ_LISTE` verilirse brand suzgeci instead of
    # OPEN part listesi is used. Sunulacak "tanidik brand" count for
    # gereken cluster budur: `results/split3.json` -> VAL (100 part,
    # geometri-ayrik, manufacturer-KARISIK) and measured ki VAL parcalarinin
    # HICBIRI secicinin training kumesinde DEGIL (DEV'de 37 tanesi vardi --
    # DEV that is why KULLANILMAZ).
    lst_ =os .environ .get ("EZ_LISTE","")
    if lst_ :
        import json as _j 
        _s =_j .load (open (lst_ ,encoding ="utf-8"))
        istenen =set (_s [os .environ .get ("EZ_BOLME","val")]["parts"])
        candidate =[(p ,r )for p ,r in kay .items ()
        if str (p )in istenen and len (r .get ("G",[]))and p in STEP ]
        print (f"part listesi: {lst_ } / "
        f"{os .environ .get ('EZ_BOLME','val')} -> {len (candidate )} part "
        f"(STEP'i ve GT'si olan)",flush =True )
    else :
        candidate =[(p ,r )for p ,r in kay .items ()
        if r .get ("mfg")in MARKALAR and len (r .get ("G",[]))
        and p in STEP ]
    rng =np .random .default_rng (0 )
    if len (candidate )>N_PARCA :
        candidate =[candidate [i ]for i in rng .choice (len (candidate ),N_PARCA ,
        replace =False )]
    print (f"{len (candidate )} part | STEP'ten TAM zincir, iki yol yan yana",
    flush =True )

    agg ={k :collections .Counter ()for k in ("saha","olculen")}
    dokum =[]
    n =0 
    for pid ,r in candidate :
        try :
            Vr ,Fr =__import__ ("export_robot_glb").step_to_mesh (STEP [pid ])
            # Y24: MESH-GRAFI UZERINDE OLASILIK DUZELTME (mean-alan CRF
            # yaklasimi). Kenar komsuluklari on birkac kind agirlikli
            # mean, tekil wrong etiketli tepeleri bastirir. Segmentasyon
            # kalitesi YANAL hataya baglandigi for (Bolum 21.46 baglayici
            # kisit) this arm dogrudan darbogaza nisan aliyor.
            # "kind,weight" biciminde; empty = closed (bit-same baseline).
            # Y16: remesh hedefi cevreden ayarlanabilir. Remesh KENDISI
            # a noise kaynagidir; different hedefler different ucgenleme and
            # different prediction produces. Varsayilan 6000 = mevcut davranis.
            _hedef =int (os .environ .get ("EZ_REMESH","6000"))
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =_hedef )
            V =np .ascontiguousarray (V ,np .float64 )
            F =np .ascontiguousarray (F ,np .int64 )
            # Y22: MC dropout. 0 = closed (bit-same baseline davranis).
            _mc =int (os .environ .get ("EZ_MCDROP","0"))
            pbs =[]
            for model ,meta in modeller :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                return_probs =True ,mc_dropout =_mc )
                pbs .append (np .asarray (pb ,float ))
            _crf =os .environ .get ("EZ_CRF","").strip ()
            if _crf :
                _t ,_w =(float (x )for x in _crf .split (","))
                pbs =[_graf_duzelt (pb ,F ,int (_t ),_w )for pb in pbs ]
                # HAM ADAY HAVUZU (2026-08-13). Hata otopsisi kacanlarin
                # %94'unun yakininda HIC prediction olmadigini showed. Ama this
                # "havuzda candidate YOK" mu, "candidate VARDI gate eledi" mi -- ayrimi
                # tahminle gecmek wrong becomes. Havuz here, gate'ten ONCE
                # kaydedilir; so HAVUZ RECALL'u with CIKTI recall'u yan yana
                # okunur and kalan emegin havuza mi skora mi gitmesi gerektigi
                # OLCUMLE belli becomes.
                # DUZELTME 2026-08-14: pool DISARIDAN yeniden uretilemez.
                # `extract` operator onbellegini kullanir, buradaki taze inference
                # kullanmaz -> olasiliklar different, pool different. Kanit: disaridan
                # uretilen pool ciktinin UST KUMESI bile degildi (84 GT ciktida
                # present/havuzda absent -- yapisal as imkansiz). Havuz residual
                # `extract`in ICINDEN, gate'ten hemen before yakalanir.
            del robot_cp .HAVUZ_KANCA [:]
            import wire_gate as _WG 
            del _WG .POSE_KANCA [:]
            # SAHA yolu: ihracatcilarin bugun cagirdigi
            saha =robot_cp .extract (modeller ,STEP [pid ],dev ,ca ,mav )
            _poz =list (_WG .POSE_KANCA )
            _hav =(robot_cp .HAVUZ_KANCA [-1 ]if robot_cp .HAVUZ_KANCA 
            else None )
            # OLCULEN path: kampanyanin olctugu
            olc =canonical_chain .product_output (V ,F ,pbs ,STEP [pid ],cfg )
        except Exception as e :# noqa: BLE001
            print (f"  {pid }: {type (e ).__name__ }: {e }",flush =True )
            continue 
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (np .linalg .norm (V .max (0 )-V .min (0 )))
        for ad ,cps in (("saha",saha ),("olculen",olc )):
            P ,D =_pd (cps )
            c =agg [ad ]
            # UC METRIK BIRDEN. `headline.py` manseti IKI ayri olcutle
            # hesapliyor and yapilandirmadaki `robot_hazir_F1` ISARETSIZ
            # olandir (`esle(..., 2.0, 10.0, False)` -- `signed`
            # varsayilani False), i.e. 180 derece TERS a direction DOGRU
            # sayilir. Robot for correct criterion ISARETLI olandir.
            # Ikisi de raporlanir ki sunulan sayinin hangisi oldugu
            # ASLA ambiguous kalmasin.
            for label_ ,im in (("robot_isaretli",True ),
            ("robot_isaretsiz",False )):
                tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,
                False ,signed =im )[:3 ]
                c [label_ +"_tp"]+=tp 
                c [label_ +"_fp"]+=fp 
                c [label_ +"_fn"]+=fn 
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,0.0 ,180.0 ,True )[:3 ]
            c ["tespit_tp"]+=tp ;c ["tespit_fp"]+=fp ;c ["tespit_fn"]+=fn 
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
            signed =True )[:3 ]
            c ["tp"]+=tp ;c ["fp"]+=fp ;c ["fn"]+=fn 
            c ["cp"]+=len (P )
            # TAHMIN DOKUMU (2026-08-13). Her threshold/count denemesi bugun 27
            # dakikalik TAM ZINCIR kosusu gerektiriyor. Tahminler diske
            # dokulurse same denemeler SANIYELER inside cevrimdisi is done.
            # Metrikleri DEGISTIRMEZ, only output adds.
        for ad ,cps in (("saha",saha ),("olculen",olc )):
            P ,D =_pd (cps )
            dokum .append ({
            "pid":str (pid ),"yol":ad ,
            "P":P .tolist (),"D":D .tolist (),
            "confidence":[float (c .get ("confidence",float ("nan")))
            for c in cps ],
            "votes":[float (c .get ("_votes",float ("nan")))
            for c in cps ],
            "G":G .tolist (),"Gd":Gd .tolist (),"diag":float (dg ),
            # GERCEK GOVDE BILGISI (2026-08-13). Ilk dokum only
            # tahminleri tasiyordu; "disari bak" kuralini denerken
            # body merkezi instead of TAHMIN EDILEN CP'lerin ortalamasini
            # vekil kullanmak zorunda kaldim and arm haksiz yere dustu
            # (-0.0749). Kahin sign kolu tavanin +0.0142 oldugunu
            # showed, i.e. mekanizma not VEKIL kotuydu. Artik mesh
            # merkezi and each tahminin YEREL YUZEY NORMALI dokuluyor.
            "havuz_P":(_hav ["P"]if _hav else []),
            "havuz_D":(_hav ["D"]if _hav else []),
            # POSE KIRPMA TESHISI: kirpma ONCESI yer degistirme.
            # Tek kosudan each `maks_mm` degeri yeniden kurulabilir.
            "pose_p0":[k ["p0"]for k in _poz ],
            "pose_dw":[k ["dw"]for k in _poz ],
            "pose_dir":[k ["dir"]for k in _poz ],
            "mesh_merkez":V .mean (0 ).tolist (),
            "yerel_normal":_yerel_normal (V ,F ,P ).tolist (),
            "halka_normal":_halka_normal (V ,F ,P ).tolist ()})
        agg ["saha"]["gt"]+=len (G )
        n +=1 
        if n %5 ==0 :
            print (f"  {n }/{len (candidate )} ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{n } part | GT {agg ['saha']['gt']}")
    print ("\n=== UC METRIK (same parts, same zincir) ===")
    print (f"{'yol':<10}{'tespit':>10}{'robot ISARETSIZ':>18}"
    f"{'robot ISARETLI':>17}")
    for _ad in ("saha","olculen"):
        _c =agg [_ad ]
        def _f (on ,_c =_c ):
            return (2 *_c [on +"_tp"]/max (
            2 *_c [on +"_tp"]+_c [on +"_fp"]+_c [on +"_fn"],1 ))
        print (f"{_ad :<10}{_f ('tespit'):>10.4f}"
        f"{_f ('robot_isaretsiz'):>18.4f}"
        f"{_f ('robot_isaretli'):>17.4f}")
    print ("  NOT: yapilandirmadaki `robot_hazir_F1` ISARETSIZ olandir;")
    print ("       robot for gecerli criterion ISARETLI olandir.")
    print (f"{'yol':<10}{'robot F1':>10}{'precision':>10}{'recall':>9}"
    f"{'uretilen CP':>13}")
    out ={}
    for ad in ("saha","olculen"):
        c =agg [ad ]
        f1 =2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )
        kes =c ["tp"]/max (c ["tp"]+c ["fp"],1 )
        rec =c ["tp"]/max (c ["tp"]+c ["fn"],1 )
        out [ad ]={"f1":f1 ,"precision":kes ,"recall":rec ,
        "cp":int (c ["cp"]),"tp":int (c ["tp"]),
        "fp":int (c ["fp"]),"fn":int (c ["fn"])}
        print (f"{ad :<10}{f1 :>10.4f}{kes :>10.4f}{rec :>9.4f}{c ['cp']:>13}")
    df1 =out ["olculen"]["f1"]-out ["saha"]["f1"]
    dke =out ["olculen"]["precision"]-out ["saha"]["precision"]
    print (f"\n  F1 farki       {df1 :+.4f}")
    print (f"  precision farki {dke :+.4f}")
    ac =df1 >0 and dke >=-0.02 
    print (f"\nBAYRAK KARARI: {'ACILABILIR'if ac else 'ACILMAZ'}")
    print ("  Kural: F1 ARTTI **VE** precision 0.02'den extra GERILEMEDI.")
    print ("  Gerekce: wrong CP = robotun empty yere hareketi; F1 same kalsa")
    print ("           bile precision dususu sahada GERILEMEDIR.")
    json .dump ({"damga":makbuz_hash .damga (),"n_parca":n ,
    "gt":int (agg ["saha"]["gt"]),"yollar":out ,
    "f1_farki":df1 ,"kesinlik_farki":dke ,"acilabilir":bool (ac ),
    "not":"Saha zinciri (robot_cp.extract) vs olculen zincir "
    "(canonical_chain.product_output), AYNI parcalarda. "
    "D7'ye BAKILMADI."},
    open ("results/zincir_esli_kiyas.json","w"),indent =1 )
    _dk =os .environ .get ("EZ_DOKUM","results/_tahmin_dokumu.json")
    json .dump (dokum ,open (_dk ,"w"),indent =0 )
    print (f"tahmin dokumu -> {_dk } ({len (dokum )} kayit)")
    print (f"receipt -> results/zincir_esli_kiyas.json "
    f"({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
