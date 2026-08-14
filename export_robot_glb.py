# -*- coding: utf-8 -*-
"""Robot CP'lerini DONDURULEBILIR GLB as cikar (Windows 3D Viewer'da double tikla, incele).

Gri mesh + each CP'de renkli kure (kirmizi=otonom, turuncu=review) + direction oku (silindir body). Kullanici
each CP'nin GERCEK a aciklikta mi durdugunu 3D'de dondurerek gorebilir. Uretici GT gerekmez.

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe export_robot_glb.py 3001381 3001475
"""
import os ,sys ,glob ,json 
import numpy as np ,torch ,trimesh 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import thesis_remesh ,robot_cp ,diffusionnet as D ,connector3d 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT );OP ="results/step_infer/ops"
dev ="cuda"if torch .cuda .is_available ()else "cpu"
# PARCA KIMLIGI TEK KAYNAKTAN (2026-08-07, duman testinde yakalandi).
# Eski hali `basename.split("_")[1]` idi and JSON sozlesmeli dosyalarda ("UPUN.016029_
# ElectricalTerminal_...stp") parcayi BULAMIYORDU -> "STEP absent" deyip CIZIM URETMIYORDU.
# Yani musteriye gosterilecek gorsel never olusmuyordu.
from korpus_kimlik import step_kimlik as _sk 
STEP ={_sk (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
OUTDIR ="results/robot_glb"
SADE =False 


def arrow (origin ,direction ,length ,r ):
    """silindir body + koni three, origin'den disari."""
    d =np .asarray (direction ,float );d =d /(np .linalg .norm (d )+1e-9 )
    parts =[]
    shaft =trimesh .creation .cylinder (radius =r *0.35 ,height =length *0.7 ,sections =12 )
    shaft .apply_translation ([0 ,0 ,length *0.35 ])
    head =trimesh .creation .cone (radius =r *0.9 ,height =length *0.35 ,sections =12 )
    head .apply_translation ([0 ,0 ,length *0.7 ])
    a =trimesh .util .concatenate ([shaft ,head ])
    # +Z'yi d yonune dondur
    z =np .array ([0 ,0 ,1.0 ]);v =np .cross (z ,d );s =np .linalg .norm (v );c =float (z @d )
    if s <1e-8 :
        Rm =np .eye (3 )if c >0 else np .diag ([1 ,-1 ,-1.0 ])
    else :
        vx =np .array ([[0 ,-v [2 ],v [1 ]],[v [2 ],0 ,-v [0 ]],[-v [1 ],v [0 ],0 ]])
        Rm =np .eye (3 )+vx +vx @vx *((1 -c )/(s *s ))
    T =np .eye (4 );T [:3 ,:3 ]=Rm ;T [:3 ,3 ]=origin 
    a .apply_transform (T )
    return a 


TEL_CAP =1.78 # 2.5mm^2 iletken capi (most yaygin klemens)
ILERI_MIN =5.0 # this mesafeden yakinda engel varsa "onu closed"


def _fiz_bayraklar (mesh ,p ,d ):
    """CP for fiziksel kusur bayraklari. Doner: [(name, [R,G,B]), ...]

    Aletler A0'da KALIBRE EDILDI: `is_inside` 26/26 (%0.0 yanilma), `mouth_width` %1 deviation.
    Olculemeyen a sey UYDURULMAZ -- alet 0 dondururse bayrak KONMAZ.
    """
    from cp_geometry import is_inside ,mouth_width ,ray_hits 
    p =np .asarray (p ,float )
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    out =[]
    ici =False 
    try :
        ici =bool (is_inside (mesh ,p ))
        if ici :
            out .append (("govde_ici",[255 ,0 ,0 ]))
    except Exception :
        pass 
        # SENTETIK KALIBRASYON ILK SURUMU CURUTTU (2026-08-05) -- two error birden:
        #
        #  1. `ters_yon` "p + 2mm*d ICERIDE mi" diye soruyordu. Nokta ZATEN body icindeyse
        #     this HER ZAMAN correct cikiyor -> `govde_ici`nin KOPYASI. Disaridaki GERCEK ters
        #     yonu whereas KACIRIYOR, because 2mm ilerlemek yuzeye ulasmaya yetmiyor.
        #     (kup merkezi -> ['govde_ici','ters_yon']; kup disi iceri-bakan -> [] )
        #  2. `onu_kapali` `exit_length` kullaniyordu and yuzeye 0.5mm'de bile atesLEMEDI --
        #     that fonksiyon baska a sey olcuyor.
        #
        # Duzeltilmis tanimlar, ikisi de DOGRUDAN isin mesafesiyle:
        #   ters_yon    : direction along ilerleyince govdeden CIKILAMIYOR (disaridayken first
        #                 carpma yakinda VE arkasi full) ya da noktadan geri gidince disari
        #                 cikiliyor -- i.e. vektor govdeye BAKIYOR
        #   onu_kapali  : ileride ILERI_MIN'den yakinda a surface present (tel giremez)
    try :
        ileri =ray_hits (mesh ,p ,d ,max_mm =60.0 )
        geri =ray_hits (mesh ,p ,-d ,max_mm =60.0 )
        ilk_ileri =float (ileri [0 ])if len (ileri )else float ("inf")
        if not ici :
        # Disaridaki point: direction govdeye bakiyorsa ilerde YUZEY present, geride YOK.
        # Yon disari bakiyorsa full tersi.
            if np .isfinite (ilk_ileri )and not len (geri ):
                out .append (("ters_yon",[0 ,204 ,255 ]))
            if ilk_ileri <ILERI_MIN :
                out .append (("onu_kapali",[255 ,140 ,0 ]))
        else :
        # Icerideki point: cikis mesafesi ILERI_MIN'den kisaysa already duvara yapisik
            if ilk_ileri <ILERI_MIN :
                out .append (("onu_kapali",[255 ,140 ,0 ]))
    except Exception :
        pass 
    try :
        ic ,_ort =mouth_width (mesh ,p ,d )
        if 0.0 <float (ic )<TEL_CAP :
            out .append (("dar_agiz",[255 ,255 ,0 ]))
    except Exception :
        pass 
    return out 


def main ():
# FIZIKSEL DEFECT ISARETCILERI VARSAYILAN OLARAK KAPALI (2026-08-05).
# GLB tamamen CIKTI katmanidir: F1 CP'nin konumundan/yonunden is computed, cizimden
# DEGIL. Yani this anahtar no sayiyi etkilemez -- only ne gordugunu belirler.
#   varsayilan : sade -- kirmizi/turuncu kure + ok (tier'i gosterir)
#   --fiz      : ustune fiziksel kusur isaretcileri (neden wrong oldugunu gosterir)
    argv =[a for a in sys .argv [1 :]if a not in ("--fiz","--sade")]
    FIZ ="--fiz"in sys .argv [1 :]
    # TEK RENK: tier ayrimini gosterme, only bulunan CP'leri ciz.
    global SADE 
    SADE =("--sade"in sys .argv [1 :]
    or os .environ .get ("CP_GLB_SADE","0")not in ("0","","false"))
    pids =argv or open ("_demo_parts.txt").read ().split ()
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    cfg =json .load (open ("cp_config.json"))
    ca =float (cfg .get ("robot_conf_auto",0.5 ));mav =int (cfg .get ("robot_min_auto_votes",3 ))
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .makedirs (OUTDIR ,exist_ok =True )
    for pid in pids :
        if pid not in STEP :
            print (f"  {pid }: STEP yok");continue 
        Vr ,Fr =step_to_mesh (STEP [pid ])
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        # modelin segmentasyonu (vote>=2 uyeleri mean) -> baglanti vertekslerini boya
        acc =None 
        pbs =[]# <- LISTE de saklaniyor (below gerekli)
        for model ,meta in models :
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
            pb =np .asarray (pb ,float );pbs .append (pb )
            acc =pb if acc is None else acc +pb 
        probs =acc /len (models );conn =probs [:,CE ]+probs [:,CT ]
        # OLCULEN ZINCIR (2026-08-12). `robot_cp.extract` KAMPANYANIN
        # KAZANCLARINI TASIMIYOR: `product_p6` (direction bankasi + ortak siralayici) and
        # `product_genis` only `canonical_chain.product_output` icinden cagriliyor
        # and ihracatcilarin HICBIRI orayi cagirmiyordu. Yani bugune up to
        # GLB'deki isaretler TABANIN ciktisiydi (D7 0.2980), olctugumuz 0.3115
        # not (rapor bolum 8).
        #
        # VARSAYILAN KAPALI: urunun bugunku ciktisi does not change. Acmak for
        # cp_config `glb_kanonik_zincir=true`. Acikken also CIFT CIKARIM da
        # kalkar (extract own inside same modelleri yeniden kosuyordu).
        if cfg .get ("glb_kanonik_zincir",False ):
            import canonical_chain 
            ham =canonical_chain .product_output (V ,F ,pbs ,STEP [pid ])
            # `product_output` tier ATAMAZ (that rule `extract` icindeydi, residual
            # ortak `assign_tier`). Atanmazsa asagidaki c["tier"] KeyError gives;
            # `_format_cps` hem kayitlari kurar hem tier'i atar.
            cps =robot_cp ._format_cps (ham ,ca ,mav )
            print (f"  {pid }: KANONIK ZINCIR ({len (cps )} CP)",flush =True )
        else :
            cps =robot_cp .extract (models ,STEP [pid ],dev ,ca ,mav )
            # HALKA NORMALI ILE ISARET DUZELTMESI (2026-08-14).
            # Her CP'nin direction ISARETINI mouth cevresi face normaliyle uyumlu yapar;
            # konumu and ekseni DEGISTIRMEZ, parametresi YOKTUR.
            # MEASURED (VAL 100, esli part bootstrap):
            #   TANIDIK brand / olculen zincir : +0.0195  GA [+0.0030,+0.0378] KESIN
            #   TANIDIK brand / saha yolu      : +0.0329  GA sifiri ICERIYOR
            #   hard-gorulmemis brand (150)     : +0.0023 / −0.0016  (notr)
            # VARSAYILAN KAPALI: kampanya along uygulanan disipline according to
            # "GA sifiri iceriyorsa dagitma". Kanit YALNIZ `olculen` zincirde and
            # TANIDIK brand populasyonunda conclusive; this an dagitilan path `saha`.
            # Acmak: cp_config `ring_sign=true`.
        if cfg .get ("ring_sign",False ):
            import ring_sign 
            cps ,_cev =ring_sign .duzelt (cps ,V ,F )
            print (f"  {pid }: halka sign duzeltmesi -> {_cev } CP cevrildi",
            flush =True )
        scene =trimesh .Scene ()
        ctr =V .mean (0 )
        # MESH: gri gövde; modelin "baglanti" dedigi verteksler MAVI (koyulugu olasilikla)
        vc =np .tile ([205 ,205 ,205 ,255 ],(len (V ),1 )).astype (np .uint8 )
        blue =conn >=0.5 
        vc [blue ]=np .stack ([np .full (blue .sum (),30 ),np .full (blue .sum (),90 ),
        (150 +105 *conn [blue ]).clip (0 ,255 ).astype (int ),
        np .full (blue .sum (),255 )],1 ).astype (np .uint8 )
        mesh_ham =trimesh .Trimesh (V ,F ,process =False )# KAYDIRILMAMIS: fizik olcumleri
        mesh =trimesh .Trimesh (V -ctr ,F ,process =False )
        mesh .visual .vertex_colors =vc 
        scene .add_geometry (mesh ,node_name ="mesh")
        ext =float ((V .max (0 )-V .min (0 )).max ())
        for i ,c in enumerate (cps ):
            p =np .asarray (c ["point"],float )-ctr 
            d =np .asarray (c ["direction"],float )
            # TEK RENK MODU (2026-08-14, `--sade` ya da CP_GLB_SADE=1).
            # Varsayilan gorunum tier'i renkle gosterir (kirmizi=otonom,
            # turuncu=review). Saha kullanimda "robot ne buldu" sorusuna
            # bakilirken this ayrim noise does; sade modda TUM CP'ler
            # same renkte cizilir. CP'LER ELENMEZ -- only renk ayrimi
            # kalkar, i.e. gosterilen bilgi AZALMAZ.
            col =([230 ,20 ,20 ,255 ]if SADE else 
            ([230 ,20 ,20 ,255 ]if c ["tier"]=="auto"
            else [245 ,150 ,20 ,255 ]))
            # KUCUK sabit kure (1.5mm) -- residual gomulmuyor, centre net
            ball =trimesh .creation .icosphere (subdivisions =3 ,radius =1.5 )
            ball .apply_translation (p );ball .visual .vertex_colors =np .tile (col ,(len (ball .vertices ),1 ))
            scene .add_geometry (ball ,node_name =f"cp{i }_ball")
            ar =arrow (p ,d ,ext *0.16 ,1.6 )
            ar .visual .vertex_colors =np .tile (col ,(len (ar .vertices ),1 ))
            scene .add_geometry (ar ,node_name =f"cp{i }_arrow")
            # --- M10: FIZIKSEL DEFECT ISARETCILERI (2026-08-05)
            # WHY: A4 olcumu -- wrong pozitifler fiziksel kusurda ZENGIN (body ici
            # 3.16x, dar mouth 2.18x, onu closed 1.85x). Bu F1'de gorunmuyor but GLB'ye
            # bakan insan for "this point neden wrong" sorusunun DOGRUDAN yaniti.
            # AYRI KANAL: ana kurenin RENGI does not change (that tier'i gosterir); kusur, ustune
            # binen YARI BOYUTTA ikinci kuredir. Boylece two bilgi same anda okunur.
            for _ad ,_rgb in (_fiz_bayraklar (mesh_ham ,np .asarray (c ["point"],float ),d )
            if FIZ else ()):
                bs =trimesh .creation .icosphere (subdivisions =2 ,radius =0.8 )
                bs .apply_translation (p )
                bs .visual .vertex_colors =np .tile (_rgb +[255 ],(len (bs .vertices ),1 ))
                scene .add_geometry (bs ,node_name =f"cp{i }_fiz_{_ad }")
        out =os .path .join (OUTDIR ,f"{pid }_robot.glb")
        scene .export (out )
        na =sum (1 for c in cps if c ["tier"]=="auto")
        _fs =[k for k in scene .geometry if "_fiz_"in k ]# --fiz kapaliyken empty
        import collections as _c 
        _fc =_c .Counter (k .split ("_fiz_")[1 ]for k in _fs )
        _tier =(""if SADE else 
        f" ({na } otonom kirmizi / {len (cps )-na } review turuncu)")
        print (f"  {pid }: {len (cps )} CP{_tier }"
        f" | FIZIKSEL KUSUR {len (_fs )}: {dict (_fc )if _fc else 'yok'}"
        f" -> {out }",flush =True )
    print (f"\n-> {OUTDIR }/  (cift tikla: Windows 3D Viewer)")


if __name__ =="__main__":
    main ()
