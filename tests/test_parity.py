# -*- coding: utf-8 -*-
"""OLCULEN urun with DAGITILAN urun same kalmali.

2026-07-31 denetimi three SESSIZ ayrisma buldu and ucu de sayilari invalid kiliyordu:
  1) robot_cp'nin low-CP turetmesi `step_path` ALMIYORDU -> B-rep ekseni real uründe
     parcalarin ~%89.5'inde never calismiyordu, oysa measurement onunla yapiliyordu.
  2) `_votes` benzersiz MODEL saymiyordu: same modelin yakin two adayi two oy yaziyordu,
     4 checkpoint varken ciktida votes=5 gorulduyordu. Bu ozellikle agir, because durust
     (geometri) bolmede gate'in genellesen TEK ozelligi votes.
  3) Olcum esikleri with dagitilan esikler farkliydi.
Bu testler that ucunu de kilitler.
"""
import os ,re ,sys 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
os .environ .setdefault ("BA_ALLOW_SEEN","1")
HERE =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))


def test_her_connection_points_cagrisi_step_path_aliyor ():
    src =open (os .path .join (HERE ,"robot_cp.py"),encoding ="utf-8").read ()
    calls =re .findall (r"cp_openings\.connection_points\((?:[^()]|\([^()]*\))*\)",src ,re .S )
    assert calls ,"cagri bulunamadi -- desen bozulmus olabilir"
    missing =[i for i ,t in enumerate (calls ,1 )if "step_path"not in t ]
    assert not missing ,(
    f"{missing } numarali cagri(lar) step_path almiyor -> B-rep ekseni o yolda CALISMAZ "
    f"but measurement onunla yapilir; measured_path urun dagitilan urun olmaz")


def _cp (x ,conf ):
    return {"point":np .array ([float (x ),0.0 ,0.0 ]),"direction":np .array ([0.0 ,0.0 ,1.0 ]),
    "confidence":conf ,"source_label":3 ,"n_verts":10 ,"area":1.0 ,
    "insertion_depth_mm":1.0 }


def test_ayni_model_iki_oy_veremez ():
    import robot_cp 
    out =robot_cp ._vote2 ([[_cp (0 ,0.9 ),_cp (1 ,0.8 )]],cluster_mm =5.0 ,min_votes =1 )
    assert len (out )==1 
    assert out [0 ]["_votes"]==1 ,"tek modelden gelen iki yakin candidate IKI oy sayilmis"


def test_farkli_modeller_ayri_oy_sayilir ():
    import robot_cp 
    out =robot_cp ._vote2 ([[_cp (0 ,0.9 )],[_cp (0.5 ,0.8 )],[_cp (0.2 ,0.7 )]],
    cluster_mm =5.0 ,min_votes =1 )
    assert len (out )==1 
    assert out [0 ]["_votes"]==3 


def test_votes_checkpoint_sayisini_asamaz ():
    import json ,robot_cp 
    cfg =json .load (open (os .path .join (HERE ,"cp_config.json"),encoding ="utf-8"))
    n =len (cfg ["current_product"]["checkpoints"])
    lists =[[_cp (i *0.3 ,0.9 -0.01 *i )for i in range (3 )]for _ in range (n )]
    out =robot_cp ._vote2 (lists ,cluster_mm =5.0 ,min_votes =1 )
    for c in out :
        assert c ["_votes"]<=n ,f"votes={c ['_votes']} > checkpoint sayisi {n }"


def test_konum_agirlikli_ortalama_uygulaniyor ():
    """J: anlasan uyelerin konumu confidence-agirlikli ortalanir (temsilcininki DEGIL)."""
    import robot_cp 
    out =robot_cp ._vote2 ([[_cp (0.0 ,0.9 )],[_cp (2.0 ,0.3 )]],cluster_mm =5.0 ,min_votes =1 )
    x =float (out [0 ]["point"][0 ])
    assert 0.0 <x <2.0 ,f"konum ortalanmamis (x={x })"
    assert abs (x -(0.0 *0.75 +2.0 *0.25 ))<1e-6 ,f"agirliklar wrong (x={x })"


def test_fiz_bayragi_ve_gate_genisligi_uyumlu ():
    """URETIM YOLU: cp_config.gate_fiz_feats with dagitilan gate'in column count TUTMALI.

    Bayrak only ortam degiskeninden okunsaydi urun hattinda SESSIZCE closed kalirdi
    (robot_cp WG_FIZ_FEATS set etmiyor) and 18 sutunlu gate 13 sutunla beslenirdi.
    """
    import json ,pickle ,importlib ,os 
    import numpy as np 
    os .environ .pop ("WG_FIZ_FEATS",None )# ortam ezmesi YOK: config karar versin
    import wire_gate 
    importlib .reload (wire_gate )
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    with open ("results/wire_gate.pkl","rb")as f :
        m =pickle .load (f )
    nf =m .get ("n_feat")or len (m .get ("feat_names",[]))
    assert wire_gate .USE_FIZ_FEATS ==bool (cfg .get ("gate_fiz_feats")),"bayrak cp_config'ten okunmuyor"
    # PARCA-ICI DONUSUM (2026-08-01): model a donusum tasiyorsa uretim genisligi ONUNLA
    # karsilastirilmali -- donusum column sayisini ikiye katlar. Ham sayiyi karsilastirmak
    # dagitilan modeli SAHTE as "uyumsuz" gosterirdi; donusumu atlamak whereas real a
    # uyumsuzlugu KACIRIRDI. Bu yuzden real fonksiyon uzerinden olculuyor.
    ham =len (wire_gate .FEAT_NAMES )
    uretim =wire_gate .within_part (np .zeros ((3 ,ham )),m .get ("donusum")).shape [1 ]
    assert uretim ==nf ,(f"uretim {ham } ham sutun -> donusum({m .get ('donusum')!r }) -> {uretim } sutun, "
    f"gate {nf } bekliyor")


def test_fiz_ozellikleri_step_path_yokken_notr ():
    """step_path verilmezse 5 fiziksel column SIFIR must be -- old cagrilar bozulmasin."""
    import os ,importlib 
    os .environ ["WG_FIZ_FEATS"]="1"
    import wire_gate 
    importlib .reload (wire_gate )
    import numpy as np 
    V =np .random .rand (40 ,3 )*10.0 
    F =np .array ([[0 ,1 ,2 ],[1 ,2 ,3 ]])
    probs =np .ones ((40 ,5 ))/5.0 
    cps =[{"point":np .zeros (3 ),"direction":np .array ([0.0 ,0.0 ,1.0 ]),
    "area_mm2":4.0 ,"_votes":2 ,"confidence":0.5 }]
    X =wire_gate .feats_for (V ,F ,probs ,cps ,3 ,1 ,step_path =None )
    # GENISLIKTEN BAGIMSIZ: baska bloklar (topoloji) open may be. Onemli which is FIZ blogunun
    # own dilimi -- 13 baseline sutundan sonraki 5 column. Genislige sabitlenmis old iddia,
    # topoloji blogu acilinca kirilmisti (2026-08-01).
    i =wire_gate .FEAT_NAMES .index (wire_gate .FEAT_NAMES_FIZ [0 ])
    fiz =X [:,i :i +len (wire_gate .FEAT_NAMES_FIZ )]
    assert fiz .shape [1 ]==5 
    assert (fiz ==0 ).all (),f"step_path yokken fiziksel sutunlar notr not: {fiz }"
    os .environ .pop ("WG_FIZ_FEATS",None )
    importlib .reload (wire_gate )


def test_sessiz_notrlesme_sayiliyor ():
    """Ozellik bloklari error halinde NOTR returns -- this DOGRU, but SESSIZ olmasi YANLIS.

    Bu projede a times yasandi: rtree kurulu olmadigi for trimesh contains()/ray each cagride
    patliyordu, `except: continue` yutuyordu and two fonksiyon SESSIZ NO-OP'a donusmustu.
    Cikti "makul" gorundugu for aylarca difference edilmedi. Artik each notr path sayiliyor.
    """
    import os ,importlib 
    import numpy as np 
    os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_EK_FEATS"]="1"
    import wire_gate 
    importlib .reload (wire_gate )
    wire_gate .FALLBACK .clear ()
    V =np .random .rand (40 ,3 )*10.0 
    F =np .array ([[0 ,1 ,2 ],[1 ,2 ,3 ]])
    probs =np .ones ((40 ,5 ))/5.0 
    cps =[{"point":np .zeros (3 ),"direction":np .array ([0.0 ,0.0 ,1.0 ]),
    "area_mm2":4.0 ,"_votes":2 ,"confidence":0.5 }]
    wire_gate .feats_for (V ,F ,probs ,cps ,3 ,1 ,step_path =None )# step_path YOK -> notr
    ozet =wire_gate .fallback_ozet ()
    assert ozet .get ("fiz:step_path_yok")==1 ,ozet 
    assert ozet .get ("ek:step_path_yok")==1 ,ozet 
    assert wire_gate .fallback_ozet ()=={},"ozet cagrisi sayaci sifirlamiyor"
    os .environ .pop ("WG_FIZ_FEATS",None );os .environ .pop ("WG_EK_FEATS",None )
    importlib .reload (wire_gate )


def test_goreli_esik_config_ile_acilir_ve_taban_calisir ():
    """GORELI ESIK: part-ici goreli karar + mutlak baseline.

    Iki sey kilitleniyor:
      1) bayrak cp_config'ten okunuyor (ortam degiskeni only deney for ezer),
      2) TABAN calisiyor -- tabansiz rule, never real CP olmayan a parcada bile
         "most yuksegin yarisi"ni kabul edip garanti wrong uretirdi.
    """
    import json ,importlib ,os 
    os .environ .pop ("WG_GORELI",None )
    import wire_gate 
    importlib .reload (wire_gate )
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    assert wire_gate .GORELI_ESIK ==bool (cfg .get ("gate_goreli_esik")),"bayrak config'ten gelmiyor"
    if not wire_gate .GORELI_ESIK :
        return 
    ratio ,baseline =wire_gate .GORELI_ORAN ,wire_gate .GORELI_TABAN 
    # tum skorlari ZAYIF a part: most high 0.2 whereas none of them tabani gecemez -> HICBIRI secilmez
    zayif =[0.20 ,0.12 ,0.05 ]
    secili =[s for s in zayif if s >=ratio *max (zayif )and s >=baseline ]
    assert secili ==[],f"baseline calismiyor: {secili }"
    # normal part: most high 0.8 -> 0.4 uzerindekiler secilir
    guclu =[0.80 ,0.45 ,0.30 ,0.10 ]
    secili =[s for s in guclu if s >=ratio *max (guclu )and s >=baseline ]
    assert secili ==[0.80 ,0.45 ],secili 


def test_topo_blogu_mesh_tabanli_ve_step_path_gerektirmez ():
    """TOPOLOJI blogu MESH tabanlidir: `step_path` olmadan da GERCEK value produces.

    Bu, blogun varlik sebebi: renk/B-rep yollarinda kapsama %33-65'te kaliyordu, topoloji
    korpusun %93.7'sinde full. Test, blogun sessizce sifira dusmedigini kilitler.
    """
    import os ,importlib 
    import numpy as np 
    import trimesh 
    os .environ ["WG_TOPO"]="1"
    import wire_gate 
    importlib .reload (wire_gate )
    if not wire_gate .USE_TOPO_FEATS :
        return 
        # delikli plaka: icbukey kenarlar VARDIR
    plaka =trimesh .creation .box (extents =(20 ,20 ,4 ))
    delik =trimesh .creation .cylinder (radius =3 ,height =10 )
    m =plaka .difference (delik )
    V =np .asarray (m .vertices ,float );F =np .asarray (m .faces ,int )
    probs =np .ones ((len (V ),5 ))/5.0 
    cps =[{"point":np .array ([0.0 ,0.0 ,2.0 ]),"direction":np .array ([0.0 ,0.0 ,1.0 ]),
    "area_mm2":28.0 ,"_votes":2 ,"confidence":0.5 }]
    X =wire_gate .feats_for (V ,F ,probs ,cps ,3 ,1 ,step_path =None )# step_path YOK
    i =wire_gate .FEAT_NAMES .index (wire_gate .FEAT_NAMES_TOPO [0 ])
    topo =X [:,i :i +4 ]
    assert topo .shape [1 ]==4 
    assert not (topo ==0 ).all (),f"topoloji step_path olmadan sifira dustu: {topo }"
    os .environ .pop ("WG_TOPO",None )
    importlib .reload (wire_gate )
