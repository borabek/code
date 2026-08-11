# -*- coding: utf-8 -*-
"""OLCULEN urun ile DAGITILAN urun ayni kalmali.

2026-07-31 denetimi uc SESSIZ ayrisma buldu ve ucu de sayilari gecersiz kiliyordu:
  1) robot_cp'nin dusuk-CP turetmesi `step_path` ALMIYORDU -> B-rep ekseni gercek uründe
     parcalarin ~%89.5'inde hic calismiyordu, oysa olcum onunla yapiliyordu.
  2) `_votes` benzersiz MODEL saymiyordu: ayni modelin yakin iki adayi iki oy yaziyordu,
     4 checkpoint varken ciktida votes=5 gorulduyordu. Bu ozellikle agir, cunku durust
     (geometri) bolmede gate'in genellesen TEK ozelligi votes.
  3) Olcum esikleri ile dagitilan esikler farkliydi.
Bu testler o ucunu de kilitler.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BA_ALLOW_SEEN", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_her_connection_points_cagrisi_step_path_aliyor():
    src = open(os.path.join(HERE, "robot_cp.py"), encoding="utf-8").read()
    calls = re.findall(r"cp_openings\.connection_points\((?:[^()]|\([^()]*\))*\)", src, re.S)
    assert calls, "cagri bulunamadi -- desen bozulmus olabilir"
    missing = [i for i, t in enumerate(calls, 1) if "step_path" not in t]
    assert not missing, (
        f"{missing} numarali cagri(lar) step_path almiyor -> B-rep ekseni o yolda CALISMAZ "
        f"ama olcum onunla yapilir; olculen urun dagitilan urun olmaz")


def _cp(x, conf):
    return {"point": np.array([float(x), 0.0, 0.0]), "direction": np.array([0.0, 0.0, 1.0]),
            "confidence": conf, "source_label": 3, "n_verts": 10, "area": 1.0,
            "insertion_depth_mm": 1.0}


def test_ayni_model_iki_oy_veremez():
    import robot_cp
    out = robot_cp._vote2([[_cp(0, 0.9), _cp(1, 0.8)]], cluster_mm=5.0, min_votes=1)
    assert len(out) == 1
    assert out[0]["_votes"] == 1, "tek modelden gelen iki yakin aday IKI oy sayilmis"


def test_farkli_modeller_ayri_oy_sayilir():
    import robot_cp
    out = robot_cp._vote2([[_cp(0, 0.9)], [_cp(0.5, 0.8)], [_cp(0.2, 0.7)]],
                          cluster_mm=5.0, min_votes=1)
    assert len(out) == 1
    assert out[0]["_votes"] == 3


def test_votes_checkpoint_sayisini_asamaz():
    import json, robot_cp
    cfg = json.load(open(os.path.join(HERE, "cp_config.json"), encoding="utf-8"))
    n = len(cfg["current_product"]["checkpoints"])
    lists = [[_cp(i * 0.3, 0.9 - 0.01 * i) for i in range(3)] for _ in range(n)]
    out = robot_cp._vote2(lists, cluster_mm=5.0, min_votes=1)
    for c in out:
        assert c["_votes"] <= n, f"votes={c['_votes']} > checkpoint sayisi {n}"


def test_konum_agirlikli_ortalama_uygulaniyor():
    """J: anlasan uyelerin konumu guven-agirlikli ortalanir (temsilcininki DEGIL)."""
    import robot_cp
    out = robot_cp._vote2([[_cp(0.0, 0.9)], [_cp(2.0, 0.3)]], cluster_mm=5.0, min_votes=1)
    x = float(out[0]["point"][0])
    assert 0.0 < x < 2.0, f"konum ortalanmamis (x={x})"
    assert abs(x - (0.0 * 0.75 + 2.0 * 0.25)) < 1e-6, f"agirliklar yanlis (x={x})"


def test_fiz_bayragi_ve_gate_genisligi_uyumlu():
    """URETIM YOLU: cp_config.gate_fiz_feats ile dagitilan gate'in sutun sayisi TUTMALI.

    Bayrak yalnizca ortam degiskeninden okunsaydi urun hattinda SESSIZCE kapali kalirdi
    (robot_cp WG_FIZ_FEATS set etmiyor) ve 18 sutunlu gate 13 sutunla beslenirdi.
    """
    import json, pickle, importlib, os
    import numpy as np
    os.environ.pop("WG_FIZ_FEATS", None)          # ortam ezmesi YOK: config karar versin
    import wire_gate
    importlib.reload(wire_gate)
    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    with open("results/wire_gate.pkl", "rb") as f:
        m = pickle.load(f)
    nf = m.get("n_feat") or len(m.get("feat_names", []))
    assert wire_gate.USE_FIZ_FEATS == bool(cfg.get("gate_fiz_feats")), \
        "bayrak cp_config'ten okunmuyor"
    # PARCA-ICI DONUSUM (2026-08-01): model bir donusum tasiyorsa uretim genisligi ONUNLA
    # karsilastirilmali -- donusum sutun sayisini ikiye katlar. Ham sayiyi karsilastirmak
    # dagitilan modeli SAHTE olarak "uyumsuz" gosterirdi; donusumu atlamak ise gercek bir
    # uyumsuzlugu KACIRIRDI. Bu yuzden gercek fonksiyon uzerinden olculuyor.
    ham = len(wire_gate.FEAT_NAMES)
    uretim = wire_gate.parca_ici(np.zeros((3, ham)), m.get("donusum")).shape[1]
    assert uretim == nf, \
        (f"uretim {ham} ham sutun -> donusum({m.get('donusum')!r}) -> {uretim} sutun, "
         f"gate {nf} bekliyor")


def test_fiz_ozellikleri_step_path_yokken_notr():
    """step_path verilmezse 5 fiziksel sutun SIFIR olmali -- eski cagrilar bozulmasin."""
    import os, importlib
    os.environ["WG_FIZ_FEATS"] = "1"
    import wire_gate
    importlib.reload(wire_gate)
    import numpy as np
    V = np.random.rand(40, 3) * 10.0
    F = np.array([[0, 1, 2], [1, 2, 3]])
    probs = np.ones((40, 5)) / 5.0
    cps = [{"point": np.zeros(3), "direction": np.array([0.0, 0.0, 1.0]),
            "area_mm2": 4.0, "_votes": 2, "confidence": 0.5}]
    X = wire_gate.feats_for(V, F, probs, cps, 3, 1, step_path=None)
    # GENISLIKTEN BAGIMSIZ: baska bloklar (topoloji) acik olabilir. Onemli olan FIZ blogunun
    # kendi dilimi -- 13 taban sutundan sonraki 5 sutun. Genislige sabitlenmis eski iddia,
    # topoloji blogu acilinca kirilmisti (2026-08-01).
    i = wire_gate.FEAT_NAMES.index(wire_gate.FEAT_NAMES_FIZ[0])
    fiz = X[:, i:i + len(wire_gate.FEAT_NAMES_FIZ)]
    assert fiz.shape[1] == 5
    assert (fiz == 0).all(), f"step_path yokken fiziksel sutunlar notr degil: {fiz}"
    os.environ.pop("WG_FIZ_FEATS", None)
    importlib.reload(wire_gate)


def test_sessiz_notrlesme_sayiliyor():
    """Ozellik bloklari hata halinde NOTR doner -- bu DOGRU, ama SESSIZ olmasi YANLIS.

    Bu projede bir kez yasandi: rtree kurulu olmadigi icin trimesh contains()/ray her cagride
    patliyordu, `except: continue` yutuyordu ve iki fonksiyon SESSIZ NO-OP'a donusmustu.
    Cikti "makul" gorundugu icin aylarca fark edilmedi. Artik her notr yol sayiliyor.
    """
    import os, importlib
    import numpy as np
    os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_EK_FEATS"] = "1"
    import wire_gate
    importlib.reload(wire_gate)
    wire_gate.FALLBACK.clear()
    V = np.random.rand(40, 3) * 10.0
    F = np.array([[0, 1, 2], [1, 2, 3]])
    probs = np.ones((40, 5)) / 5.0
    cps = [{"point": np.zeros(3), "direction": np.array([0.0, 0.0, 1.0]),
            "area_mm2": 4.0, "_votes": 2, "confidence": 0.5}]
    wire_gate.feats_for(V, F, probs, cps, 3, 1, step_path=None)   # step_path YOK -> notr
    ozet = wire_gate.fallback_ozet()
    assert ozet.get("fiz:step_path_yok") == 1, ozet
    assert ozet.get("ek:step_path_yok") == 1, ozet
    assert wire_gate.fallback_ozet() == {}, "ozet cagrisi sayaci sifirlamiyor"
    os.environ.pop("WG_FIZ_FEATS", None); os.environ.pop("WG_EK_FEATS", None)
    importlib.reload(wire_gate)


def test_goreli_esik_config_ile_acilir_ve_taban_calisir():
    """GORELI ESIK: parca-ici goreli karar + mutlak taban.

    Iki sey kilitleniyor:
      1) bayrak cp_config'ten okunuyor (ortam degiskeni yalnizca deney icin ezer),
      2) TABAN calisiyor -- tabansiz kural, hic gercek CP olmayan bir parcada bile
         "en yuksegin yarisi"ni kabul edip garanti yanlis uretirdi.
    """
    import json, importlib, os
    os.environ.pop("WG_GORELI", None)
    import wire_gate
    importlib.reload(wire_gate)
    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    assert wire_gate.GORELI_ESIK == bool(cfg.get("gate_goreli_esik")), "bayrak config'ten gelmiyor"
    if not wire_gate.GORELI_ESIK:
        return
    oran, taban = wire_gate.GORELI_ORAN, wire_gate.GORELI_TABAN
    # tum skorlari ZAYIF bir parca: en yuksek 0.2 ise hicbiri tabani gecemez -> HICBIRI secilmez
    zayif = [0.20, 0.12, 0.05]
    secili = [s for s in zayif if s >= oran * max(zayif) and s >= taban]
    assert secili == [], f"taban calismiyor: {secili}"
    # normal parca: en yuksek 0.8 -> 0.4 uzerindekiler secilir
    guclu = [0.80, 0.45, 0.30, 0.10]
    secili = [s for s in guclu if s >= oran * max(guclu) and s >= taban]
    assert secili == [0.80, 0.45], secili


def test_topo_blogu_mesh_tabanli_ve_step_path_gerektirmez():
    """TOPOLOJI blogu MESH tabanlidir: `step_path` olmadan da GERCEK deger uretir.

    Bu, blogun varlik sebebi: renk/B-rep yollarinda kapsama %33-65'te kaliyordu, topoloji
    korpusun %93.7'sinde dolu. Test, blogun sessizce sifira dusmedigini kilitler.
    """
    import os, importlib
    import numpy as np
    import trimesh
    os.environ["WG_TOPO"] = "1"
    import wire_gate
    importlib.reload(wire_gate)
    if not wire_gate.USE_TOPO_FEATS:
        return
    # delikli plaka: icbukey kenarlar VARDIR
    plaka = trimesh.creation.box(extents=(20, 20, 4))
    delik = trimesh.creation.cylinder(radius=3, height=10)
    m = plaka.difference(delik)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, int)
    probs = np.ones((len(V), 5)) / 5.0
    cps = [{"point": np.array([0.0, 0.0, 2.0]), "direction": np.array([0.0, 0.0, 1.0]),
            "area_mm2": 28.0, "_votes": 2, "confidence": 0.5}]
    X = wire_gate.feats_for(V, F, probs, cps, 3, 1, step_path=None)   # step_path YOK
    i = wire_gate.FEAT_NAMES.index(wire_gate.FEAT_NAMES_TOPO[0])
    topo = X[:, i:i + 4]
    assert topo.shape[1] == 4
    assert not (topo == 0).all(), f"topoloji step_path olmadan sifira dustu: {topo}"
    os.environ.pop("WG_TOPO", None)
    importlib.reload(wire_gate)
