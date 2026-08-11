# -*- coding: utf-8 -*-
"""ROBOT GORSELLESTIRME -- ince terminallerde GERCEKTEN gorunen GLB.

NEDEN BU ARAC (2026-07-28 FBI):
  Onceki denemeler basarisizdi. Kok neden: trimesh GLB'yi MATERYALSIZ yaziyor (sadece COLON_0 vertex
  rengi) -> alphaMode tanimsiz -> Windows 3D Viewer govdeyi OPAK render ediyor. "Seffaf govde" hic
  seffaf olmadi ve isaretciler govdenin icinde kayboldu. Terminaller 8mm INCE, CP'ler tam ortasinda.

COZUM: seffafliga hic guvenme. Her CP icin govdenin ICINDEN GECEN ve IKI UCTAN TASAN bir IGNE ciz.
Hangi taraf disarida olursa olsun gorunur; goruntuleyiciden bagimsiz.

RENKLER:  YESIL igne = uretici CP (ground truth)
          KIRMIZI igne = robotun bulduğu
          MAVI ince cizgi = eslesme (dogru bulunmus)
          Yalniz YESIL = kacirilan (FN) | Yalniz KIRMIZI = fazladan (FP)

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe robot_viz.py <parca_no> [...]
"""
import os, sys, json
import numpy as np, trimesh
from cp_geometry import (seat_to_mouth, outward_along_axis, is_inside,   # ORTAK modul
                         exit_length, ray_hits, _reach)
from viz_contract import (RGBA, RECEIPT_VERSION, receipt_path,   # CIZIM SOZLESMESI (bkz .md)
                          MODES, ROBOT_ROLE, MAX_MARKER_OFFSET_MM)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")
OUT_DIR = "results/robot_glb"
_WRITTEN = []      # bu kosuda yazilan GLB'ler -- denetim TAM bunlari okur


class _NoReference(Exception):
    """Bu parcanin uretici JSON'u yok -- BEKLENEN durum, hata degil.

    Urun gorunumu gorulmemis parcada calisir; orada GT hic yoktur. Ayri bir tur olmasinin sebebi
    genel `except Exception` dalinin ekrana "(no manufacturer reference: TypeError)" gibi bir
    hata adi basmasi -- kullanici icin bu, bozulmus gibi okunuyordu.
    """


def greedy_match(P, G, Gd, tol, axis_tol=40.0):
    """big_arbiter ile AYNI eslesme: her GT ve her tahmin EN FAZLA BIR KEZ kullanilir."""
    pairs = []
    if not len(P) or not len(G):
        return pairs
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    perp = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    perp = np.where(np.abs(al) <= axis_tol, perp, np.inf)
    up, ug = set(), set()
    for d, a, b in sorted((perp[a, b], a, b) for a in range(len(P)) for b in range(len(G))):
        if d > tol or a in up or b in ug:
            continue
        up.add(a); ug.add(b); pairs.append((a, b, float(d), float(al[a, b])))
    return pairs


def pin(p, d, L, rad, col):
    """Noktadan DISARI dogru tek tarafli igne: taban noktada, uc govdenin disinda."""
    d = np.asarray(d, float); n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    c = trimesh.creation.cylinder(radius=rad, height=L, sections=14)
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d))
    c.apply_translation(np.asarray(p, float) + d * L / 2.0)   # taban p'de, uc p+d*L
    c.visual.vertex_colors = np.tile(col, (len(c.vertices), 1))
    return c


def ball(p, rad, col):
    s = trimesh.creation.uv_sphere(radius=rad, count=[12, 12])
    s.apply_translation(p)
    s.visual.vertex_colors = np.tile(col, (len(s.vertices), 1))
    return s


def cone(p, d, L, rad, col):
    """p'den d yonunde L boyunda konik ok ucu (taban p'de, sivri uc p+d*L)."""
    d = np.asarray(d, float); n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    c = trimesh.creation.cone(radius=rad, height=L, sections=16)
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d))
    c.apply_translation(np.asarray(p, float))
    c.visual.vertex_colors = np.tile(col, (len(c.vertices), 1))
    return c


def surface_along(body, p, d):
    """CP govdenin ICINDEYSE d yonunde ILK yuzey kesisimine tasi. Disaridaysa OYNATMA.

    Ok, CP'den baslayip govdeyi delerek cikarsa boyu CP'nin DERINLIGINE baglanir; oku yuzeyden
    baslatmak boyu parcayla oranti tutar.

    DIKKAT -- BU FONKSIYON BIR KEZ YANLIS YAZILDI (2026-07-29, olculdu): (a) 'iceride mi'
    kontrolu yoktu ve (b) EN UZAK kesisim (h[-1]) aliniyordu. Ikisi birlikte, yonu ters cikan
    isaretcileri butun blogu deldirip KARSI YUZEYE firlatti: 306 isaretcinin 148'i (%48)
    CP'sinden 3mm'den fazla koptu, en kotusu 88.9mm. Ekranda oklar parcadan ayri, havada duruyordu.

    Dogru konvansiyon zaten cp_geometry.seat_to_mouth'ta yaziliydi ("disaridaki nokta
    OYNATILMAZ; yoksa isin karsi duvara carpar"). Ayni kural burada da gecerli: ONCE iceride mi
    diye bak, sonra ILK kesisimi al.
    """
    p = np.asarray(p, float); d = np.asarray(d, float)
    if not is_inside(body, p):
        return p
    h = ray_hits(body, p, d, _reach(body, p))
    if not len(h):
        return p
    t = float(h[0])
    # SINIR: bir acikligin agzi CP'den birkac mm otededir. Ilk yuzey bundan uzaktaysa o yon bu
    # CP'ye ait degildir (yon yanlis cikmis) -- isaretciyi OYNATMA. Bu kap olmadan, uzun eksen
    # boyunca serbest bir kanala bakan isaretciler hala 20-89mm firliyordu (olculdu: 26/259).
    return p + d * t if t <= MAX_MARKER_OFFSET_MM else p


def escape_dir(body, p, d, arrow_len):
    """Isaretcinin GERCEKTEN disari cikabildigi yonu sec: once d, olmazsa -d.

    NEDEN: outward_along_axis bir ISARET seciyor ve bu isaret bazen ters cikiyor. Sonuc, okun
    govdenin ICINDE kalmasi (olculdu 2026-07-29: 4/19 uc iceride, 5 uctan ileride malzeme).
    Gorunmeyen bir isaretci, yanlis yere firlamis bir isaretci kadar kotudur.

    Cizim burada kendi kendinin testi olur: bir yon secilir, ucun govde disinda kalip kalmadigi
    OLCULUR, kalmiyorsa ters yon denenir. Ikisi de olmazsa d korunur (en azindan konum dogru
    kalir) -- ve denetci bunu M3/M5 ile zaten yakalar, sessizce gecmez.
    """
    p = np.asarray(p, float); d = np.asarray(d, float)
    for cand in (d, -d):
        s = surface_along(body, p, cand)
        tip = s + cand * arrow_len
        if not is_inside(body, tip) and len(ray_hits(body, tip, cand, _reach(body, tip))) == 0:
            return cand
    return d


def visualize(pid, models=None, dev=None, mode="robot_only"):
    """mode='robot_only' (VARSAYILAN, urun gorunumu): sadece robotun koydugu CP'ler, tek renk.
    mode='compare' (teshis): uretici ile karsilastirma renkleri + eslesme cizgileri."""
    import torch
    import thesis_remesh, robot_cp
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible

    if mode not in MODES:
        raise ValueError(f"bilinmeyen mod {mode!r}, gecerli: {MODES}")
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open("cp_config.json"))
    if models is None:
        models = [load_any(c, dev=dev)[:2] for c in cfg["current_product"]["checkpoints"]]
    # PARCA COZUMLEME. eligible() yalnizca UYGUN parcalari verir: uretici JSON'u olan VE sizinti
    # korumasindan gecen. Urun gorunumu (robot_only) icin bu yanlis kapi -- robot gorulmemis bir
    # parcada calisir, GT'si olmayanda da calismak ZORUNDA. 0270018 diskte duruyorken
    # "part not found" diyordu, cunku egitimde kullanildigi icin havuz disindaydi.
    # Once uygun havuz (varsa GT ile karsilastirma da yapilir), sonra DISKTEKI STEP.
    if str(pid).lower().endswith((".stp", ".step")) and os.path.isfile(pid):
        # DISARIDAN gelen STEP: internetten indirilmis, musteriden gelmis, herhangi bir dosya.
        # Urunun asil kullanim bicimi bu -- korpusa girmesi ya da parca numarasi olmasi gerekmez.
        stp = pid
        pid = os.path.splitext(os.path.basename(stp))[0]
        mfg = "EXT"
        jf = None
        import glob as _g
        jhit = (_g.glob(os.path.join(r"C:\Users\DE00024082\Desktop\JSON", f"*.{pid}.json"))
                or _g.glob(os.path.join("_ds1", "DataSet", f"*.{pid}_*.json")))
        if jhit:
            jf = jhit[0]
            mfg = os.path.basename(jf).split(".")[0]
        print(f"  (harici STEP: {os.path.basename(stp)}"
              + (", uretici JSON'u bulundu -- karsilastirma yapilacak" if jf
                 else ", uretici referansi YOK -- saf urun gorunumu") + ")")
        hit = None
    else:
        hit = [(m, p, jf, s) for m, p, jf, s in eligible() if p == pid]
    if hit:
        mfg, _, jf, stp = hit[0]
    elif hit is not None:
        import glob as _g
        cand = sorted(_g.glob(os.path.join("all_wscad_stp", f"*_{pid}_*.stp")))
        if not cand:
            print(f"{pid}: no STEP file found on disk (all_wscad_stp/*_{pid}_*.stp)")
            return None
        stp = cand[-1]
        jf = None
        jhit = (_g.glob(os.path.join(r"C:\Users\DE00024082\Desktop\JSON", f"*.{pid}.json"))
                or _g.glob(os.path.join("_ds1", "DataSet", f"*.{pid}_*.json")))
        if jhit:
            jf = jhit[0]
        mfg = os.path.basename(jf).split(".")[0] if jf else "STEP"
        print(f"  ({pid}: scoring pool disinda -- diskteki STEP ile URUN gorunumu"
              + (", uretici JSON'u bulundu" if jf else ", uretici referansi YOK") + ")")

    cps = robot_cp.extract(models, stp, dev, conf_auto=float(cfg.get("robot_conf_auto", 0.5)))

    Vr, Fr = step_to_mesh(stp)
    V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    P = np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3))
    Pd = np.array([c["direction"] for c in cps], float) if cps else np.zeros((0, 3))

    # URETICI CP'si OPSIYONEL: gercek kullanimda (gorulmemis parca) yoktur. Varsa konsolda
    # karsilastirma yazilir; CIZIME yalniz 'compare' modunda girer.
    Gm = np.zeros((0, 3)); Gdm = np.zeros((0, 3)); names = []
    try:
        if jf is None:
            raise _NoReference                 # yukarida zaten acikca yazildi, tekrar basma
        j = json.load(open(jf, encoding="utf-8-sig"))
        Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
        G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j["ConnectionPoints"]], float)
        Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]]
                       for c in j["ConnectionPoints"]], float)
        if len(G):
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            names = [c.get("Name") for c in j["ConnectionPoints"]]
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R; Gdm = Gd @ R               # uretici CP -> mesh frame
    except _NoReference:
        pass
    except Exception as e:
        print(f"   (no manufacturer reference: {type(e).__name__})")

    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tol = max(3.0, 0.06 * diag)
    pairs = greedy_match(P, Gm, Gdm, tol)
    tp = len(pairs); fp = len(P) - tp; fn = len(Gm) - tp
    prec = tp / max(len(P), 1); rec = tp / max(len(Gm), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9) if len(Gm) else float("nan")

    print(f"\n=== {mfg}.{pid} | {np.round(V.max(0)-V.min(0),1)} mm | mode={mode} ===")
    if len(Gm):
        print(f"  manufacturer CP {len(Gm)} | robot {len(P)} | TP {tp} FP {fp} FN {fn} -> F1 {f1:.3f}")
    else:
        print(f"  robot placed {len(P)} CP (no manufacturer reference for this part)")
    if mode == "compare" and len(Gm):
        mp = {a: (b, d, a2) for a, b, d, a2 in pairs}
        for i, c in enumerate(cps):
            if i in mp:
                b, d, a2 = mp[i]
                print(f"   pred{i+1} -> GT{b+1} ('{names[b]}') perp {d:.1f}mm, axial {a2:+.1f}mm  [CORRECT]")
            else:
                print(f"   pred{i+1} @ {np.round(P[i],1)}  [EXTRA]")
        for b in range(len(Gm)):
            if b not in {x[1] for x in pairs}:
                print(f"   GT{b+1} ('{names[b]}') @ {np.round(Gm[b],1)}  [MISSED]")
    else:
        for i in range(len(P)):
            c = cps[i] if cps else {}
            # Robotun gercekten ihtiyaci olan tam kayit: nerede, hangi yone, hangi genislige,
            # ne kadar derine, ve otonom mu yoksa insana mi sorulacak.
            print(f"   CP{i+1} @ {np.round(P[i],1)}  dir {np.round(Pd[i],2)}"
                  f"  size {c.get('size_mm',0):.2f}mm  depth {c.get('depth_mm',0):.1f}mm"
                  f"  conf {c.get('confidence',0):.2f}  [{c.get('tier','?')}]")

    # ---- GLB: igneler govdeden TASAR (dogrulanir) ----
    # Igne, deligin GERCEK ekseni boyunca disari bakar (bkz cp_geometry.outward_along_axis).
    # Boy olculu tutulur: parcadan uzun igne okunmaz hale getiriyordu; gerekirse asagida buyutulur.
    ext = V.max(0) - V.min(0)
    L = float(np.clip(0.14 * diag, 6.0, 28.0))
    rad = float(np.clip(0.010 * diag, 0.5, 1.6))
    body = trimesh.Trimesh(V, F, process=False)
    body.visual.vertex_colors = np.tile(RGBA["body"], (len(V), 1))
    # ROBOT-ONLY: uretici isaretcisi CIZILMEZ. Karsilastirma renkleri yalniz teshis modunda.
    draw_gt = (mode == "compare")
    # GT'yi YUVADAN AGIZA tasi (yoksa isaretci kati icinde kalir, insan 'delik yok' gorur)
    Gmouth = []
    if draw_gt:
        for g, gd in zip(Gm, Gdm):
            mth, off = seat_to_mouth(body, g, gd)
            Gmouth.append(mth)
    Gmouth = np.array(Gmouth) if Gmouth else np.zeros((0, 3))
    parts = [body]
    matched_g = {b for _, b, _, _ in pairs}; matched_p = {a for a, _, _, _ in pairs}

    # DISARI YONU: her isaretci icin BIR KEZ, gercek govdeye bakarak (bbox ile DEGIL -- kutu testi
    # noktanin parcadaki yerine gore isareti ters cevirip igneleri saga-sola savuruyordu).
    Gout = [outward_along_axis(body, g, gd) for g, gd in zip(Gmouth, Gdm)]
    Pout = [outward_along_axis(body, p_, d_) for p_, d_ in zip(P, Pd)]

    # IGNE BOYU: tahmin yok -- her isaretci icin gereken uzunluk OLCULUR (bkz exit_length).
    # Kisa igne govde icinde kaybolur, uzun igne parcayi ezer; ikisi de yasandi (2026-07-28).
    # OK GEOMETRISI (2026-07-29'da bastan yazildi):
    #  * ok GOVDENIN DISINDA baslar (yuzey cikis noktasi) -> boy artik CP derinligine bagli DEGIL
    #  * boy PARCAYA gore sabit -> bir parcadaki tum oklar ayni uzunlukta, duzenli gorunur
    #  * sekil gercek ok: silindir govde + konik uc (lolipop degil)
    # Her isaretci hala TAM 2 bagli bilesen (govde + uc), boylece denetcinin M1/M2/M6 sayimi aynen gecerli.
    L_ARROW = float(np.clip(0.11 * diag, 4.0, 12.0))
    marg = float(np.clip(0.04 * diag, 2.0, 6.0))       # uzatma gerekirse ucun disarida kalma payi
    R_SHAFT = float(np.clip(0.006 * diag, 0.28, 0.75))
    L_HEAD = 0.40 * L_ARROW
    parts_m, tips, markers = [], [], []

    def add_arrow(start, od, role, index, shaft_scale=1.0):
        od = escape_dir(body, start, od, L_ARROW)       # cikamiyorsa isareti duzelt
        s = surface_along(body, start, od)              # gorunur ok YUZEYDEN baslar
        # BOY: varsayilan SABIT (duzenli gorunum). Sadece bu boyla govdeden CIKILAMIYORSA
        # gereken kadar uzatilir -- gorunmeyen isaretci, uzun ok'tan daha kotudur. Uzatma
        # OLCULUR (exit_length), tahmin edilmez; parcalarin %90'inda devreye hic girmez.
        L = L_ARROW
        if is_inside(body, s + od * L) or len(ray_hits(body, s + od * L, od, _reach(body, s))) > 0:
            L = max(L_ARROW, float(exit_length(body, s, od, marg, L_ARROW, 0.45 * diag)))

        # YON GUVENILIR MI: uzattiktan sonra bile uc govdede kaliyor ya da hemen onunde malzeme
        # varsa, bu CP'nin ekseni belirlenememis demektir. O zaman OK CIZILMEZ -- yalniz kure.
        # Guvenmedigimiz bir yonu robota gostermektense, "konumu biliyoruz, yonu bilmiyoruz"
        # demek dogrudur. Denetci bunu makbuzdaki 'arrow' alanindan bilir ve o isaretciye
        # M3/M5 uygulamaz; sessiz gecis degil, ACIK beyandir.
        tip_ = s + od * L
        ahead = ray_hits(body, tip_, od, _reach(body, tip_))
        ok_dir = (not is_inside(body, tip_)) and (not len(ahead) or float(ahead[0]) >= L)
        rs0 = R_SHAFT * shaft_scale
        if not ok_dir:
            parts_m.append(ball(s, rs0 * 2.4, RGBA[role]))
            markers.append({"role": role, "base": [float(x) for x in s],
                            "dir": [float(x) for x in od], "len": 0.0,
                            "tip": [float(x) for x in s], "index": index, "arrow": False,
                            "cp": [float(x) for x in np.asarray(start, float)]})
            return
        head = min(L_HEAD, 0.4 * L)
        head_base = s + od * (L - head)
        tip = s + od * L
        rs = R_SHAFT * shaft_scale
        # KURE = CP'nin kendisi (robotun gidecegi nokta), okun TABANINDA.
        # Ok yonu gosterir, kure YERI gosterir; ikisi ayri bilgidir ve ikisi de gerekli.
        parts_m.append(ball(s, rs * 2.1, RGBA[role]))
        parts_m.append(pin(s, od, L - head, rs, RGBA[role]))
        parts_m.append(cone(head_base, od, head, rs * 2.6, RGBA[role]))
        tips.append(tip)
        markers.append({"role": role, "base": [float(x) for x in s], "dir": [float(x) for x in od],
                        "len": float(L), "tip": [float(x) for x in tip], "index": index,
                        "arrow": True,
                        "cp": [float(x) for x in np.asarray(start, float)]})

    for b_, g in enumerate(Gmouth):
        add_arrow(g, Gout[b_], "gt_hit" if b_ in matched_g else "gt_miss", b_)
    for a_, p_ in enumerate(P):
        # robot_only: TEK renk (robotun CP'si). compare: dogru/fazladan ayrimi.
        role = ROBOT_ROLE if not draw_gt else ("pred_ok" if a_ in matched_p else "pred_fp")
        add_arrow(p_, Pout[a_], role, a_, shaft_scale=0.85)
    parts += parts_m

    n_out = sum(0 if is_inside(body, t) else 1 for t in tips)
    print(f"   visibility: {n_out}/{len(tips)} needle tips outside body"
          + ("" if n_out == len(tips) else "   <-- UYARI: icerde kalan var"))

    for a, b, _, _ in (pairs if draw_gt else []):                         # mavi eslesme cizgisi
        v = Gm[b] - P[a]; n = float(np.linalg.norm(v))
        if n > 1e-6:
            c = trimesh.creation.cylinder(radius=rad * 0.45, height=n, sections=8)
            c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v / n))
            c.apply_translation((P[a] + Gm[b]) / 2)
            c.visual.vertex_colors = np.tile(RGBA["link"], (len(c.vertices), 1))
            parts.append(c)
    os.makedirs(OUT_DIR, exist_ok=True)
    # AYNI PARCANIN ESKI CIZIMLERINI SIL. Dosya adi CP sayisini/F1'i tasidigi icin sayi degisince
    # yeni dosya YENI ada yazilir ve eskisi klasorde KALIR -- 2026-07-29'da tam bu oldu: ok yonu
    # duzeltilince CP sayisi degisti ve eski, bozuk okli GLB'ler results icinde durdu.
    import glob as _glob
    for _old in _glob.glob(os.path.join(OUT_DIR, f"{mfg}_{pid}_*.glb")):
        try:
            os.remove(_old)
            _r = receipt_path(_old)
            if os.path.exists(_r):
                os.remove(_r)
        except OSError:
            pass
    # DOSYA ADI moda gore: robot_only'de F1 YOKTUR (gorulmemis parcada uretici referansi de yok),
    # yerine robotun koydugu CP sayisi yazilir. compare modunda F1 kalir (-Worst/-Best onu kullanir).
    out = os.path.join(OUT_DIR, f"{mfg}_{pid}_F1_{f1:.2f}.glb" if draw_gt
                       else f"{mfg}_{pid}_CP{len(P)}.glb")
    trimesh.util.concatenate(parts).export(out)
    _WRITTEN.append(out)

    # --- MAKBUZ: sozlesmenin denetlenebilir kaydi (bkz CIZIM_SOZLESMESI.md) ---
    # CIZILEN sayilar yazilir: robot_only'de uretici isaretcisi ve cizgi YOKTUR -> 0.
    receipt = {"version": RECEIPT_VERSION, "mode": mode, "part": pid, "mfg": mfg,
               "n_gt": int(len(Gm)) if draw_gt else 0, "n_pred": int(len(P)),
               "tp": int(tp) if draw_gt else 0, "fp": int(fp) if draw_gt else 0,
               "fn": int(fn) if draw_gt else 0,
               "f1": round(float(f1), 6) if draw_gt else None,
               "ref_n_gt": int(len(Gm)), "ref_f1": (round(float(f1), 6) if len(Gm) else None),
               "n_links": int(len(pairs)) if draw_gt else 0, "markers": markers,
               "body_verts": int(len(V)), "body_faces": int(len(F)),
               "mesh_sha": __import__("hashlib").sha256(
                   np.ascontiguousarray(V, np.float64).round(6).tobytes()).hexdigest()[:16],
               "glb": os.path.basename(out)}
    with open(receipt_path(out), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1)
    print(f"   -> {out}")
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    mode = "compare" if "--compare" in argv else "robot_only"
    ids = [a for a in argv if not a.startswith("--")]
    if not ids:
        print(__doc__); sys.exit()
    import torch
    from infer_step_cp import load_any
    cfg = json.load(open("cp_config.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cfg["current_product"]["checkpoints"]]
    n_fail = 0
    for pid in ids:
        try:
            if visualize(pid, models, dev, mode=mode) is None:
                n_fail += 1
        except Exception as e:
            # Hata YUTULMAZ: parca atlanir ama sayilir ve cikis kodu bunu yansitir. Sessiz basarisizlik
            # (2026-07-28 dersi) bir daha "her sey yolunda" gorunemez.
            n_fail += 1
            print(f"{pid}: ERROR {type(e).__name__} {str(e)[:80]}")
    if mode == "robot_only":
        print("\nCOLOR LEGEND (robot-only view -- what the robot would do on an unseen part):")
        print("  RED needle = a connection point the robot placed")
        print("  (no manufacturer markers, no match links -- run with --compare for diagnostics)")
    else:
        print("\nCOLOR LEGEND (compare view -- diagnostics only):")
        print("  GREEN  needle = manufacturer CP, robot FOUND it")
        print("  YELLOW needle = manufacturer CP, robot MISSED it (FN)")
        print("  RED    needle = robot prediction, CORRECT (TP)")
        print("  PINK   needle = robot prediction, SPURIOUS (FP)")
        print("  BLUE   line   = match link (shows mouth-vs-seat offset)")
    # ZORUNLU DENETIM -- CAGRI YOLUNDAN BAGIMSIZ.
    # Denetim eskiden YALNIZCA viz.ps1 sarmalayicisinda kosuyordu; robot_viz.py dogrudan
    # cagrildiginda (internetten inen bir STEP'i denemenin en dogal yolu bu) hicbir GLB
    # dogrulanmadan uretiliyordu. Cizen kod kendini dogrulayamaz -- denetci burada,
    # cizimden sonra, dosyalari DISKTEN geri okuyarak kosar.
    written = [w for w in _WRITTEN if os.path.exists(w)]
    if written:
        print("", flush=True)
        print("Auditing (CIZIM_SOZLESMESI, 9 rules)...", flush=True)
        import subprocess
        rc = subprocess.call([sys.executable, "glb_audit.py", *written,
                              "--json", "results/_audit_last.json"])
        if rc != 0:
            print("AUDIT FAILED -- bkz results/_audit_last.json", flush=True)
            n_fail += 1
    elif not n_fail:
        print("Hic GLB uretilmedi -- denetlenecek sey yok (bu bir BASARISIZLIK).")
        n_fail += 1

    if n_fail:
        print(f"\n{n_fail}/{len(ids)} parts FAILED to render")
        sys.exit(1)
