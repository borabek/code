# -*- coding: utf-8 -*-
"""ROBOT CP EXTRACTION (FAZ B1): a WSCAD STEP file in -> a clean machine-readable CP list out.

This is the product's actual interface, and it implements the thesis goal directly: Scheffler
Masterarbeit Abbildung 1 = "automatic identification of Position, Groesse (size) and Normalvektoren of
the assembly-relevant features"; the derivation follows thesis section 5.3.6 / Abbildung 44 (centroid +
normal + recess depth of a DiffusionNet-segmented Kabeleinfuehrung). Motivation (thesis section 1):
switch-cabinet WIRING is ~49% of assembly time and still almost fully manual -> these CPs feed that
automation. Given any STEP file, the product (4-model UNION vote>=1 + wire-gate 0.35; post-proc
min_v 30 / vertex_conf 0.5 / cluster_mm 5; all from cp_config.json) returns, per connection point:
    point      : [x, y, z]  opening mouth (thesis v_o / Clusterschwerpunkt), in the STEP frame
    direction  : [dx, dy, dz] unit insertion axis = thesis Normalvektor (the way the wire goes in)
    size_mm    : MEASURED narrow width of the mouth -> which wire gauge fits. Rays are fired from
                 the mouth perpendicular to the channel axis and the free span is measured
                 (cp_geometry.mouth_width). Validated on known geometry: 0.02mm error over
                 2-20mm, and on a flat slot it returns the NARROW edge, which is the dimension a
                 wire has to pass. Until 2026-07-30 this was estimated from the segmented region's
                 area instead, which reported ~25mm openings on 4mm terminals because the Contact
                 class covers the whole contact patch. `size_source` says which of the two produced
                 the number; `mouth_wide_mm` carries the wide dimension of a slot.
    depth_mm   : insertion depth (thesis Flaechenschwerpunkt/"Tiefe der Aussparung") -> how far to insert
    confidence : 0..1  segmentation confidence for that opening
    kind       : "CableEntry" | "Contact"
    tier       : "auto"   (confidence >= --conf-auto -> robot acts autonomously)
                 "review" (below -> flag for a human)   <-- the two-tier robot policy

WHY TIERS: leakage-free OOF precision is ~0.72 (the old "~0.98" was INFLATED -- it counted tool/actuator
openings as correct; corrected by the wire-gate). A robot that PHYSICALLY inserts wire should act only on
high-agreement, high-confidence CPs (AUTO) and surface the rest for a human (REVIEW).

Output: JSON (default) or CSV. Coordinates are in the ORIGINAL STEP coordinate frame, so a robot cell
that already works in STEP coordinates can consume them directly (no remesh/align on the robot side).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe robot_cp.py part.stp [more.stp ...] \
         [--csv] [--conf-auto from cp_config] [--out result.json]
"""
import os, sys, glob, json, csv, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, connector3d, cp_openings, thesis_remesh
from infer_step_cp import step_to_mesh, load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OP = "results/step_infer/ops"
KIND = {CE: "CableEntry", CT: "Contact"}


_CFG_ONBELLEK = {}
# Gate hatasi olan parcalar. Bos DEGILSE kosu sifir olmayan cikis kodu doner (fail-safe).
_GATE_HATA = []


def _load_cfg():
    """cp_config'i BIR KEZ oku ve onbellege al.

    Iki sorunu birden cozer:
      1) `json.load(open(...))` dosya tanitici SIZDIRIR. Ayni desen brep_axes'te ADAY BASINA
         cagriliyordu ve 200 parcalik bir olcumu SESSIZCE oldurdu (2026-07-31, t8 kosusu;
         log binlerce ResourceWarning ile doluydu, traceback yoktu).
      2) Bu fonksiyon PARCA BASINA cagriliyor -- ayni kucuk dosya yuzlerce kez diskten okunuyordu.
    """
    if "v" not in _CFG_ONBELLEK:
        yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cp_config.json")
        try:
            with open(yol, "r", encoding="utf-8") as f:
                _CFG_ONBELLEK["v"] = json.load(f)
        except Exception:
            _CFG_ONBELLEK["v"] = {}
    return _CFG_ONBELLEK["v"]


def _vote2(cp_lists, cluster_mm=5.0, min_votes=1):
    """CP-level vote-pool across models: keep an opening >=min_votes models independently found, and
    tag each kept CP with `_votes` (how many models agreed). The PRODUCT calls this with min_votes=1
    (UNION) -- the union's higher recall survives because the downstream wire-gate restores precision
    (raw union precision alone collapses to ~0.40). NOT softmax-averaging (measured dead: cuts recall).
    Highest-confidence member of each agreeing group is the representative. Function default is 1 to
    match the product; callers pass cp_config.robot_min_votes."""
    # MODEL KIMLIGI: her aday hangi uyeden geldigini tasir. Olmadan `_votes` benzersiz MODEL
    # saymiyordu -- ayni modelin yakin iki adayi IKI oy yaziyordu ve 4 checkpoint varken
    # ciktida votes=5 gorulebiliyordu (2026-07-31 denetimi). Bu ozellikle agir, cunku durust
    # (geometri) bolmede gate'in GENELLESEN TEK ozelligi votes (AUC dususu 0.018).
    allc = [dict(c, _mid=i) for i, lst in enumerate(cp_lists) for c in lst]
    allc.sort(key=lambda c: -float(c.get("confidence", 0.0)))
    # EKSEN-FARKINDALIKLI HAVUZLAMA (cp_config.robot_eksen_havuz): iki aday ayni
    # aciklktan sayilir -> DIK mesafe <= yanal_mm VE eksenler hizali (<=20 derece);
    # DERINLIK farki serbest. Gerekce: tez CP'yi AGIZDA (v_o), uretici KONTAKTA
    # tanimlar; uyeler ayni kanali farkli derinlikte isaretleyebilir. Puanlayici
    # (big_arbiter.greedy) zaten boyle calisiyor -- bu, urunun kendi icindeki
    # tutarsizligi kapatir. Olculdu: birlesme %67.7 -> %70.7; ayrica yanal 3mm
    # siniri KOMSU KUTUPLARI (adim 3.5-6mm) duz 5mm kureden daha iyi korur.
    try:
        _c2 = _load_cfg()
    except Exception:
        _c2 = {}
    _eks = bool(_c2.get("robot_eksen_havuz", False))
    _yan = float(_c2.get("robot_eksen_havuz_yanal_mm", 3.0))
    kept = []
    for c in allc:
        p = np.asarray(c["point"], float)
        if _eks:
            d = np.asarray(c["direction"], float)
            d = d / (np.linalg.norm(d) + 1e-9)
            hit = None
            for k in kept:
                e = np.asarray(k["direction"], float)
                e = e / (np.linalg.norm(e) + 1e-9)
                if abs(float(d @ e)) < 0.9397:      # cos(20 derece)
                    continue
                v = p - np.asarray(k["point"], float)
                if float(np.linalg.norm(v - (v @ e) * e)) <= _yan:
                    hit = k; break
        else:
            hit = next((k for k in kept if np.linalg.norm(p - np.asarray(k["point"], float)) <= cluster_mm), None)
        if hit is None:
            c = dict(c); c["_mids"] = {c["_mid"]}
            c["_pts"] = [p]; c["_ws"] = [float(c.get("confidence", 1.0))]
            kept.append(c)
        else:
            if c["_mid"] in hit["_mids"]:
                continue          # AYNI model ikinci kez oy VERMEZ (yakin ikinci adayi atilir)
            hit["_mids"].add(c["_mid"])
            hit["_pts"].append(p); hit["_ws"].append(float(c.get("confidence", 1.0)))
    for c in kept:
        c["_votes"] = len(c["_mids"])          # BENZERSIZ model sayisi
    out = [c for c in kept if c["_votes"] >= min_votes]
    # KONUM = anlasan uyelerin GUVEN-AGIRLIKLI ORTALAMASI (2026-07-31, olculdu).
    # Eskiden yalnizca en guvenli uyenin noktasi tutuluyor, digerlerininki ATILIYORDU --
    # toplulugun en klasik faydasi (bagimsiz hatalarin ortalamada sonmesi) kullanilmiyordu.
    # Geometri bolmesinde, ayni 100 parca, tek degisken:
    #   tespit 0.6511 -> 0.6612 | yanal<=2mm 0.4980 -> 0.5220 | robot-hazir 0.4384 -> 0.4558
    # Duz ortalama da kazaniyor (0.6593/0.5159/0.4499) ama guven-agirlikli daha iyi.
    for c in out:
        c.pop("_mids", None); c.pop("_mid", None)
        P = np.asarray(c.pop("_pts"), float); w = np.asarray(c.pop("_ws"), float)
        if len(P) > 1:
            w = w / max(w.sum(), 1e-9)
            c["point"] = (P * w[:, None]).sum(0)
    return out



def adaylari_uret(V, F, pbs, step_path, cfg=None):
    """URUNUN ADAY URETIMI -- TEK KAYNAK. Runtime ve GATE EGITIMI bunu cagirir.

    NEDEN VAR (2026-08-01 denetimi): gate egitim verisi (`gate_regrow.py`) adaylari AYRI bir
    kod yolundan uretiyordu ve iki yerde SAPIYORDU:
      * `step_path` GECILMIYORDU -> B-rep analitik ekseni ve fiziksel ozellikler egitimde YOK,
        uründe VAR. Gate, gercekte gordugu adaylardan FARKLI adaylarla egitiliyordu.
      * `f1_sweep.union_all` kullaniliyordu, urun ise `_vote2` (BENZERSIZ model sayimi).
        Sonuc: egitim verisinde `votes` 12'ye kadar cikiyordu, uründe tavan MODEL SAYISI (4).
        Ve `votes`, durust bolmede gate'in GENELLESEN TEK ozelligi.

    pbs: model basina olasilik dizisi.
    Doner: **DORT** deger -- (cps, ortalama_olasilik, cok_CP_mu, per).
    (`per` = model basina aday listesi; oy havuzu icin gerekli.)

    NOT 2026-08-07: bu satir "uc deger doner" diyordu ve bir olcum betiginde
    unpack hatasina yol acti; betikteki `except Exception` de hatayi yutup
    ekrana SAHTE bir sonuc bastirdi. Imza degisirse BURASI da degisir.
    """
    import numpy as _np
    _cfg = cfg if cfg is not None else _load_cfg()
    _pp = _cfg.get("prediction_postproc", {})
    mv = int(_pp.get("min_vertices", 30)); vc = float(_pp.get("vertex_confidence_mask", 0.5))
    cl = float(_pp.get("cluster_mm", 5.0))
    mkv = int(_cfg.get("robot_min_votes", 1))
    pr = float(_cfg.get("current_product", {}).get("params", {}).get("conn_promote", 0.0))
    P = [_np.asarray(q, float) for q in pbs]

    # BIRLESME YARICAPLARI ARTIK CONFIG'TEN (2026-08-06, P1).
    # Onceden `dedupe_mm` 10.0 SABIT kodluydu ve oy havuzu `_vote2`'nin varsayilanini
    # (5.0) kullaniyordu. P1 taramasi (D6, aday kahini one-to-one Macar) bu ucunun
    # KOMSU GERCEK GIRISLERI tek adaya yuttugunu olctu: 3/10/5 -> 1/2/2 ile kahin
    # 0.8808 -> 0.8966, COK-CP 0.4740 -> 0.5071 (+0.0331), dusuk-CP kayipsiz.
    # Varsayilanlar ESKI degerler: config'i olmayan bir kurulum aynen eskisi gibi calisir.
    dd = float(_pp.get("dedupe_mm", 10.0))
    oy = float(_pp.get("vote_pool_mm", 5.0))
    # SPLIT_RATIO CONFIG'TEN (2026-08-09 / P3-a). `cp_openings._split_elongated`
    # birlesmis IKI komsu agzi ayirir ama urun yolu bu parametreyi HIC gecirmiyordu
    # -> varsayilan 0.0 = KAPALI, yani olu kod. Artik taranabilir.
    sr = float(_pp.get("split_ratio", 0.0))

    def _turet(promote):
        ek = {"conn_promote": promote} if promote else {}
        return [cp_openings.connection_points(
            V, F, q.argmax(-1), min_v=mv, classes=(CE, CT), dedupe_mm=dd,
            probs=q, vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cl,
            split_ratio=sr, step_path=step_path, **ek) for q in P]

    per = _turet(0.0)
    n_union = len(_vote2(per, cluster_mm=oy, min_votes=mkv)) if len(per) > 1 else len(per[0])
    is_hi = _highcp_router(step_path, V, n_union) if pr > 0.0 else False
    if pr > 0.0 and is_hi:
        per = _turet(pr)
    if len(per) == 1:
        cps = per[0]
        for c in cps:
            c["_votes"] = 1
    else:
        cps = _vote2(per, cluster_mm=oy, min_votes=mkv)
    # UYE LISTELERI de donuyor: `uye_yonu_sec` birlestirmede ATILAN uye yonlerini kullanir
    # (R7: o yonlerde robot-haziri 0.5523 -> 0.6059 yapacak bilgi VAR).
    return cps, sum(P) / len(P), bool(is_hi), per


def _highcp_router(step_path, V, n_cand, model_path="results/highcp_router.pkl"):
    """Bu parca COK-CP mi? SADECE geometriden karar verir -- uretici metadata'si YOK.

    conn_promote cok-CP'de +0.072 kazandirir ama dusuk-CP'de -0.018 kaybettirir; kuresel
    uygulanamaz. Robot bilinmeyen bir parcada CP sayisini bilmedigi icin rejimi KENDI anlamali.
    Girdi: STEP bbox (3 kenar + kosegen + kutu alani/hacmi) + promote'suz aday sayisi/yogunlugu.
    Aile-disi AUC 0.9942. Model yoksa False doner (promote kapali = eski davranis).
    """
    import os, pickle
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as fh:
            R = pickle.load(fh)
        ext = np.sort(np.asarray(V).max(0) - np.asarray(V).min(0))[::-1]
        diag = float(np.linalg.norm(ext))
        area = float(2 * (ext[0]*ext[1] + ext[0]*ext[2] + ext[1]*ext[2]))
        vol = float(ext[0]*ext[1]*ext[2])
        x = np.array([[ext[0], ext[1], ext[2], diag, area, vol, n_cand,
                       n_cand / max(area, 1e-6) * 1e3, n_cand / max(diag, 1e-6)]], float)
        return bool(R["model"].predict_proba(x)[0, 1] >= R.get("threshold", 0.30))
    except Exception as e:
        print(f"  [router atlandi: {e}]", file=sys.stderr)
        return False

def extract(models, step_path, dev, conf_auto, min_auto_votes=1, cp_count=None, spatial_rerank=False):
    """Run the product pipeline on one STEP file -> list of CP dicts in the STEP frame.

    `models` = [(model, meta), ...]. With several, CPs are combined by UNION (vote>=1, per
    cp_config.robot_min_votes) then filtered by the RF wire-gate -- the measured product
    (leakage-free OOF F1: ALL 0.750 / WEI 0.690 / PXC 0.703; RF gate, GB-era 0.693 RETIRED). A
    single model still works.

    TIER = AUTO only when confidence >= conf_auto AND votes >= min_auto_votes. The vote count (how
    many of the models independently found the opening) is the ensemble's own reliability signal --
    calibrated on the manufacturer arbiter (calibrate_conf.py) because a threshold tuned for a single
    model is wrong for the vote (its representative confidence = the highest-confidence member).
    """
    if not isinstance(models, (list, tuple)):
        raise TypeError("extract() expects a list of (model, meta) pairs")
    Vr, Fr = step_to_mesh(step_path)                         # original STEP tessellation
    V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    _cfg = _load_cfg()
    _pp = _cfg.get("prediction_postproc", {})
    _mv = int(_pp.get("min_vertices", 30)); _vc = float(_pp.get("vertex_confidence_mask", 0.5))
    _cl = float(_pp.get("cluster_mm", 5.0))
    # ADAY URETIMI TEK KAYNAKTAN: `adaylari_uret`. Gate egitimi de AYNI fonksiyonu cagirir
    # (2026-08-01 denetimi: egitim yolu step_path'siz ve union_all ile ayrisiyordu).
    pbs = []; acc = None
    for model, meta in models:
        # per-k_eig cache dir: karisik k_eig (64/96) tek mesh-anahtarli onbellegi paylasip
        # cakisiyor; k_eig basina ayri dizin bunu kaldirir.
        _opd = f"{OP}_k{int(meta.get('k_eig', 64))}"
        _, probs = D.predict(model, meta, V, F, device=dev, op_cache_dir=_opd, return_probs=True)
        probs = np.asarray(probs, float)
        acc = probs if acc is None else acc + probs
        pbs.append(probs)
    cps, avg_probs, _is_highcp, _uyeler = adaylari_uret(V, F, pbs, step_path, cfg=_cfg)
    per_model = pbs          # asagidaki uzunluk kontrolleri icin
    # WIRE/TOOL GATE: drop tool/actuator openings (Werkzeugeinschub), keep real wire entries.
    # The Contact segmentation class merges contact+tool by thesis design (line 1303), and neither the
    # class nor local geometry separates them (recon AUC 0.61); a learned structural discriminator does
    # (held-out AUC 0.867; end-to-end precision 0.607->0.751 @ thr 0.30, recall 0.684->0.651). Opt-in via
    # cp_config.robot_wire_gate so older behaviour is preserved when the model file is absent.
    if _cfg.get("robot_wire_gate", True):
        try:
            import wire_gate
            # cp_count given (metadata-assisted mode) -> keep top-N by wire_score; else base threshold.
            # --spatial-rerank (opt-in, top-N modu only): spatial-augmented gate ranking (+0.002 ALL / WEI +0.01-0.02).
            if spatial_rerank and cp_count is not None:
                cps = wire_gate.apply_spatial(V, F, avg_probs, cps, CE, CT, top_n=cp_count,
                                              step_path=step_path)
            else:
                # REJIM-KOSULLU GATE ESIGI (2026-07-29, olculdu): cok-CP parcalarinda 0.35
                # recall'u 0.485'e kirpiyor -- oysa adaylar 0.798'de. 10 cok-CP parcasinda
                # esik suzgeci: 0.15->F1 0.608, 0.25->0.634, 0.35->0.587, 0.45->0.343.
                # Dusuk-CP'de yuksek esik iyi (precision), cok-CP'de dusuk esik iyi (recall).
                # Rejimi zaten yonlendirici belirliyor; ayni karari burada da kullan.
                _thr = float(_cfg.get("robot_wire_gate_threshold", 0.30))
                if _is_highcp:
                    _thr = float(_cfg.get("robot_wire_gate_threshold_highcp", _thr))
                cps = wire_gate.apply(V, F, avg_probs, cps, CE, CT,
                                      threshold=_thr, top_n=cp_count,
                                      step_path=step_path)
        except Exception as e:
            # FAIL-SAFE (2026-08-01 denetimi): eskiden burada SESSIZCE devam ediliyordu ve urun
            # HAM BIRLESIMI dondururdu -- ham birlesimin kesinligi ~0.40, yani cikti "normal"
            # gorunurken aslinda urun DEGILDI. Artik parca acikca REVIEW'a dusurulur, kayda
            # gecer ve kosunun cikis kodu sifir OLMAZ. Sessiz bozulma imkansiz.
            _GATE_HATA.append((os.path.basename(step_path or "?"), str(e)))
            print(f"  [!! WIRE-GATE HATASI -- parca REVIEW'a dusuruldu: {e}]", file=sys.stderr)
            for _c in cps:
                _c["_gate_hata"] = str(e)

    # POSE HEAD: gate KARARINDAN SONRA, kabul edilmis CP'lerin YANAL sapmasini duzelt.
    # Tavan olcumu (results/t_tavan.json): kahin gate robot-haziri yalniz +0.059 tasiyor,
    # KONUM +0.325. Yani gate'in otesindeki tek gercek kaldirac buydu.
    # Olculdu (kilitli kume, GRUP bootstrap): robot-hazir 0.4407 -> 0.4835
    # (+0.0428, GA [+0.0220,+0.0680] KANITLI), tespit +0.0001.
    # Gate hatasi olan parcalarda UYGULANMAZ: orada elimizdeki ham birlesimdir.
    if _cfg.get("robot_pose_head", False) and cps and not any(c.get("_gate_hata") for c in cps):
        try:
            import wire_gate as _wg
            _Xp = _wg.feats_for(V, F, avg_probs, cps, CE, CT, step_path=step_path)
            cps = _wg.pose_duzelt(_Xp, cps)
            # SECICI ACI DUZELTMESI: yalniz "yonu yanlis" denen adaylarda. Tespit
            # olcutu aciya bakmadigi icin bu adimin tespite bedeli YAPISAL OLARAK SIFIR.
            if _cfg.get("robot_aci_secici", False):
                cps = _wg.aci_duzelt(_Xp, cps)
            # UYE YON SECIMI: birlestirmede atilan uye yonlerinden en iyisini sec.
            if _cfg.get("robot_uye_secici", False):
                cps = _wg.uye_yonu_sec(_Xp, cps, _uyeler)
            # AYRIK YON SECICI: fiziksel bir yon SOZLUGUNDEN en iyisini SECER.
            # Zincirin SONUNDA cagrilir; sozlugun "mevcut" girisi onceki tum
            # duzeltmelerin ciktisi olmalidir. Olculdu (secici 1301 AYRI parcada
            # egitildi; olcum kumesini ve LOCKED gruplarini HIC gormedi):
            #   robot-hazir 0.5893 -> 0.6213 (+0.0319, GA [+0.0120,+0.0527])
            #   tespit DEGISMEZ -- YAPISAL: tespit olcutu aciya bakmaz.
            if _cfg.get("robot_yon_secici", False):
                cps = _wg.yon_sozluk_sec(_Xp, cps, V, step_path=step_path,
                                         uyeler=_uyeler)
        except Exception as e:
            print(f"  [pose duzeltmesi atlandi: {e}]", file=sys.stderr)
    # OLCULEN agiz genisligi. Eski size_mm segmentlenmis BOLGENIN alanindan geliyordu
    # (2*sqrt(alan/pi)); Contact sinifi tum kontak yuzeyini kapsadigi icin 4mm'lik klemenslerde
    # ~25mm "delik capi" yaziyordu -- robotun kullanamayacagi bir sayi. Burada agizda, kanal
    # eksenine DIK yonlerde isin atilip bosluk OLCULUR (dogrulandi: 2-20mm arasi 0.02mm hata,
    # yassi yuvada dar kenari verir). Olculemezse (0.0) eski tahmin korunur -- uydurma yok.
    try:
        from cp_geometry import mouth_width
        for c in cps:
            ic, ort = mouth_width((V, F), c["point"], c["direction"])
            if ic > 0:
                c["_mouth_min_mm"], c["_mouth_mean_mm"] = ic, ort
    except Exception as e:
        print(f"  [agiz olcumu atlandi: {e}]", file=sys.stderr)
    # the remesh keeps the STEP coordinate frame (thesis_remesh does not recentre), so points are
    # already in STEP coordinates -- the robot cell can use them as-is.
    return _format_cps(cps, conf_auto, min_auto_votes)


def tier_ata(c, conf_auto, min_auto_votes, auto_thr=None):
    """TEK CP icin tier: "auto" | "review".

    ORTAK YERE TASINDI (2026-08-12). Bu kural `_format_cps` govdesine gomuluydu
    ve yalnizca `extract` yolundan gecen ciktilar tier alani aliyordu. Oysa
    olculen zincir `kanonik_zincir.urun_cikti`; GLB ihracatcisini oraya
    baglamak icin tier'in AYRI cagrilabilmesi gerekiyor (bkz. rapor bolum 8:
    "olculen zincir GLB'ye girmiyor", 2. madde TIER TUZAGI).

    DAVRANIS DEGISMEDI -- govde birebir tasindi:
      * gate hatasi varsa       -> review (ham birlesimin kesinligi ~0.40)
      * gate skoru varsa        -> wire_score >= auto_thr ise auto
      * gate skoru YOKSA (eski) -> confidence >= conf_auto VE votes >= min_votes

    OLCULEN GERCEK (D7, gorulmemis marka): dagitilan esikte (0.6) isaretlerin
    %100'u AUTO cikiyor ve kesinlik 0.3471 -- REVIEW katmani BOS. Skor tabani
    0.6006, yani esik dagilimin ALTINDA kaliyor ve hicbir seyi elemiyor.
    Bu fonksiyon o kusuru DUZELTMEZ, yalnizca tek yere toplar.
    """
    # GUVENLI ANAHTAR: `cp_config.robot_auto_kapali = true` ise HICBIR isaret
    # otonom isaretlenmez, hepsi REVIEW olur.
    #
    # NEDEN VAR (olculdu, D7 gorulmemis marka, makbuz tier_cokusu_d7.json):
    # dagitilan esikte (0.6) isaretlerin %100'u AUTO ve kesinlik 0.3471 --
    # REVIEW katmani BOS. Skor tabani 0.6006, yani esik dagilimin ALTINDA ve
    # hicbir seyi elemiyor. Robot ucte ikisi yanlis isarete kendi basina
    # guveniyor. Esigi yukseltmek KURTARMIYOR (0.95'te bile kesinlik 0.4652).
    #
    # VARSAYILAN FALSE: urunun bugunku davranisi DEGISMEZ. Anahtar, kalibre
    # bir skor cikana kadar sahada guvenli tarafa gecmek isteyen icin.
    try:
        if bool(_load_cfg().get("robot_auto_kapali", False)):
            return "review"
    except Exception:
        pass
    if auto_thr is not None and "wire_score" in c:
        if c.get("_gate_hata"):
            return "review"
        return "auto" if float(c.get("wire_score", 1.0)) >= auto_thr else "review"
    conf = float(c.get("confidence", 0.0))
    votes = int(c.get("_votes", 1))
    return "auto" if (conf >= conf_auto and votes >= min_auto_votes) else "review"


def _format_cps(cps, conf_auto, min_auto_votes):
    """CP dict list -> machine-readable output records (STEP frame), sorted by (votes, confidence).

    TIER KURALI -- 2026-07-28 kullanici karari, 2026-07-29'da KODA BAGLANDI:
      AUTO = wire_score (gate skoru) >= cp_config.robot_auto_gate_threshold (0.66).

    ONCEKI KUSUR (bu satir 2026-07-29'a kadar yanlisti): tier SEGMENTASYON guvenine bakiyordu
    (confidence >= 0.5). Sonucu kilitli holdout'ta olculdu: REVIEW BOS cikti, her sey AUTO
    isaretlendi ve precision 0.7735'ti -- yani robot 4 yerlestirmede 1 hatayi insana HIC
    sormadan yapiyordu. Karar verilmis, config'e yazilmis, ama koda hic baglanmamisti
    (`robot_auto_gate_threshold` hicbir yerde okunmuyordu -- K6 denetimi buldu).

    Olculen dogru davranis: esik 0.66 -> AUTO precision 0.9508, CP'lerin %52'si otonom.
    Gate kapaliysa (wire_score yok) eski kurala duser, cunku o zaman gate skoru anlamsizdir.
    """
    _auto_thr = None
    try:
        _auto_thr = float(_load_cfg().get("robot_auto_gate_threshold"))
    except Exception:
        _auto_thr = None
    out = []
    for c in cps:
        conf = float(c.get("confidence", 0.0))
        votes = int(c.get("_votes", 1))
        d = np.asarray(c["direction"], float); d = d / (np.linalg.norm(d) + 1e-9)
        area = float(c.get("area", 0.0))
        # OLCULEN agiz genisligi varsa o kullanilir; yoksa segment alanindan tahmin (eski yol).
        # ONCELIK SIRASI: B-rep yaricapi (analitik, kesin) > isinla olcum > alan tahmini.
        # Isinla olcum, CP kanal merkezinde degilse EN KUCUK mesafeyi alir ve fiziksel olmayan
        # degerler uretebilir (1070018: 13 CP'nin 4'u 0.5mm altinda, en dusugu 0.05mm).
        # B-rep eslesmisse silindirin kendi yaricapi zaten tam dogru olcudur.
        _br = c.get("brep_radius_mm")
        if _br:
            size_mm = round(2.0 * float(_br), 2)
            size_src = "brep"
        else:
            size_mm = float(c.get("_mouth_min_mm", 0.0))
            size_src = "measured"
        if size_mm <= 0:
            size_mm = round(2.0 * (area / np.pi) ** 0.5, 2) if area > 0 else 0.0
            size_src = "area_estimate"
        out.append({
            "point": [round(float(x), 4) for x in c["point"]],
            "direction": [round(float(x), 4) for x in d],
            "size_mm": size_mm,                    # thesis Groesse (Abb.1): telin sigmasi gereken DAR olcu
            "size_source": size_src,               # "measured" = isinla olculdu, "area_estimate" = eski tahmin
            "mouth_wide_mm": round(float(c.get("_mouth_mean_mm", 0.0)), 2),  # yuvanin genis olcusu
            "depth_mm": round(float(c.get("insertion_depth_mm", 0.0)), 2),  # thesis Flaechenschwerpunkt depth
            "confidence": round(conf, 3),
            "votes": votes,                                        # how many models agreed (ensemble reliability)
            "wire_score": round(float(c.get("wire_score", 1.0)), 3),  # wire-gate: 1=wire entry, 0=tool/actuator opening
            "kind": KIND.get(int(c.get("source_label", CE)), "CableEntry"),
            "n_verts": int(c.get("n_verts", 0)),
            # GATE HATASI -> ASLA auto. Gate calismadiysa elimizdeki HAM BIRLESIMDIR ve
            # onun kesinligi ~0.40; robot buna dayanarak tel takmamali.
            "tier": tier_ata(c, conf_auto, min_auto_votes, _auto_thr),
        })
    out.sort(key=lambda r: (-r["votes"], -r["confidence"]))
    return out


def extract_highcp(models7, step_path, dev, conf_auto, min_auto_votes, cp_count):
    """HIGH-CP metadata-assisted mode (SEPARATE from base product): high-CP(>=11) terminals only.
    7 models (4 product + 3 keig128) x 2 resolutions (6k+9k) -> UNION -> lattice-augmented selector
    (highcp_selector.pkl) -> top-N by cp_count. OOF F1 0.807 part-out / 0.799 family-out (receipt:
    results/product_f1_receipt.json). cp_count REQUIRED. ~14 forward passes/part -- heavy, opt-in only."""
    import highcp_selector
    if cp_count is None or cp_count < 1:
        raise ValueError("extract_highcp needs cp_count (manufacturer CP count)")
    _cfg = _load_cfg(); hc = _cfg.get("robot_highcp", {})
    d6 = hc.get("derive_6k", {"min_v": 30, "vertex_conf": 0.5, "cluster_mm": 5.0})
    d9 = hc.get("derive_9k", {"min_v": 12, "vertex_conf": 0.20, "cluster_mm": 0.0})

    def derive(V, F, pb, dd):
        return cp_openings.connection_points(V, F, pb.argmax(-1), min_v=int(dd["min_v"]), classes=(CE, CT),
                                             dedupe_mm=10.0, probs=pb, vertex_conf=float(dd["vertex_conf"]),
                                             ct_depth_min_mm=1.0, cluster_mm=float(dd["cluster_mm"]),
                                             step_path=step_path)   # PARITE (bkz yukari)
    Vr, Fr = step_to_mesh(step_path)
    V6, F6 = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V6 = np.ascontiguousarray(V6, np.float64); F6 = np.ascontiguousarray(F6, np.int64)
    per = []; acc = None
    for model, meta in models7:
        _, pb = D.predict(model, meta, V6, F6, device=dev,
                          op_cache_dir=f"{OP}_k{int(meta.get('k_eig', 64))}", return_probs=True)
        pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb; per.append(derive(V6, F6, pb, d6))
    avg = acc / len(models7)
    V9, F9 = thesis_remesh.remesh_uniform(Vr, Fr, target=9000)
    V9 = np.ascontiguousarray(V9, np.float64); F9 = np.ascontiguousarray(F9, np.int64)
    for model, meta in models7:
        _, pb = D.predict(model, meta, V9, F9, device=dev,
                          op_cache_dir=f"{OP}_k{int(meta.get('k_eig', 64))}_9k", return_probs=True)
        per.append(derive(V9, F9, np.asarray(pb, float), d9))
    cps = _vote2(per, min_votes=1)                          # union of all 6k+9k candidates
    if not cps:
        return []
    cps = highcp_selector.apply(cps, V6, avg, CE, CT, int(cp_count),
                                model_path=hc.get("selector", "results/highcp_selector.pkl"))
    return _format_cps(cps, conf_auto, min_auto_votes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="STEP file(s), or bare part ids found in all_wscad_stp")
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="one or more checkpoints; several -> UNION (vote>=1) + wire-gate (the product). "
                         "Default: cp_config.robot_vote2_checkpoints.")
    ap.add_argument("--conf-auto", type=float, default=None,
                    help="AUTO tier: confidence >= this. Default from cp_config.robot_conf_auto.")
    ap.add_argument("--min-auto-votes", type=int, default=None,
                    help="AUTO tier: votes >= this. Default from cp_config.robot_min_auto_votes.")
    ap.add_argument("--cp-count", type=int, default=None,
                    help="METADATA-ASSISTED mode: manufacturer CP count. If given, keep the top-N CPs by "
                         "wire_score instead of the gate threshold (measured OOF F1 0.775 vs base 0.750; "
                         "high-CP full-stack 0.807 part-out / 0.799 family-out). "
                         "Report as a SEPARATE mode, not the base product F1.")
    ap.add_argument("--highcp", action="store_true",
                    help="Force HIGH-CP mode (7-model 6k+9k multires + lattice selector). Needs --cp-count. "
                         "Auto-on when --cp-count >= robot_highcp.hicp_threshold. SEPARATE mode, not base F1.")
    ap.add_argument("--no-highcp", action="store_true", help="Disable auto high-CP mode even at cp_count>=11.")
    ap.add_argument("--spatial-rerank", action="store_true",
                    help="OPT-IN (top-N/--cp-count mode only): spatial-augmented gate ranking (5/5 seed-robust "
                         "+0.002 ALL, WEI +0.01-0.02). Needs results/wire_gate_spatial.pkl. Not for base threshold.")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = _load_cfg()
    cks = a.ckpt or cfg["robot_vote2_checkpoints"]
    # ALL product params flow from cp_config (CLI only overrides) so script == deployed product
    conf_auto = a.conf_auto if a.conf_auto is not None else float(cfg.get("robot_conf_auto", 0.5))
    min_auto_votes = a.min_auto_votes if a.min_auto_votes is not None else int(cfg.get("robot_min_auto_votes", 3))
    models = [load_any(c, dev=a.device)[:2] for c in cks]
    # Banner must state the rule that ACTUALLY runs. _format_cps prefers the gate score whenever a
    # wire_score is present and robot_auto_gate_threshold is set, and only falls back to
    # conf/votes when the gate is off -- printing the fallback unconditionally misreported the
    # deployed tier rule.
    _gate_on = bool(cfg.get("robot_wire_gate", True)) and cfg.get("robot_auto_gate_threshold") is not None
    _tier = (f"wire_score>={cfg['robot_auto_gate_threshold']}" if _gate_on
             else f"conf>={conf_auto} & votes>={min_auto_votes}")
    # BANNER GERCEGI SOYLEMELI: goreli esik acikken sabit esik sayilari KULLANILMIYOR ama banner
    # onlari yaziyordu (2026-08-01). Yaniltici, cunku ikisi cok farkli davraniyor -- ayni iki
    # parcada sabit esik 0 CP, goreli esik 3 ve 2 CP donduruyor.
    if cfg.get("gate_goreli_esik"):
        _kural = (f"goreli esik (parca-ici {cfg.get('gate_goreli_oran', 0.5)}x en yuksek, "
                  f"taban {cfg.get('gate_goreli_taban', 0.25)})")
    else:
        _kural = (f"sabit esik {cfg.get('robot_wire_gate_threshold', 0.35)}"
                  f"/{cfg.get('robot_wire_gate_threshold_highcp')} (dusuk/cok-CP)")
    # Sutun sayisi URETIMDEN degil DAGITILAN MODELDEN okunmali: parca-ici donusum (2026-08-01)
    # genisligi ikiye katliyor ve banner ham sayiyi yazarsa yine yaniltir -- bu bloga tam olarak
    # bunun icin dokunuldu, ayni hatayi bir sonraki katmanda tekrarlamayalim.
    try:
        import wire_gate as _wg
        _m = _wg._load(_wg.MODEL_PATH)
        _ham = len(_wg.FEAT_NAMES)
        _nf = (_m or {}).get("n_feat") or _ham
        _don = (_m or {}).get("donusum")
        _nf = f"{_nf} sutun" + (f" ({_ham} ham + parca-ici {_don})" if _don else "")
        if (_m or {}).get("clf_z") is not None:
            _nf += (f" + COKUS YONLENDIRME (parca maks skoru < {float(_m['esik_cokus']):.3f} "
                    f"ise {2 * _ham} sutunlu parca-ici {_m.get('donusum_z')} modeli)")
    except Exception:
        _nf = "? sutun"
    print(f"URUN: {len(models)} model UNION(vote>=1) + wire-gate {_nf} | {_kural}"
          f" | AUTO tier: {_tier}" if len(models) > 1
          else f"URUN: {os.path.basename(cks[0])} (tek model)", flush=True)

    # HIGH-CP mode: opt-in (--highcp) or auto when cp_count >= threshold. Loads 3 extra keig128 models.
    hc = cfg.get("robot_highcp", {})
    hicp_thr = int(hc.get("hicp_threshold", 11))
    hc_ready = (os.path.exists(hc.get("selector", "")) and
                all(os.path.exists(c) for c in hc.get("extra_checkpoints", [])))
    use_highcp = (not a.no_highcp) and (a.highcp or (a.cp_count is not None and a.cp_count >= hicp_thr))
    models7 = None
    if use_highcp:
        if a.cp_count is None:
            print("  [high-CP mode --cp-count gerektirir; base moda dusuldu]", file=sys.stderr); use_highcp = False
        elif not hc_ready:
            print("  [high-CP artefact/checkpoint eksik; base moda dusuldu]", file=sys.stderr); use_highcp = False
        else:
            extra = [load_any(c, dev=a.device)[:2] for c in hc["extra_checkpoints"]]
            models7 = models + extra
            print(f"HIGH-CP mode: {len(models7)} model x 6k+9k -> lattice selector top-{a.cp_count} "
                  f"(SEPARATE mode, OOF 0.807 part-out / 0.799 family-out)", flush=True)

    # PARCA KIMLIGI TEK KAYNAKTAN (2026-08-07). Onceden `basename.split("_")[1]`
    # kullaniliyordu; bu YALNIZCA `wscaduniverse_<pid>_<ts>.stp` sozlesmesinde dogru.
    # JSON sozlesmeli dosyalarda (`UPUN.016029_ElectricalTerminal_...stp`) parca adi
    # **"ElectricalTerminal"** cikiyordu -- yani TESLIM EDILEN JSON'da parca adi YANLIS.
    # Duman testinde goruldu. `korpus_kimlik.step_kimlik` iki sozlesmeyi de bilir.
    from korpus_kimlik import step_kimlik as _sk
    pool = {_sk(s): s for s in glob.glob("all_wscad_stp/*.stp")}

    results = {}
    for p in a.parts:
        path = p if os.path.exists(p) else pool.get(p)
        if not path:
            print(f"  {p}: STEP bulunamadi", file=sys.stderr); continue
        pid = _sk(path)
        if use_highcp:
            cps = extract_highcp(models7, path, a.device, conf_auto, min_auto_votes, a.cp_count)
        else:
            cps = extract(models, path, a.device, conf_auto, min_auto_votes, cp_count=a.cp_count, spatial_rerank=a.spatial_rerank)
        results[pid] = cps
        na = sum(1 for c in cps if c["tier"] == "auto")
        print(f"  {pid}: {len(cps)} CP  ({na} auto / {len(cps)-na} review)", flush=True)

    # FAIL-SAFE OZETI: gate hatasi yasandiysa kosu SESSIZ BASARILI donmez.
    if _GATE_HATA:
        print("", file=sys.stderr)
        print(f"!! WIRE-GATE {len(_GATE_HATA)} PARCADA CALISMADI -- o parcalar HAM BIRLESIM "
              f"(kesinlik ~0.40) ve tamami REVIEW isaretli:", file=sys.stderr)
        for _pid, _e in _GATE_HATA[:10]:
            print(f"     {_pid}: {_e}", file=sys.stderr)
        print("   Bu ciktilar ROBOTA VERILMEMELIDIR.", file=sys.stderr)

    if a.csv:
        out = a.out or "robot_cp.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["part_id", "x", "y", "z", "dx", "dy", "dz", "size_mm", "depth_mm", "confidence", "kind", "tier"])
            for pid, cps in results.items():
                for c in cps:
                    w.writerow([pid, *c["point"], *c["direction"], c["size_mm"], c["depth_mm"], c["confidence"], c["kind"], c["tier"]])
        print(f"-> {out}")
    else:
        out = a.out or "robot_cp.json"
        json.dump(results, open(out, "w"), indent=1)
        print(f"-> {out}")
    if _GATE_HATA:
        sys.exit(3)          # sessiz basari YOK: gate hatasi cikis koduna yansir


if __name__ == "__main__":
    main()
