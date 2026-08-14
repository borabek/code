# -*- coding: utf-8 -*-
"""D5-2: YENI KORPUSU TURET + GEOMETRI ANAHTARLARINI AYNI GECISTE URET.

DataSet5 ile uygun korpus 1926 -> 4720 parca / 23 uretici oldu. Gate egitimi icin yeni
parcalarin TURETILMESI gerekiyor (ag cikarimi + aday uretimi + X/XR oznitelikleri).

TEK GECIS: turetme zaten her parcanin STEP'ini yukleyip remesh'liyor; geometri anahtari
(D5-1) AYNI ANDA hesaplanir. Ayri gecis ~1.5 saat tasarruf edilir.

SIRALAMA URETICI-DENGELI: her turda her ureticiden bir parca alinir. Boylece YARIDA
KESILSE BILE elde 23 ureticiden DENGELI bir alt kume olur ve ARA OLCUM anlamli olur.
(Alfabetik sirayla yapilsaydi ilk 800 parca yalniz A-B + ABB olurdu.)

DEVAM EDILEBILIR: her 25 parcada diske yazar; yeniden calistirilinca kaldigi yerden devam.

TEZ DEGISMEZ: ayni ag (4 checkpoint), ayni uniform ~6000 remesh, ayni `v_o` aday
ureticisi. Degisen tek sey KORPUS BUYUKLUGU -- bu, "yalniz veri" deneyinin sarti.
"""
import argparse
import collections
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CIKTI = "results/_der_yeni.pkl"      # vardiyada _der_yeni_<k>.pkl olur
GEO = "results/_geo_yeni.json"
ISARET = "results/_d5_su_an.txt"        # su an islenen parca -- takilma teshisi icin
ATLA_DOSYA = "results/_d5_atla.txt"     # kilitlendigi bilinen parcalar
OPS_TMP = "results/_ops_gecici"         # parca-yerel operator onbellegi (her parcada silinir)
DISK_ESIK_GB = 3.0                      # bunun altinda TEMIZ dur (gmsh sessizce surunmeye baslar)


def dengeli_sira(E):
    """Uretici-dengeli sira: her turda her ureticiden bir parca."""
    g = collections.defaultdict(list)
    for t in E:
        g[t[0]].append(t)
    for k in g:
        g[k].sort(key=lambda x: x[1])
    # SONSUZ DONGU HATASI (2026-08-04, kendi hatam): once `while any(g.values())` yaziyordum
    # ama listelerden eleman CIKARMIYORUM, indeksle geziyorum -> listeler hic bosalmiyor,
    # kosul hep True kaliyor ve `i` tum uzunluklari astiktan sonra dongu SONSUZA kadar
    # donuyordu. Turetme parcalara HIC baslamadi, 20 dakika bos dongude CPU yakti; ben de
    # iki kosuyu "parca kilitlendi" sanip oldurdum. Dogrusu EN UZUN listeye kadar donmek.
    enb = max(len(v) for v in g.values()) if g else 0
    out = []
    for i in range(enb):
        for k in sorted(g):
            if i < len(g[k]):
                out.append(g[k][i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinir", type=int, default=0)
    # VARDIYA (2026-08-04): olculdu -- 16 cekirdegin yalnizca %12'si, GPU %24 kullaniliyor.
    # Darbogaz spektral operator kurulumu ve SERI calisiyor. Is N vardiyaya bolununce
    # neredeyse dogrusal hizlaniyor: 13 saat -> ~3.5 saat (4 vardiya, 8 cekirdek, %50 pay durur).
    # CKPT OVERRIDE (G6): yeni seg agiyla YENIDEN turetme icin. Gate'in yeni aday
    # dagilimiyla egitilmesi SART -- eski gate'e yeni adaylari vermek [[gate-refit-minv4]]
    # dersinin ihlali olur (iki gate AYNI dagilimda egitilmeli).
    ap.add_argument("--ckpt", nargs="+", default=[], help="cp_config yerine bu ckpt'leri kullan")
    # G6: TUM korpus YENI agla yeniden turetilir -- eski turetme "zaten var" SAYILMAZ,
    # cunku eski adaylar eski agin ciktisi. Karistirmak gate'i iki farkli dagilimla egitir.
    ap.add_argument("--hepsi", action="store_true", help="eski turetmeyi 'var' sayma")
    # PID FILTRESI (R4a): TOPLULUK olcumu icin TUM korpusu turetmeye gerek yok --
    # yalniz iki OLCUM kumesi (250 sinav + 194) yeter, ~444 parca.
    ap.add_argument("--pids-file", default="", help="yalniz bu dosyadaki pid'leri turet")
    ap.add_argument("--cikti-eki", default="", help="cikti dosya adina ek (G6 icin _g6)")
    ap.add_argument("--vardiya", type=int, default=0)
    ap.add_argument("--toplam", type=int, default=1)
    a = ap.parse_args()

    import protokol
    protokol.tez_dogrula()
    import torch
    import cad_eval
    import diffusionnet as D_
    import robot_cp as RC
    import thesis_remesh
    import wire_gate
    from big_arbiter import eligible
    from build_zengin_parite import _normaller, zengin
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from geometri_anahtar import anahtar
    from infer_step_cp import load_any, step_to_mesh

    os.makedirs(OPS_TMP, exist_ok=True)
    for _e in os.listdir(OPS_TMP):        # onceki kosudan artik kalmasin
        shutil.rmtree(os.path.join(OPS_TMP, _e), ignore_errors=True)
    global CIKTI, GEO, ISARET
    ek = a.cikti_eki
    if a.toplam > 1:
        CIKTI = f"results/_der_yeni{ek}_{a.vardiya}.pkl"
        GEO = f"results/_geo_yeni{ek}_{a.vardiya}.json"
        ISARET = f"results/_d5_su_an{ek}_{a.vardiya}.txt"
    elif ek:
        CIKTI = f"results/_der_yeni{ek}.pkl"; GEO = f"results/_geo_yeni{ek}.json"
    ATLA = set()
    if os.path.exists(ATLA_DOSYA):
        ATLA = {x.strip() for x in io.open(ATLA_DOSYA, encoding="utf-8") if x.strip()}
        print(f"atlanacak (kilitlenen) parca: {len(ATLA)}", flush=True)
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    E = eligible()
    # zaten turetilmis olanlar (gate korpusu + olcum kumesi)
    var = set()
    if not a.hepsi:
        d = np.load("results/zengin_parite_w2.npz", allow_pickle=True)
        var |= {str(x) for x in d["pids"]}
        for y in ("results/_der_tam.pkl", "results/_der_kontrol.pkl"):
            if os.path.exists(y):
                with open(y, "rb") as f:
                    var |= {r["pid"] for r in pickle.load(f)}
    # DEVAM ETMENIN BOLME HATASI (2026-08-04, kendi hatam):
    # Her vardiya YALNIZ KENDI pkl'ini "zaten var" sayiyordu. Devam edildiginde dort
    # vardiyanin `hedef` listeleri FARKLILASIYOR ve `i % 4 == k` artik AYNI listeyi
    # bolmuyor: bazi parcalar IKI KEZ turetiliyor, bazilari HIC turetilmiyor.
    # (Belirti: hedef toplami 2484, oysa 2690-823 = 1867 olmaliydi.)
    # Dogrusu: bolmeyi belirleyen `var` kumesi TUM vardiyalarin ciktisini icermeli;
    # `OUT` ise yalniz bu vardiyanin kayitlarini tutmaya devam eder.
    for _k in range(a.toplam if a.toplam > 1 else 1):
        _f = f"results/_der_yeni{ek}_{_k}.pkl" if a.toplam > 1 else CIKTI
        if os.path.exists(_f):
            with open(_f, "rb") as _h:
                var |= {r["pid"] for r in pickle.load(_h)}
    OUT, GEOD = [], {}
    if os.path.exists(CIKTI):
        with open(CIKTI, "rb") as f:
            OUT = pickle.load(f)
        var |= {r["pid"] for r in OUT}
        if os.path.exists(GEO):
            GEOD = json.load(io.open(GEO, encoding="utf-8"))
        print(f"devam: {len(OUT)} parca zaten turetilmis", flush=True)

    if a.pids_file:
        _sec = {x.strip() for x in io.open(a.pids_file, encoding="utf-8") if x.strip()}
        E = [t for t in E if t[1] in _sec]
        print(f"PID FILTRESI: {len(_sec)} istendi -> {len(E)} bulundu", flush=True)
    hedef = dengeli_sira([t for t in E if t[1] not in var])
    if a.toplam > 1:
        hedef = [t for i, t in enumerate(hedef) if i % a.toplam == a.vardiya]
        print(f"VARDIYA {a.vardiya}/{a.toplam}", flush=True)
    if a.sinir:
        hedef = hedef[:a.sinir]
    print(f"uygun korpus {len(E)} | zaten var {len(var)} | TURETILECEK {len(hedef)}")
    print("  ilk 12 (uretici-dengeli):", [t[0] for t in hedef[:12]], flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = a.ckpt or cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    print(f"  ckpt: {[c.split('/')[-1] for c in cks]}", flush=True)
    models = [load_any(c, dev=dev)[:2] for c in cks]
    t0 = time.time(); hata = 0
    for k, (mfg, pid, jf, stp) in enumerate(hedef, 1):
        if k % 5 == 0:
            hiz = (time.time() - t0) / k
            kalan = hiz * (len(hedef) - k) / 60
            print(f"  {k}/{len(hedef)}  {time.time()-t0:.0f}s  hata={hata}  "
                  f"({hiz:.1f}s/parca, kalan ~{kalan:.0f} dk)", flush=True)
            with open(CIKTI, "wb") as f:
                pickle.dump(OUT, f)
            with io.open(GEO, "w", encoding="utf-8") as f:
                json.dump(GEOD, f)
            # DISK BEKCISI (2026-08-05, IKINCI kez vurdu): disk dolunca gmsh HATA VERMEZ,
            # 9s/parca -> 6-12 DK/parca'ya duser ([[disk-dolunca-gmsh-donuyor]]). Belirti
            # "yavasladi" oldugu icin saatler suclunun pesinde harcaniyor. Burada TEMIZ
            # duruyoruz: kayitlar zaten yazildi, yeniden calistirinca kaldigi yerden devam.
            _bos = shutil.disk_usage(".").free / 1e9
            if _bos < DISK_ESIK_GB:
                print(f"\n** DISK {_bos:.1f} GB < {DISK_ESIK_GB} GB -- TEMIZ DURULUYOR. "
                      f"{len(OUT)} kayit yazildi; yer acip yeniden baslatin. **", flush=True)
                break
        # HANGI PARCADA TAKILDI: her parcadan ONCE imza atilir. 2026-08-04'te turetme
        # ilk 5 parcanin birinde asili kaldi ve 5'te-bir yazdigim icin SUCLUYU bulamadim
        # (hafizada kayitli tuzak: bazi parcalar turetmeyi kilitliyor). Artik imza var.
        with io.open(ISARET, "w", encoding="utf-8") as _f:
            _f.write(f"{k}	{mfg}	{pid}	{stp}")
        if pid in ATLA:
            print(f"    {pid}: ATLANDI (kilitlenen parca listesinde)", flush=True)
            continue
        try:
            Vr, Fr = step_to_mesh(stp)
            GEOD[pid] = anahtar(Vr, Fr)          # D5-1 ayni gecisde
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            # OPERATOR ONBELLEGI PARCA-YEREL VE GECICI (2026-08-04, DISK ACIL DURUMU):
            # Eskiden paylasilan `results/step_infer/ops_k*` dizinine yaziyordu. Ama bu
            # gecis her parcayi BIR KEZ turetiyor; onbellek bir daha HIC okunmuyor, buna
            # karsilik parca basina ~9 MB disk yiyordu. 2690 parca = ~25 GB, ve diskte
            # 0.9 GB kalmisti -- vardiyalar bu yuzden surunuyordu.
            # Parca-yerel gecici dizin: AYNI parcada ayni k_eig'i paylasan modeller
            # operatoru yine yeniden kullanir (4 model -> 2 hesap), parca bitince silinir.
            tmp = tempfile.mkdtemp(prefix="ops_", dir=OPS_TMP)
            try:
                pbs = []
                for model, meta in models:
                    _, pb = D_.predict(model, meta, V, F, device=dev,
                                       op_cache_dir=os.path.join(
                                           tmp, f"k{int(meta.get('k_eig', 64))}"),
                                       return_probs=True)
                    pbs.append(np.asarray(pb, float))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            cps, probs, _, uyeler = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
            if not cps:
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); X = None; XR = None
            else:
                P = np.array([c["point"] for c in cps], float)
                Pd = np.array([c["direction"] for c in cps], float)
                X = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
                try:
                    XR = zengin(V, F, probs, cps, _normaller(V, F))
                except Exception:
                    XR = None
            j = json.load(io.open(jf, encoding="utf-8-sig"))
            g = j.get("ConnectionPoints") or []
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in g], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in g], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            R, t_, _ = cad_eval.align_frames(Vr, Vj)
            G = (G - t_) @ R; Gd = Gd @ R
            OUT.append({"pid": pid, "mfg": mfg, "geo": GEOD[pid], "kume": None,
                        "diag": float(np.linalg.norm(V.max(0) - V.min(0))),
                        "n": len(G), "P": P, "Pd": Pd, "X": X, "XR": XR,
                        "G": G, "Gd": Gd, "UYE": uyeler})
        except Exception as e:
            hata += 1
            if hata <= 5:
                print(f"    {pid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
    with open(CIKTI, "wb") as f:
        pickle.dump(OUT, f)
    with io.open(GEO, "w", encoding="utf-8") as f:
        json.dump(GEOD, f)
    c = collections.Counter(r["mfg"] for r in OUT)
    print(f"\n{len(OUT)} kayit -> {CIKTI} | hata {hata} | {time.time()-t0:.0f}s")
    print(f"  geometri anahtari -> {GEO} ({len(GEOD)} parca)")
    print(f"  uretici: {dict(c.most_common(12))}")
    print(f"  GT toplam {sum(r['n'] for r in OUT)} | aday {sum(len(r['P']) for r in OUT)}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
