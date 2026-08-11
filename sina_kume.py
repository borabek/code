# -*- coding: utf-8 -*-
"""KUME SINAVI: bir kumede secilen kararlar BASKA geometrilerde de tutuyor mu?

ZAAFIYET (kullanicinin isaret ettigi): J (konum ortalamasi) ve L2 (3 ezberci ozelligi at)
kararlari 100 parcalik TEK bir kume uzerinde SECILDI ve ayni kume uzerinde OLCULDU. Boyle bir
sayi "holdout" degildir; secim gurultuye uymus olabilir.

BU BETIK: ayni kararlari, geometri olarak AYRIK bir kumede acik/kapali olcer ve farkin
gurultuden ayirt edilebilir olup olmadigini ESLI BOOTSTRAP ile soyler.
  - DEV'de kazanan VAL'de de kazaniyorsa -> karar gercek, KILITLI harcanmadan dogrulanmis olur.
  - DEV'de kazanip VAL'de kaybediyorsa -> asiri-uydurma; karar geri alinir.

Gate egitimi test geometri gruplarini GORMEZ (keskin anahtar). LOCKED'a DOKUNULMAZ.

Kullanim:  python sina_kume.py val|dev|locked
"""
import os, sys, json, pickle
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}
# L2'nin attigi ozelliklerden SONRA kalan sutunlar. Uruna bakip okumuyoruz: L2 geri alindiktan
# sonra dagitilan gate 13 sutunlu, o yuzden pkl'den okumak ablasyonu VAKUM yapardi (dev kosusunda
# tam olarak bu oldu: "L2 kapali" satiri "URUN" ile birebir ayni cikti).
L2COLS = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]


def esle(P, Pd, G, Gd, diag, tol, am, pct, isaretli=False):
    """big_arbiter eksen-farkindali esleme -> (TP, FP, FN).

    `isaretli=False` (varsayilan, DAGITILAN davranis): aci `abs(Pd . Gd)` ile olculur, yani
    180 derece ters bir tahmin 0 derece sayilir. Bu bir EKSEN metrigidir.

    `isaretli=True`: isaret KORUNUR -- tahmin, ureticinin `InsertDirection`'iyla AYNI yone
    bakmak zorundadir. Fiziksel robot icin dogru olcut budur.

    OLCULDU (2026-08-02, 194 parca / 771 eslesen cift): eslesen ciftlerin %89.8'inde
    `Pd . Gd > 0` (medyan +1.000), yani dagitilan yon ZATEN takma yonudur ve
    `robot_cp` docstring'i dogrudur. Isaretli metrik 0.5893 -> 0.5818, kayip yalniz 0.0075.
    (Ters sozlesme `-Pd` denendi: 0.0907 ile COKUYOR -> sozlesme kesin olarak `+Pd`.)
    """
    return esle_detay(P, Pd, G, Gd, diag, tol, am, pct, isaretli)[:3]


# EKSENEL TOLERANS -- A1 (2026-08-04): SIKILASTIRMA ONERILDI, OLCUM REDDETTI. 40.0 KALIR.
#
# Oneri: "40mm keyfi, 15mm'ye cekilsin". UC olcum bunu curuttu:
#   1. TANIM FARKI DEGIL: seat->agiz mesafesi korpus capinda medyan **0.00mm** (%53'u tam
#      sifir; p90 13.1mm). Otopsideki 3 parcada 7.5-10.6mm cikmisti ama TEMSILI DEGILMIS.
#      GT'yi kendi agzina tasimak sonucu neredeyse hic degistirmedi (0.5214 -> 0.5280).
#   2. ARTEFAKT DEGIL AMA URETICIYE BAGLI: 15mm'yi asan 174 eslesme 187 parcaya dagilmis
#      (en kotu 10 parca yalnizca %27'sini tutuyor) AMA uretici kirilimi WEI %35 / PXC %12.
#      Uc kat fark = ureticinin CP'yi hangi DERINLIKTE tanimladigi, yani KONVANSIYON.
#   3. FIZIK: eksenel kayma robot icin EN ZARARSIZ hata boyutu -- robot zaten o eksen
#      boyunca yaklasir, kayma "ne kadar derine" demektir. Deligi kacirtan YANAL hata,
#      yaklasimi bozan ACI hatasidir; ikisi de ayrica olculuyor.
# 15mm'ye cekmek tespiti 0.7584 -> 0.5280 dusururdu ve kesilenin buyuk kismi KONVANSIYON
# olurdu: metrigi duzeltmek yerine URUNU cezalandirmak. Bu yuzden 40.0 KALDI.
#
# GERCEK sisme BASKA YERDE ve A2 ile raporlanir: TP'lerin %25.2'sinin kabul kutusunda
# BASKA bir GT var -> F1 0.7584 iken F1_kesin 0.6150. Manset artik bu BANDI da verir.
EKSEN_TOL_ESKI = 40.0
EKSEN_TOL = 40.0


def esle_detay(P, Pd, G, Gd, diag, tol, am, pct, isaretli=False, eksen_tol=EKSEN_TOL_ESKI):
    """`esle` + eslesme AYRINTISI. Dondurur: (tp, fp, fn, bilgi).

    `bilgi["eslesme"]` : her TP icin (tahmin_idx, gt_idx, yanal, eksenel, aci, belirsiz_mi)
    `bilgi["belirsiz"]`: kac TP'nin KABUL KUTUSUNDA birden fazla GT vardi (A2).

    BELIRSIZ ESLESME NEDIR (A2): bir tahmin, kabul kutusuna (yanal<=tt, |eksenel|<=eksen_tol,
    aci<=am) BIRDEN FAZLA GT siginiyorsa, "dogru" sayilan eslesme fiziksel olarak KOMSU
    DELIGE ait olabilir. Olculdu (2026-08-04, 194 parca): GT'lerin **%29.3'unun** tolerans
    kutusunda baska bir GT var (dusuk-CP %20.8, cok-CP %33.0). Bu, F1'in UST SINIR
    sisintisidir; `F1_kesin` yalniz tek-adayli eslesmeleri sayar.

    `eksen_tol` varsayilani ESKI davranisla (40.0) BIREBIR ayni kalir -- mevcut cagiranlarin
    hicbiri degismez. Yeni olcum EKSEN_TOL (15.0) ile ACIKCA cagirir.
    """
    hit = np.zeros(len(G), bool); used = set()
    eslesme = []
    if len(P) and len(G):
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        _c = Pd @ Gd.T
        an = np.degrees(np.arccos(np.clip(_c if isaretli else np.abs(_c), -1, 1)))
        tt = max(3.0, 0.06 * diag) if pct else tol
        kabul = (np.abs(al) <= eksen_tol) & (an <= am) & (pe <= tt)
        pe = np.where((np.abs(al) > eksen_tol) | (an > am), np.inf, pe)
        for d_, a_, b_ in sorted((pe[a, b], a, b)
                                 for a in range(len(P)) for b in range(len(G))):
            if d_ > tt or a_ in used or hit[b_]:
                continue
            hit[b_] = True; used.add(a_)
            eslesme.append((int(a_), int(b_), float(d_), float(al[a_, b_]),
                            float(an[a_, b_]), bool(kabul[a_].sum() > 1)))
    tp = int(hit.sum())
    bilgi = {"eslesme": eslesme, "belirsiz": sum(1 for e in eslesme if e[5]),
             "tt": float(max(3.0, 0.06 * diag) if pct else tol), "eksen_tol": float(eksen_tol)}
    return tp, len(P) - tp, len(G) - tp, bilgi


def esle_macar(P, Pd, G, Gd, diag, tol, am, pct, isaretli=False,
               eksen_tol=EKSEN_TOL_ESKI):
    """`esle_detay` ile AYNI kabul kutusu, ama atama OPTIMAL (Hungarian).

    NEDEN (2026-08-06): `esle_detay` acgozlu -- ciftleri mesafeye gore sirali gezip ilk
    uyani baglar. Kalabalik parcalarda bu ATAMA KAYBI uretir: bir tahmin, kendisine daha
    uygun bir GT'yi baska (daha yakin ama zaten eslesmis) bir tahmine kaptirabilir ve GT
    bos kalir. Otopside "KALABALIK" kovasi GT'nin **%12.8'i** idi ve bu kovanin bir kismi
    GERCEK bilgi eksigi degil, ATAMA sirasinin artefaktidir.

    Macar yontemi TP'yi ENBUYUKLER; boylece "kalabalik" kovasinin ne kadari saf atama
    kaybiymis, olculebilir hale gelir. Kabul kutusu (yanal/eksenel/aci) BIREBIR aynidir --
    yani bu bir tolerans gevsetmesi DEGIL, ayni kutu icinde daha iyi eslestirme.

    Doner: `esle_detay` ile ayni imza (tp, fp, fn, bilgi).
    """
    from scipy.optimize import linear_sum_assignment
    hit = np.zeros(len(G), bool)
    eslesme = []
    tt = max(3.0, 0.06 * diag) if pct else tol
    if len(P) and len(G):
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        _c = Pd @ Gd.T
        an = np.degrees(np.arccos(np.clip(_c if isaretli else np.abs(_c), -1, 1)))
        kabul = (np.abs(al) <= eksen_tol) & (an <= am) & (pe <= tt)
        # Kabul edilmeyen ciftlere BUYUK bedel; optimal atama onlari secmeye zorlanmasin.
        BUYUK = 1e6
        C = np.where(kabul, pe, BUYUK)
        ri, ci = linear_sum_assignment(C)
        for a_, b_ in zip(ri, ci):
            if not kabul[a_, b_]:
                continue
            hit[b_] = True
            eslesme.append((int(a_), int(b_), float(pe[a_, b_]), float(al[a_, b_]),
                            float(an[a_, b_]), bool(kabul[a_].sum() > 1)))
    tp = int(hit.sum())
    return tp, len(P) - tp, len(G) - tp, {
        "eslesme": eslesme, "belirsiz": sum(1 for e in eslesme if e[5]),
        "tt": float(tt), "eksen_tol": float(eksen_tol), "yontem": "macar"}


def f1w(rows):
    """parca basina (rejim,TP,FP,FN) -> KORPUS-agirlikli F1 (dusuk 0.895 / cok 0.105).

    !! TUZAK (2026-08-01'de beni yakaladi): agirliklar SABITTIR ve HER IKI rejime de uygulanir.
    Tek rejimli bir alt kume verirsen (ornegin yalniz cok-CP parcalar), diger rejim sifir sayilir
    ve sonuc o rejimin AGIRLIGIYLA CARPILMIS olur -- yani 0.105 x gercek deger. Manset betiginde
    tam bunu yaptim ve "cok-CP F1 = 0.069" diye bir COKUS gordum; gercegi 0.653'tu.
    REJIM KIRILIMI ICIN `f1_rejim()` KULLAN, bu fonksiyonu alt kumeyle CAGIRMA.

    DUZELTME 2026-08-03 (denetimde bit-duzeyinde yakalandi): yukaridaki uyari "rejime gore
    alt kume alma" diye yazilmisti, ama ayni tuzaga BASKA kapidan da dusuluyordu -- bazi
    bolmeler DOGAL OLARAK tek rejimli (o parcalarin hepsi dusuk-CP). O bolmelerde eksik
    rejim F1=0 sayilip 0.895 ile carpiliyordu. OLCULDU: ATANMAMIS 0.6635 (gercegi 0.7413),
    seri 16 0.8333 (0.9310), seri 17 0.6531 (0.7297) -- ucunde de rapor/0.895 == dusuk_CP_F1
    TAM tutuyordu. Manset (HAVUZLANMIS) ve uretici-disi bolmeler ETKILENMEDI cunku ikisinde
    de her iki rejim var. Cozum: agirliklar VAR OLAN rejimler uzerinden yeniden normalize
    edilir -- `f1_rejim()` zaten boyle yapiyordu, bu fonksiyon yapmiyordu.
    """
    agg = {}
    for k, t, f, n in rows:
        a = agg.setdefault(k, [0, 0, 0])
        a[0] += t; a[1] += f; a[2] += n
    if not agg:
        return 0.0
    o = {}
    for k, (T, Fp, Fn) in agg.items():
        p_ = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
        o[k] = 2 * p_ * rc / max(p_ + rc, 1e-9)
    top = sum(W[k] for k in o if k in W)
    if top <= 0:
        return float(np.mean(list(o.values())))
    return sum(W[k] * o[k] for k in o if k in W) / top


def f1_rejim(rows):
    """Rejim KIRILIMI: her rejim icin AGIRLIKSIZ F1 + sayilar + agirlikli toplam.

    f1w'nin alt kumeyle cagrilmasi tuzagina karsi dogru arac. Ayrica P/R'yi de rejim-agirlikli
    verir: `pr()` ham havuzdan hesaplar ve puanlanan kume korpustan CARPIKSA (olculdu: puanlanan
    kumede cok-CP %30, korpusta %10.5) kesinlik/recall yaniltir -- F1 agirlikli, P/R degilse
    ayni tabloda IKI FARKLI evren raporlanmis olur.
    """
    agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
    say = {"dusuk": 0, "cok": 0}
    for k, t, f, n in rows:
        agg[k][0] += t; agg[k][1] += f; agg[k][2] += n; say[k] += 1
    o, pp, rr = {}, {}, {}
    for k, (T, Fp, Fn) in agg.items():
        p_ = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
        o[k] = 2 * p_ * rc / max(p_ + rc, 1e-9) if say[k] else None
        pp[k], rr[k] = (p_, rc) if say[k] else (None, None)
    var = [k for k in W if say[k]]
    top = sum(W[k] for k in var) or 1.0
    return {"F1": o, "n": say,
            "agirlikli_F1": sum(W[k] * (o[k] or 0.0) for k in var) / top,
            "agirlikli_kesinlik": sum(W[k] * (pp[k] or 0.0) for k in var) / top,
            "agirlikli_recall": sum(W[k] * (rr[k] or 0.0) for k in var) / top,
            "ham_kesinlik": pr(rows)[0], "ham_recall": pr(rows)[1]}


def pr(rows):
    """HAM (agirliksiz) havuzlanmis kesinlik/recall. Carpik kumede yaniltir -> f1_rejim kullan."""
    T = sum(r[1] for r in rows); Fp = sum(r[2] for r in rows); Fn = sum(r[3] for r in rows)
    return T / max(T + Fp, 1), T / max(T + Fn, 1)


def main():
    import cp_openings, robot_cp, wire_gate
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from j_konum_ortalama import vote_avg

    kume = (sys.argv[1] if len(sys.argv) > 1 else "val").lower()
    cf = f"results/_probs_{kume}.pkl"
    if not os.path.exists(cf) and kume == "dev":
        cf = "results/_h_probs.pkl"            # DEV onbellegi eski adla duruyor
    cache = pickle.load(open(cf, "rb"))
    for r in cache:                            # eski onbellekte pid alani yok
        r.setdefault("pid", os.path.basename(r["stp"]).split("_")[1])

    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    THR = {"dusuk": float(cfg["robot_wire_gate_threshold"]),
           "cok": float(cfg["robot_wire_gate_threshold_highcp"])}

    # --- gate: test geometri gruplarini disla (KESKIN anahtar) ---
    d = np.load("results/gate_regrow_data_rt2.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]
    pids = np.array([str(x) for x in d["pids"]])
    gk = json.load(open("results/_strict_geometry_keys.json"))
    tg = {gk.get(r["pid"], "yok:" + r["pid"]) for r in cache}
    Gg = np.array([gk.get(p, "yok:" + p) for p in pids])
    keep = ~np.isin(Gg, list(tg))
    print(f"KUME={kume} | {len(cache)} parca | gate {int(keep.sum())}/{len(y)} aday "
          f"({len(tg)} test geometri grubu dislandi) | esikler {THR}", flush=True)

    def gate(cols):
        Xk = X[keep][:, cols] if cols else X[keep]
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                      random_state=0).fit(Xk, y[keep])
    CLF = {True: gate(L2COLS), False: gate(None)}

    # --- TURETME: yalnizca J'ye bagli, gate'e degil -> 4 yapilandirma 2 turetmeyle kosar ---
    DER = {}
    for j_on in (True, False):
        rows = []
        for r in cache:
            V = np.ascontiguousarray(r["V"], np.float64)
            F = np.ascontiguousarray(r["F"], np.int64)
            plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
            mk = lambda pr_, **kw: cp_openings.connection_points(
                V, F, pr_.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                probs=pr_, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                step_path=r["stp"], **kw)
            merge = ((lambda L: vote_avg(L, min_votes=1, mode="wmean")) if j_on
                     else (lambda L: robot_cp._vote2(L, min_votes=1)))
            base = merge([mk(pb) for pb in plist])
            is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
            cps = merge([mk(pb, conn_promote=0.25) for pb in plist]) if is_hi else base
            Xc = None
            if cps:
                Xc = wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT)
            rows.append(dict(
                X=Xc, is_hi=is_hi,
                P=np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3)),
                Pd=np.array([c["direction"] for c in cps], float) if cps else np.zeros((0, 3)),
                G=r["G"], Gd=r["Gd"], n=r["n"], diag=r["diag"]))
        DER[j_on] = rows
        print(f"  turetme J={'acik' if j_on else 'kapali'} bitti "
              f"({sum(len(x['P']) for x in rows)} aday)", flush=True)

    def kos(j_on, l2_on):
        clf = CLF[l2_on]; cols = L2COLS if l2_on else None
        det, rob = [], []
        for r in DER[j_on]:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if r["X"] is not None:
                sc = clf.predict_proba(r["X"][:, cols] if cols else r["X"])[:, 1]
                m = sc >= (THR["cok"] if r["is_hi"] else THR["dusuk"])
                if m.any():
                    P = r["P"][m]; Pd = r["Pd"][m]
            k = "cok" if r["n"] >= 8 else "dusuk"
            det.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob.append((k,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        return det, rob

    print(f"\n{'yapilandirma':<26}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    PP, res = {}, {}
    for lab, j_, l_ in (("URUN (J, 13 ozellik)", True, False), ("J kapali", False, False),
                        ("L2 acik", True, True), ("J kapali + L2 acik", False, True)):
        det, rob = kos(j_, l_)
        PP[lab] = (det, rob)
        p_, r_ = pr(det)
        res[lab] = {"tespit": f1w(det), "robot": f1w(rob), "kesinlik": p_, "recall": r_}
        print(f"{lab:<26}{f1w(det):>9.4f}{f1w(rob):>9.4f}{p_:>9.3f}{r_:>9.3f}", flush=True)

    pickle.dump(PP, open(f"results/sinav_{kume}_parca.pkl", "wb"))   # sonraki analiz bedava
    rng = np.random.RandomState(0)
    npart = len(PP["URUN (J, 13 ozellik)"][0])
    IX = [rng.randint(0, npart, npart) for _ in range(2000)]

    print(f"\nTEK YAPILANDIRMA %95 GA ({npart} parca, 2000 tekrar) -- iki AYRI kumenin farki "
          f"gurultu mu, bunun icin:")
    for mi, mn in ((0, "tespit"), (1, "robot ")):
        rows = PP["URUN (J, 13 ozellik)"][mi]
        bs = np.array([f1w([rows[i] for i in ix]) for ix in IX])
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f"  URUN {mn} = {f1w(rows):.4f}   [{lo_:.4f}, {hi_:.4f}]", flush=True)
        res.setdefault("GA", {})[mn.strip()] = [float(f1w(rows)), float(lo_), float(hi_)]

    print("\nESLI BOOTSTRAP (ayni parcalar iki yapilandirmada da secilir):")
    print(f"  {'karsilastirma':<38}{'fark':>9}{'%95 GA':>21}{'karar':>10}")
    for lab in ("J kapali", "L2 acik", "J kapali + L2 acik"):
        for mi, mn in ((0, "tespit"), (1, "robot")):
            a = PP["URUN (J, 13 ozellik)"][mi]; b = PP[lab][mi]
            ds = np.array([f1w([a[i] for i in ix]) - f1w([b[i] for i in ix]) for ix in IX])
            lo_, hi_ = np.percentile(ds, [2.5, 97.5])
            kar = "BELIRGIN" if lo_ > 0 or hi_ < 0 else "gurultu"
            print(f"  {'URUN - ' + lab + ' (' + mn + ')':<38}{ds.mean():>+9.4f}"
                  f"{'[' + format(lo_, '+.4f') + ', ' + format(hi_, '+.4f') + ']':>21}"
                  f"{kar:>10}", flush=True)
            res.setdefault("bootstrap", {})[f"{lab}|{mn}"] = [float(ds.mean()), float(lo_), float(hi_)]

    print("\nKARAR KURALI: bir kumede kazanip digerinde KAYBEDEN kol asiri-uydurmadir, geri alinir.")
    json.dump(res, open(f"results/sinav_{kume}.json", "w"), indent=1)
    print(f"makbuz -> results/sinav_{kume}.json")


if __name__ == "__main__":
    main()
