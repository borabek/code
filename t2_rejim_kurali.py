# -*- coding: utf-8 -*-
"""T2: REJIME KOSULLU KABUL KURALI -- olu koda donmus bir kazanci geri al (ya da gomm).

BULGU (2026-08-04, kod incelemesi): `gate_goreli_esik: True` iken
`wire_gate.karar_maskesi` `esik` parametresini TAMAMEN YOK SAYIYOR:

    if GORELI_ESIK:
        return (s >= GORELI_ORAN*max(s)) & (s >= GORELI_TABAN)    # esik kullanilmaz
    return s >= esik

Oysa `robot_cp` hala rejime gore esik HESAPLIYOR ve `apply(threshold=_thr)` diye
geciriyor (`robot_wire_gate_threshold_highcp` = 0.35). O deger URUNDE OLU.
[[pitstop-highcp-gate]] kaydinda rejime kosullu esik OLCULMUS bir kazancti
(yuksek-CP'de aday recall 0.798 iken dagitilan 0.485 idi). Goreli kural bunu ya
KAPSIYOR ya da sessizce GERI ALDI. Varsayilamaz -- olculur.

SORU: goreli kuralin PARAMETRELERI (oran, taban) rejime gore AYRILIRSA kazanc var mi?

REJIM YONLENDIRMESI GT'YE DOKUNMAZ (kritik): olcum tarafinda rejim `r["n"]>=8` ile,
yani GT'deki CP SAYISIYLA belirleniyor -- bu bir RAPORLAMA agirligidir, KARAR degiskeni
olarak kullanilamaz (sizinti). Burada yonlendirme yalniz ADAY SAYISIYLA yapilir; bu
tamamen boru hattindan gelir, GT gormez. Yonlendirmenin GT rejimiyle uyumu da raporlanir.

PROTOKOL: AYAR = dev + atanmamis (94 parca), HUKUM = val (100 parca).
DEV tek basina (54) rejime bolununce cok kucuk kalirdi; VAL'e HIC dokunulmaz.

KILL (onceden yazildi): VAL tespitinde +0.005 VE GA sifiri disliyor.
"""
import io, json, os, sys
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import olcum_kumesi, wire_gate
from sina_kume import esle, f1w

ORANLAR = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
TABANLAR = (0.15, 0.20, 0.25, 0.30, 0.35)
AYIRIM = 25          # aday sayisi >= bu ise "cok aday" kolu (GT GORMEZ)


def main():
    DER, gate, ek = T2.yukle()
    s3 = json.load(io.open("results/split3.json", encoding="utf-8"))
    val = {str(p) for p in s3["val"]["parts"]}
    HAZ = []
    for r in DER:
        h = {"pid": r["pid"], "geo": r["geo"], "rj": "cok" if r["n"] >= 8 else "dusuk",
             "G": np.asarray(r["G"], float), "Gd": np.asarray(r["Gd"], float),
             "diag": r["diag"], "sk": None, "na": 0}
        if r["X"] is not None and r.get("XR") is not None:
            X = np.hstack([r["X"], r["XR"]])
            if X.shape[1] * 2 == gate["n_feat"]:
                h["sk"] = wire_gate.karar_skoru(gate, X)
                h["X"] = X; h["P"] = np.asarray(r["P"], float)
                h["Pd"] = np.asarray(r["Pd"], float); h["UYE"] = r.get("UYE")
                h["na"] = len(h["sk"])
        h["kol"] = "cok_aday" if h["na"] >= AYIRIM else "az_aday"
        HAZ.append(h)
    # yonlendirmenin GT rejimiyle uyumu (yalniz RAPOR)
    a = np.array([h["kol"] == "cok_aday" for h in HAZ])
    b = np.array([h["rj"] == "cok" for h in HAZ])
    print(f"yonlendirme (aday>={AYIRIM}): cok_aday {int(a.sum())} / az_aday {int((~a).sum())}")
    print(f"  GT rejimiyle uyum: {float((a == b).mean()):.1%} "
          f"(yonlendirme GT GORMEZ, bu yalniz bilgi)")

    def puanla(h, oran, taban):
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
        if h["sk"] is not None and len(h["sk"]):
            s = h["sk"]
            k = (s >= oran * max(float(s.max()), 1e-9)) & (s >= taban)
            if k.any():
                P = h["P"][k].copy(); Pd = h["Pd"][k].copy()
                c = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                c = wire_gate.pose_duzelt(h["X"][k], c)
                c = wire_gate.aci_duzelt(h["X"][k], c)
                if h.get("UYE"):
                    c = wire_gate.uye_yonu_sec(h["X"][k], c, h["UYE"])
                P = np.array([x["point"] for x in c], float)
                Pd = np.array([x["direction"] for x in c], float)
        return (esle(P, Pd, h["G"], h["Gd"], h["diag"], 0.0, 180.0, True),
                esle(P, Pd, h["G"], h["Gd"], h["diag"], 2.0, 10.0, False, isaretli=True))

    # --- tum hucreleri BIR KEZ hesapla (parca x oran x taban)
    print(f"\n{len(ORANLAR)*len(TABANLAR)} hucre x {len(HAZ)} parca hesaplaniyor...", flush=True)
    HUC = {}
    for o in ORANLAR:
        for t in TABANLAR:
            HUC[(o, t)] = [puanla(h, o, t) for h in HAZ]
        print(f"  oran {o:.2f} bitti", flush=True)

    ayar = [j for j, h in enumerate(HAZ) if h["pid"] not in val]
    hukum = [j for j, h in enumerate(HAZ) if h["pid"] in val]
    print(f"AYAR {len(ayar)} parca | HUKUM(VAL) {len(hukum)} parca")

    def f1_of(idx, sec):
        """sec: kol -> (oran,taban).  det, rob dondurur."""
        det = [(HAZ[j]["rj"],) + HUC[sec[HAZ[j]["kol"]]][j][0] for j in idx]
        rob = [(HAZ[j]["rj"],) + HUC[sec[HAZ[j]["kol"]]][j][1] for j in idx]
        return f1w(det), f1w(rob), det, rob, [HAZ[j]["geo"] for j in idx]

    MEV = {"cok_aday": (0.50, 0.25), "az_aday": (0.50, 0.25)}
    # --- AYAR kumesinde kol basina en iyi (digeri mevcutta sabit tutulur)
    EN = dict(MEV)
    for kol in ("az_aday", "cok_aday"):
        alt = [j for j in ayar if HAZ[j]["kol"] == kol]
        if len(alt) < 10:
            print(f"  {kol}: yalniz {len(alt)} parca -- AYARLANMAZ, mevcut kalir")
            continue
        en, eb = None, None
        for o in ORANLAR:
            for t in TABANLAR:
                v = f1w([(HAZ[j]["rj"],) + HUC[(o, t)][j][0] for j in alt])
                if en is None or v > en:
                    en, eb = v, (o, t)
        mv = f1w([(HAZ[j]["rj"],) + HUC[MEV[kol]][j][0] for j in alt])
        print(f"  {kol} ({len(alt)} parca): mevcut {mv:.4f} -> en iyi {eb} {en:.4f} ({en-mv:+.4f})")
        EN[kol] = eb

    print(f"\nAYAR kumesinde secilen kural: {EN}")
    fn = lambda rows: f1w([q for _, q in rows]) - f1w([p for p, _ in rows])
    d0, r0, D0, R0, gg = f1_of(hukum, MEV)
    d1, r1, D1, R1, _ = f1_of(hukum, EN)
    _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(D0, D1)), gg, fn, n=3000)
    _, rlo, rhi = olcum_kumesi.grup_bootstrap(list(zip(R0, R1)), gg, fn, n=3000)
    print(f"\n--- HUKUM: VAL ({len(hukum)} parca) ---")
    print(f"{'kural':<26}{'tespit':>10}{'robot(FIZ)':>13}")
    print(f"{'mevcut (tek kural)':<26}{d0:>10.4f}{r0:>13.4f}")
    print(f"{'rejime kosullu':<26}{d1:>10.4f}{r1:>13.4f}")
    print(f"\nVAL tespit farki: {d1-d0:+.4f}  GA[{lo:+.4f},{hi:+.4f}] "
          f"{'GERCEK' if (lo > 0 or hi < 0) else 'gurultu'}")
    print(f"VAL robot  farki: {r1-r0:+.4f}  GA[{rlo:+.4f},{rhi:+.4f}]")
    ha0 = f1_of(range(len(HAZ)), MEV); ha1 = f1_of(range(len(HAZ)), EN)
    print(f"\n[bilgi] HAVUZLANMIS: tespit {ha0[0]:.4f} -> {ha1[0]:.4f} | "
          f"robot {ha0[1]:.4f} -> {ha1[1]:.4f}   (HUKUM DEGIL: ayar kumesi havuzda)")
    gecti = (d1 - d0) >= 0.005 and lo > 0
    print(f"\nKILL: VAL tespit +0.005 VE GA>0 -> "
          f"{'GECTI' if gecti else 'GECMEDI -- tek kural KALIR'}")
    with io.open("results/t2_rejim_kurali.json", "w", encoding="utf-8") as f:
        json.dump({"ayirim_aday": AYIRIM, "secilen": {k: list(v) for k, v in EN.items()},
                   "val_mevcut": {"tespit": d0, "robot": r0},
                   "val_yeni": {"tespit": d1, "robot": r1},
                   "val_d_tespit": d1 - d0, "ga_tespit": [lo, hi],
                   "val_d_robot": r1 - r0, "ga_robot": [rlo, rhi],
                   "havuz_mevcut": {"tespit": ha0[0], "robot": ha0[1]},
                   "havuz_yeni": {"tespit": ha1[0], "robot": ha1[1]},
                   "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/t2_rejim_kurali.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
