# -*- coding: utf-8 -*-
"""T1: KARAR KURALI GATE'DEN SONRA YENIDEN AYARLANDI MI? (tespit kolu)

GEREKCE: gate 2026-08-02'de ZENGIN oznitelik kumesiyle YENIDEN EGITILDI (116 sutun,
+110 yeni WEI parcasi). Kabul kurali ise
    (s >= ORAN * max(s)) & (s >= TABAN),  ORAN=0.5, TABAN=0.25
degerleriyle duruyor. Bu proje ayni cinsten bayatligi IKI KEZ yasadi:
  * [[gate-refit-minv4]] -- dagitilan gate topyekun bayatti, refit +0.1273
  * [[topoloji-yaricapi-kapandi]] -- modele bagli ayar modelden bagimsiz degistirilebiliyordu
Model degisti; ona bagli esikler degismedi. Once bunu olcmek, yeni bir mekanizma
aramaktan ONCE gelir.

DURUSTLUK PROTOKOLU (bu betigin ASIL meselesi): esik taramasi, tarandigi kumede
KESINLIKLE sisirir. O yuzden:
    TARAMA yalniz DEV (54 parca) uzerinde yapilir,
    HUKUM yalniz VAL (100 parca) uzerinde okunur,
    secim VAL'e BAKILMADAN once yapilir ve tek bir hucre raporlanir.
Havuzlanmis sayi da basilir ama SECIM ICIN KULLANILMAZ (yalniz bilgi).

KILL (onceden yazildi): VAL tespitinde +0.005 ve GA sifiri disliyor. Aksi halde
mevcut kural KALIR.
"""
import io, json, os, sys
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import olcum_kumesi, wire_gate
from sina_kume import esle, f1w

ORANLAR = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
TABANLAR = (0.15, 0.20, 0.25, 0.30, 0.35)


def main():
    DER, gate, ek = T2.yukle()
    s3 = json.load(io.open("results/split3.json", encoding="utf-8"))
    dev = {str(p) for p in s3["dev"]["parts"]}
    val = {str(p) for p in s3["val"]["parts"]}

    # --- skorlari ve duzeltilmis CP'leri BIR KEZ hesapla (kural skoru degistirmez)
    HAZ = []
    for r in DER:
        rec = {"pid": r["pid"], "geo": r["geo"], "rj": "cok" if r["n"] >= 8 else "dusuk",
               "G": np.asarray(r["G"], float), "Gd": np.asarray(r["Gd"], float),
               "diag": r["diag"], "sk": None}
        if r["X"] is not None and r.get("XR") is not None:
            X = np.hstack([r["X"], r["XR"]])
            if X.shape[1] * 2 == gate["n_feat"]:
                rec["sk"] = wire_gate.karar_skoru(gate, X)
                rec["X"] = X; rec["P"] = np.asarray(r["P"], float)
                rec["Pd"] = np.asarray(r["Pd"], float); rec["UYE"] = r.get("UYE")
        HAZ.append(rec)
    print(f"hazir: {sum(1 for h in HAZ if h['sk'] is not None)}/{len(HAZ)} parcada skor var",
          flush=True)

    def kos(oran, taban, kume):
        det, rob, gg = [], [], []
        for h in HAZ:
            if kume is not None and h["pid"] not in kume:
                continue
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
            det.append((h["rj"],) + esle(P, Pd, h["G"], h["Gd"], h["diag"], 0.0, 180.0, True))
            rob.append((h["rj"],) + esle(P, Pd, h["G"], h["Gd"], h["diag"], 2.0, 10.0,
                                         False, isaretli=True))
            gg.append(h["geo"])
        return det, rob, gg

    # ---------- 1) TARAMA: YALNIZ DEV
    print(f"\n--- TARAMA (YALNIZ DEV, {len(dev)} parca) ---")
    print(f"{'oran':>6}" + "".join(f"{t:>9.2f}" for t in TABANLAR))
    TAR = {}
    for o in ORANLAR:
        sat = f"{o:>6.2f}"
        for t in TABANLAR:
            d_, _r, _g = kos(o, t, dev)
            v = f1w(d_); TAR[(o, t)] = v
            sat += f"{v:>9.4f}"
        print(sat, flush=True)
    mev = TAR[(0.50, 0.25)]
    en = max(TAR, key=TAR.get)
    print(f"\nDEV'de mevcut kural (0.50/0.25): {mev:.4f}")
    print(f"DEV'de en iyi hucre: oran {en[0]:.2f} / taban {en[1]:.2f} -> {TAR[en]:.4f} "
          f"({TAR[en]-mev:+.4f})")

    # ---------- 2) HUKUM: YALNIZ VAL, TEK hucre
    print(f"\n--- HUKUM (YALNIZ VAL, {len(val)} parca; DEV'de secilen TEK hucre) ---")
    d0, r0, gg = kos(0.50, 0.25, val)
    d1, r1, _ = kos(en[0], en[1], val)
    fn = lambda rows: f1w([q for _, q in rows]) - f1w([p for p, _ in rows])
    _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(d0, d1)), gg, fn, n=3000)
    _, rlo, rhi = olcum_kumesi.grup_bootstrap(list(zip(r0, r1)), gg, fn, n=3000)
    dd = f1w(d1) - f1w(d0)
    print(f"{'kural':<20}{'tespit':>10}{'robot(FIZ)':>13}")
    print(f"{'mevcut 0.50/0.25':<20}{f1w(d0):>10.4f}{f1w(r0):>13.4f}")
    print(f"{f'DEV-secimi {en[0]:.2f}/{en[1]:.2f}':<20}{f1w(d1):>10.4f}{f1w(r1):>13.4f}")
    print(f"\nVAL tespit farki: {dd:+.4f}  GA[{lo:+.4f},{hi:+.4f}] "
          f"{'GERCEK' if (lo > 0 or hi < 0) else 'gurultu'}")
    print(f"VAL robot  farki: {f1w(r1)-f1w(r0):+.4f}  GA[{rlo:+.4f},{rhi:+.4f}]")

    # ---------- 3) BILGI: havuzlanmis (SECIM ICIN KULLANILMAZ)
    dh0, rh0, ggh = kos(0.50, 0.25, None)
    dh1, rh1, _ = kos(en[0], en[1], None)
    print(f"\n[bilgi] HAVUZLANMIS 194 parca: tespit {f1w(dh0):.4f} -> {f1w(dh1):.4f} "
          f"({f1w(dh1)-f1w(dh0):+.4f}) | robot {f1w(rh0):.4f} -> {f1w(rh1):.4f}")
    print("   (bu satir HUKUM DEGIL: kural DEV'de secildi, havuz DEV'i de iceriyor)")

    gecti = dd >= 0.005 and lo > 0
    print(f"\nKILL: VAL tespit +0.005 VE GA>0 -> "
          f"{'GECTI -- kural guncellenir' if gecti else 'GECMEDI -- mevcut kural KALIR'}")
    with io.open("results/t1_karar_kurali.json", "w", encoding="utf-8") as f:
        json.dump({"dev_tarama": {f"{o}_{t}": v for (o, t), v in TAR.items()},
                   "dev_mevcut": mev, "dev_en_iyi": {"oran": en[0], "taban": en[1],
                                                     "f1": TAR[en]},
                   "val_mevcut_tespit": f1w(d0), "val_yeni_tespit": f1w(d1),
                   "val_d_tespit": dd, "val_ga": [lo, hi],
                   "val_mevcut_robot": f1w(r0), "val_yeni_robot": f1w(r1),
                   "havuz_mevcut": f1w(dh0), "havuz_yeni": f1w(dh1),
                   "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/t1_karar_kurali.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
