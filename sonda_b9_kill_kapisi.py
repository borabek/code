"""g10 KILL KAPISI: sig boyama + duzeltilmis secim sinyali, g7'yi geciyor mu?

Olcut: ADAY KAHINI -- gate'ten ONCE, uretilen aday havuzunun GT'yi ne kadar
kapsadigi. Bire-bir Macar, TESPIT toleransi, aci serbest. (robot_cp.adaylari_uret
notundaki olcutle ayni: "aday kahini one-to-one Macar".)

NEDEN KAHIN, NEDEN F1 DEGIL: g10 yalnizca SEGMENTASYONU degistirdi. Gate ve p3c
eski dagilimda egitildi; uctan uca F1 dusukse bu g10'un kotu oldugunu DEGIL,
gate'in yeniden fit edilmedigini gosterir. Kahin, sonraki katmanlardan bagimsiz
olarak temsilin iyilesip iyilesmedigini olcer.

KILL: g10 kahini g7'yi GECMEZSE sig boyama GERI ALINIR.
"""
import os
import sys
import json
import glob

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")

import d6_kayit                      # noqa: E402
import robot_cp                      # noqa: E402
from sina_kume import esle_macar, f1w  # noqa: E402
from korpus_kimlik import step_kimlik as SK  # noqa: E402

KOLLAR = {"g10": "results/_p1_olasilik_g10", "b9": "results/_p1_olasilik_b9"}
CIKTI = "results/b9_kill_kapisi.json"


def main():
    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}

    # SADECE her iki onbellekte de bulunan parcalar -- kiyas ayni kumede olmali
    ortak = None
    for ad, yol in KOLLAR.items():
        p = {f[:-4] for f in os.listdir(yol) if f.endswith(".npz")}
        ortak = p if ortak is None else (ortak & p)
    ortak = sorted(ortak & set(kayit))
    print(f"ortak parca: {len(ortak)}")

    sonuc = {}
    for ad, yol in KOLLAR.items():
        T, rejim = [], {"dusuk": [], "cok": []}
        hata = 0  # artik yalniz raporlanir; hicbir istisna yutulmaz
        for pid in ortak:
            r = kayit[pid]
            G = np.asarray(r["G"], float)
            Gd = np.asarray(r["Gd"], float)
            if not len(G):
                continue
            # CIPLAK `except Exception` YOK. Ilk denemede vardi ve bir unpack
            # hatasini (adaylari_uret 4 deger dondurur, 3 degil -- docstring bayat)
            # 468/468 "hata" olarak yutup ekrana SAHTE bir "KILL" karari bastirdi.
            # Bir olcum betigi, olcemedigi zaman KARAR URETMEMELI: patlamali.
            d = np.load(f"{yol}/{pid}.npz")
            V = np.ascontiguousarray(d["V"], np.float64)
            F = np.ascontiguousarray(d["F"], np.int64)
            pbs = [np.asarray(q, float) for q in d["pbs"]]
            cps, _op, _cok, _per = robot_cp.adaylari_uret(V, F, pbs, S.get(pid))
            P = np.asarray([c["point"] for c in cps], float) if cps \
                else np.zeros((0, 3))
            D = np.asarray([c.get("dir", [0, 0, 1]) for c in cps], float) if cps \
                else np.zeros((0, 3))
            oge = (len(G),) + esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)[:3]
            T.append(oge)
            (rejim["cok"] if len(G) >= 8 else rejim["dusuk"]).append(oge)
        sonuc[ad] = {
            "kahin_F1": f1w(T),
            "dusuk_CP": f1w(rejim["dusuk"]) if rejim["dusuk"] else None,
            "cok_CP": f1w(rejim["cok"]) if rejim["cok"] else None,
            "n_parca": len(T), "hata": hata,
        }
        s = sonuc[ad]
        _f = lambda v: "  --  " if v is None else f"{v:.4f}"  # noqa: E731
        print(f"{ad:>4}: kahin {_f(s['kahin_F1'])} | dusuk-CP "
              f"{_f(s['dusuk_CP'])} | cok-CP {_f(s['cok_CP'])} "
              f"({s['n_parca']} parca, {hata} hata)", flush=True)

    fark = sonuc["b9"]["kahin_F1"] - sonuc["g10"]["kahin_F1"]
    gecti = fark > 0
    print(f"\nFARK (g10 - g7): {fark:+.4f}")
    print(f"KARAR: {'GECTI -- augmentation KALIR' if gecti else 'KALDI -- augmentation GERI ALINIR'}")
    with open(CIKTI, "w") as f:
        json.dump({"sonuc": sonuc, "fark": fark, "gecti": bool(gecti),
                   "olcut": "aday kahini, bire-bir Macar, tespit toleransi, aci serbest",
                   "not": "gate/p3c YENIDEN FIT EDILMEDI; uctan uca F1 bu kapiyla "
                          "olculmez"}, f, indent=1)
    print(f"makbuz -> {CIKTI}")


if __name__ == "__main__":
    main()
