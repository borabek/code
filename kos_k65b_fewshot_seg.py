"""K6.5-b: FEW-SHOT SEGMENTASYON FINE-TUNE -- listenin en yuksek beklenen degerli kolu.

SENARYO (kullanicinin gercek is akisi): gorulmemis bir markadan k parca gelir, INSAN
onlarin CP'lerini isaretler, sistem adapte olur, AYNI markanin kalan parcalarinda
olculur. k=0 = bugunku sifir-atis (robot 0.2344).

NEDEN SEG, NEDEN GATE DEGIL: gate ayagi olculdu ve TAMAMEN NULL cikti (3 markada
+-0.005). Sebep gorulmemis markada FN'lerin %76.2'sinin ADAY_YOK olmasi -- gate'e ne
ogretirsen ogret URETILMEMIS adayi geciremez. Adaptasyon TEMSIL katmaninda olmali.

ZINCIR:
  1. k parcayi GT CP'lerinden boya      g5_agiz_etiket.py --pids-file
  2. o boyamayla fine-tune              train_seg_extra.py --init-from
  3. olasilik onbellegi                 p1_olasilik_onbellek.py --ckpt <ft> --ek
  4. aday turet + gate + TAM zincir     bu betik

IKI TUZAK (ikisi de bilerek ele alindi):
  * OZ-TUTARLILIK KAPISI KAPATILIR (--oz-tut-esik 0). Kapi, urunun zaten beceremedigi
    parcalari eler; few-shot'ta bu DONGUSELDIR -- tam da ogrenmek istedigimiz zor
    parcalari atar. Gercek senaryoda etiketi INSAN koyar, oz-tutarlilik aranmaz.
  * ADAPTASYON PARCASI OLCUME GIRMEZ. Girerse sayilar ornekleme-ici olur.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

PY = "./.venv/Scripts/python.exe"
URUN_CKPT = "results/seg_g10/g10_s0.pt"
CIKTI = "results/k65b_fewshot_seg.json"


def kos(cmd, log, ek_env=None):
    """Alt sureci kos; BASARISIZ OLURSA PATLA (sessiz devam = sahte sonuc)."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    if ek_env:
        env.update(ek_env)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"ADIM BASARISIZ (kod {r.returncode}): {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marka", default="SUPU")
    ap.add_argument("--klar", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--cekilis", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="fine-tune icin DUSUK lr; 1e-3 ile ag onceki bilgisini unutur")
    ap.add_argument("--kip", choices=["yalniz", "tekrar"], default="tekrar",
                    help="yalniz = SADECE k parcayla adapte (agresif, unutma riski); "
                         "tekrar = korpus korunur, k parca N kez tekrarlanir (gerceksi)")
    ap.add_argument("--tekrar", type=int, default=30,
                    help="--kip tekrar icin: k parca kac kez tekrarlanacak")
    a = ap.parse_args()

    os.environ.setdefault("BA_ALLOW_SEEN", "1")
    sys.path.insert(0, ".")
    import d6_kayit

    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    pidler = sorted(p for p, r in kayit.items() if r["mfg"] == a.marka)
    if len(pidler) < max(a.klar) + 10:
        raise SystemExit(f"{a.marka}: yalniz {len(pidler)} parca, yetersiz")
    kume_yolu = f"results/_fs_kume_{a.marka}.json"
    if not os.path.exists(kume_yolu):
        raise SystemExit(f"{kume_yolu} yok -- once marka alt kumesini uret")
    with open(kume_yolu) as _f:
        n_beklenen = len(json.load(_f)["pidler"])
    print(f"=== {a.marka}: {len(pidler)} parca | onbellek beklentisi "
          f"{n_beklenen} ===", flush=True)

    rng = np.random.RandomState(0)
    sonuc = {}
    for k in a.klar:
        sonuc[k] = []
        for c in range(a.cekilis):
            etiket = f"{a.marka}_k{k}_c{c}"
            adapt = sorted(rng.choice(pidler, k, replace=False))
            olc = [p for p in pidler if p not in adapt]
            print(f"\n--- {etiket}: adapt {adapt} | olcum {len(olc)} parca ---",
                  flush=True)

            # DEVAM EDILEBILIRLIK -- SIKI KONTROL.
            # ILK SURUMUM GEVSEKTI (">=10 npz varsa atla") ve BAYAT bir onbellegi
            # gecerli saydi: o onbellek (a) `--kume` eklenmeden once uretilmisti,
            # yani 468 parcalik TUM sinav kumesini kapsiyordu, (b) "en iyi" ckpt'ten
            # geliyordu, "son"dan degil, (c) yarida kesilmisti (327/468).
            # Olcum onunla kosulsaydi SESSIZCE yanlis sayi verirdi.
            # Simdi: parca sayisi TAM eslesmeli VE onbellek ckpt'ten YENI olmali.
            _ob = f"results/_p1_olasilik_fs_{etiket}"
            _ck_son = f"results/seg_fs/{etiket}_last.pt"
            _tamam = False
            if os.path.exists(_ck_son) and os.path.isdir(_ob):
                _n = len([x for x in os.listdir(_ob) if x.endswith(".npz")])
                _yeni = os.path.getmtime(_ob) >= os.path.getmtime(_ck_son)
                _tamam = (_n == n_beklenen) and _yeni
                if not _tamam:
                    print(f"  YENIDEN KOSULACAK {etiket}: onbellek {_n}/"
                          f"{n_beklenen} parca, ckpt'ten yeni={_yeni}", flush=True)
            if _tamam:
                _nb = len([x for x in os.listdir(f"results/_fs_boya_{etiket}")
                           if x.endswith(".npz")])
                print(f"  ATLANDI (dogrulandi): {etiket}", flush=True)
                sonuc[k].append({"cekilis": c, "adapt": adapt, "n_olc": len(olc),
                                 "k_gercek": _nb, "k_istenen": len(adapt),
                                 "ckpt": f"results/seg_fs/{etiket}.pt",
                                 "ckpt_kullanilan": _ck_son, "onbellek": _ob})
                continue

            pf = f"results/_fs_{etiket}_pids.txt"
            with open(pf, "w") as f:
                f.write("\n".join(adapt))
            boya_dir = f"results/_fs_boya_{etiket}"
            ck = f"results/seg_fs/{etiket}.pt"
            os.makedirs("results/seg_fs", exist_ok=True)

            # 1) BOYA -- oz-tutarlilik kapisi KAPALI (dongusellik onlemi)
            # SINAV DISLAMASI KAPATILIR (ETIKET_DISLA=""). Guvenli, cunku:
            #  * yalniz --pids-file'daki k parca boyanir (baska hicbir sinav parcasi degil)
            #  * o k parca OLCUMDEN CIKARILIR (`olc` listesinde yok)
            #  * uretilen ckpt ATILIKTIR, urune girmez
            # Senaryo zaten "bu k parcayi sisteme VERIYORUZ" demek; dislama bunu bloke eder.
            kos([PY, "-u", "g5_agiz_etiket.py", "--pids-file", pf,
                 "--oz-tut-esik", "0.0", "--cikti", boya_dir],
                f"results/_fs_{etiket}_boya.log", ek_env={"ETIKET_DISLA": ""})
            # 1b) NPZ -> OBJ+labels.txt. train_seg_extra.load_extra YALNIZ dizin
            # bicimini okur; bu adim atlanirsa "0 parca yuklendi" olur ve
            # --yalniz-kismi olmasa egitim SESSIZCE korpusla kosardi.
            obj_dir = f"results/_fs_obj_{etiket}"
            kos([PY, "-u", "g5b_etiket_donustur.py", "--kaynak", boya_dir,
                 "--hedef", obj_dir], f"results/_fs_{etiket}_donus.log")
            n_boya = len([x for x in os.listdir(boya_dir) if x.endswith(".npz")])
            # BOYANAN > ISTENEN olursa SESSIZ SISME demektir -> DUR.
            # BOYANAN < ISTENEN ise: oto-boyayici INSAN ETIKETININ VEKILI ve bazi
            # parcalarda vekil calismiyor (GT'nin hicbiri algilanan bir acikliga
            # dusmuyor -> `oz_tutarlilik_dustu`). Insan o parcayi etiketleyebilirdi.
            # Bu yuzden kosumu DURDURMUYORUZ, GERCEK k'yi KAYDEDIYORUZ; olculen
            # egri boylece gercek insan etiketine gore bir ALT SINIR olur.
            if n_boya > len(adapt):
                raise RuntimeError(
                    f"{etiket}: {len(adapt)} istendi, {n_boya} boyandi -- SISME, durduruldu.")
            if n_boya == 0:
                raise RuntimeError(f"{etiket}: hicbir parca boyanamadi")
            if n_boya < len(adapt):
                print(f"  UYARI: {len(adapt)} istendi, {n_boya} boyandi "
                      f"(oto-boyayici vekil; gercek k={n_boya})", flush=True)

            # 2) FINE-TUNE
            egit = [PY, "-u", "train_seg_extra.py", "--init-from", URUN_CKPT,
                    "--partial-dir", obj_dir, "--epochs", str(a.epochs),
                    "--lr", str(a.lr), "--val-partial", "--checkpoint-out", ck]
            egit += ["--yalniz-kismi"] if a.kip == "yalniz" else                     ["--kismi-tekrar", str(a.tekrar)]
            kos(egit, f"results/_fs_{etiket}_train.log")

            # 3) OLASILIK ONBELLEGI
            # ONBELLEK YALNIZ BU MARKANIN PARCALARI ICIN: tam sinav kumesi 468
            # parca (~25 dk); olcum zaten yalniz bu markada yapiliyor (~4 dk).
            kume = f"results/_fs_kume_{a.marka}.json"
            if not os.path.exists(kume):
                raise SystemExit(f"{kume} yok -- once marka alt kumesini uret")
            # SON EPOCH KULLANILIR, "en iyi" DEGIL. Egitim betigi en iyiyi GENEL
            # dogrulamaya gore seciyor; adaptasyon genel dogrulamayi DUSURDUGU icin
            # o secim EN AZ ADAPTE OLMUS modeli kaydeder -- yani few-shot'i olcmek
            # isterken few-shot'i engeller. Few-shot'ta secim yok: sabit epoch butcesi,
            # son ckpt. (Hedef markadan ayri bir secim kumesi ayirmak olcum kumesini
            # kucultecegi icin simdilik yapilmiyor; kayit altina alindi.)
            ck_son = ck.replace(".pt", "_last.pt")
            if not os.path.exists(ck_son):
                raise RuntimeError(f"{ck_son} yok -- egitim son ckpt yazmadi")
            kos([PY, "-u", "p1_olasilik_onbellek.py", "--ckpt", ck_son,
                 "--kume", kume, "--ek", f"_fs_{etiket}"],
                f"results/_fs_{etiket}_p1.log")

            sonuc[k].append({"cekilis": c, "adapt": adapt, "n_olc": len(olc),
                             "k_gercek": n_boya, "k_istenen": len(adapt),
                             "ckpt": ck, "ckpt_kullanilan": ck.replace(".pt", "_last.pt"), "onbellek": f"results/_p1_olasilik_fs_{etiket}"})
            print(f"  HAZIR -> {ck}", flush=True)

    with open(CIKTI, "w") as f:
        json.dump({"marka": a.marka, "klar": a.klar, "cekilis": a.cekilis,
                   "epochs": a.epochs, "lr": a.lr, "taban_ckpt": URUN_CKPT,
                   "kosumlar": sonuc,
                   "uyari": "k_gercek < k_istenen olabilir: oto-boyayici INSAN "
                            "etiketinin VEKILIDIR ve bazi parcalarda GT'nin hicbiri "
                            "algilanan bir acikliga dusmez. Olculen egri, gercek "
                            "insan etiketine gore bir ALT SINIRDIR.",
                   "not": "Bu betik ADAPTASYON+ONBELLEK uretir. UCTAN UCA OLCUM "
                          "ayri adimdir (sonda_k65b_olc.py) -- boylece egitim bir kez "
                          "kosar, olcum tekrar tekrar kosulabilir."}, f, indent=1)
    print(f"\nmakbuz -> {CIKTI}")
    print("SIRADAKI: sonda_k65b_olc.py (uctan uca olcum, k=0 tabaniyla birlikte)")


if __name__ == "__main__":
    main()
