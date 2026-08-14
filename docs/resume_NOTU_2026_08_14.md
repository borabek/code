# DEVAM NOTU — 2026-08-14 morning (makine kapaniyor)

## DURUM: SUNUM PAKETI HAZIR, RISK YOK
`docs/SUNUM_PAKETI.md` sunuma gidebilir. Cekirdek sayilar (VAL 100 part):
**detection 0.7878** [0.7464-0.8263] · robot-axis 0.5764 ·
**robot-ISARETLI 0.4839** [0.4003-0.5671] · **alan farki +0.2468**.
Asagidaki yarim remaining olcumlerin hicbiri this sayilari degistirmez;
yalnizca "a arm more" ekleyebilir.

## DENETIM
* `smoke_test.py` **GECTI** · `rollback.py --kontrol` -> **1 bilincli
  deviation** (`cp_config.json`)
* **D7 OKUNMADI** (2 okuma hakki) · **LOCKED harcanmadi**
* Degisen urun dosyalari (`robot_cp.py`, `wire_gate.py`,
  `diffusionnet.py`) git'te izleniyor; **hepsi cevre degiskeniyle kapili,
  default davranis BIT-AYNI**

---

## YARIM KALAN UC OLCUM — aynen tekrar baslatilabilir

Hepsi **training gerektirmez**, yalnizca inference. Her biri ~25-50 dk.

### 1) ~~CRF dorduncu setting (10 tur / 0.9)~~ — **IPTAL EDILDI**
Kol monotoniklik gerekcesiyle CLOSED (asagidaki tabloya bak); dorduncu
setting hukmu degistirmez. Kosu suresi ensemble kollarina (2 ve 3) ayrildi.
Komut, arm ileride yeniden acilirsa kullanilmak about to duruyor:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 EZ_CRF="10,0.9" \
EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
EZ_DOKUM=results/_dokum_crf_t10w09.json \
.venv/Scripts/python.exe probe_chain_paired_compare.py > logs/val_crf_t10w09.log 2>&1
# after:
.venv/Scripts/python.exe probe_dump_compare.py \
  results/_dump_baseline.json results/_dokum_crf_t10w09.json measured_path
```

### 2) TOPLULUK GENISLETME A (5 uye: urun 4 + same recetenin 4. seed)
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
EZ_CKPT="results/seg_extra/recall_hard_s2.pt,results/seg_extra/recall_hard_keig96_s0.pt,results/seg_extra/recall_hard_keig96_s1.pt,results/seg_extra/recall_hard_keig96_s2.pt,results/seg_extra/recall_hard_keig96_s3.pt" \
EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
EZ_DOKUM=results/_dump_top5.json \
.venv/Scripts/python.exe probe_chain_paired_compare.py > logs/val_top5.log 2>&1
# after:
.venv/Scripts/python.exe probe_dump_compare.py \
  results/_dump_baseline.json results/_dump_top5.json measured_path
```

### 3) TOPLULUK GENISLETME B (8 uye: + augmentasyon tohumlari)
`y1b_aug0.15_s0/s1/s2` checkpointleri TEK TEK more guclu
(val Conn_IoU 0.6528 / 0.6990 / 0.6673).
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
EZ_CKPT="results/seg_extra/recall_hard_s2.pt,results/seg_extra/recall_hard_keig96_s0.pt,results/seg_extra/recall_hard_keig96_s1.pt,results/seg_extra/recall_hard_keig96_s2.pt,results/seg_extra/recall_hard_keig96_s3.pt,results/seg_extra/y1b_aug0.15_s0.pt,results/seg_extra/y1b_aug0.15_s1.pt,results/seg_extra/y1b_aug0.15_s2.pt" \
EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
EZ_DOKUM=results/_dump_top8.json \
.venv/Scripts/python.exe probe_chain_paired_compare.py > logs/val_top8.log 2>&1
```

### 4) ~~KAZANAN CRF AYARI SAHA YOLUNDA~~ — GEREKSIZ (arm closed)
Asagidaki komut, ileride arm yeniden acilirsa kullanilmak about to duruyor.

Sonda kancalari only `measured_path` yoluna ulasir; dagitim karari for
urun yolunda olculmeli. `robot_cp.extract` icine `CP_CRF` kancasi
KURULDU ve dogrulandi.
```bash
CP_CRF="4,0.5" EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
EZ_DOKUM=results/_dokum_crf_saha.json \
.venv/Scripts/python.exe probe_chain_paired_compare.py > logs/val_crf_saha.log 2>&1
# after SAHA yolunda kiyasla:
.venv/Scripts/python.exe probe_dump_compare.py \
  results/_dump_baseline.json results/_dokum_crf_saha.json field
```

---

## Y24 CRF — SIMDIYE KADARKI TABLO (measured_path path, VAL 100, paired bootstrap)

| setting | robot-ISR | diff | %95 GA | poz% |
|---|---|---|---|---|
| baseline | 0.4668 | — | — | — |
| 2 tur / 0.3 | 0.4670 | +0.0004 | [−0.0199,+0.0218] | 50.0 |
| **4 tur / 0.5** | **0.4766** | **+0.0100** | [−0.0128,+0.0335] | **80.5** |
| 6 tur / 0.7 | 0.4645 | −0.0020 | [−0.0249,+0.0222] | 43.0 |

**KOL CLOSED (2026-08-14 06:29).** Egri +0.0004 -> +0.0100 -> −0.0020,
i.e. **tepe yapip donuyor: MONOTONIK DEGIL**. Bu, kampanyada halka-sign
threshold taramasinda da gorulen "gurultuye uydurma" imzasidir. Tepe degeri
(+0.0100) already GA'si sifiri iceren a number. Tespit each ayarda negatif
(−0.005…−0.009).

`CP_CRF` kancasi kodda KALIYOR (default kapali, bit-same davranis):
segmentasyon kalitesi degisirse yeniden olculebilir.
Ayrinti: rapor Bolum 21.67.

---

## HATIRLATMA — this night ogrenilen dort measurement kurali
1. Segmentasyon **seed gurultusu 0.046**; altindaki hicbir seg arm tek
   kosuyla verdict giymez.
2. Sonda kancalari only **`measured_path`** yoluna ulasir; `field`da +0.0000
   gormek kolun olu oldugu anlamina GELMEZ.
3. Ara degerler **uretildikleri yerden** yakalanmali (cache).
4. **Cevrimdisi vekil karar verdirmez** -- a arm vekilde +0.9 puan
   kazanip uctan uca −0.0437 verdi.
