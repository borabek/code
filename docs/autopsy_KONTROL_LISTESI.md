# OTOPSI — GENEL ML KONTROL LISTESI, BU PROJEDE

Tarih: 2026-08-13 · Olcut: this projede **olculmus** evidence.
Hukum uc turlu: **MEASURED-DUSTU** · **ZATEN VAR** · **CANLI (denenmedi)**

---

## 1. MEASURED ve DUSTU — yeniden acmak for YENI rationale sart

| madde | this projede measured_path | receipt |
|---|---|---|
| Focal loss | **−0.3018** | `kayip_denemesi` |
| Sinif agirliklandirma | notr (−0.0058) | same |
| Parca-esitleyici agirlik | **−0.1120** | `gelistirme_taramasi` |
| Hard example mining (zor negatif) | **−0.0195** | same |
| HPO (lr / yaprak / L2) | notr ya da zararli | same |
| Ensemble / calibration / pseudo-label | hepsi DUSTU | kullanicinin onceki listesi |
| Sentetik data (rule tabanli) | uretec calisiyor, **segmentasyon sentetikte ateslemiyor (0.92x)** | `synthetic_smoke` |
| Veri kaynagi genisletme | stok BITTI: WSCAD olu, SIE'de yeni only 22 part | `wscad-data-lever-dead` |
| Noise injection / B9 augmentation | CLOSED | `g10-sig-boyama-gecti` |
| Threshold optimizasyonu | ZATEN yapiliyor (fold icinde rule aramasi) | `run_p6_kademe2` |
| Feature engineering | 25+ blok measured, biri bile uretimde kalmadi | Bolum 19 |
| Multi-modal fusion | metadata YOK -- yalnizca geometri | — |

## 2. ZATEN VAR — tekrar kurmak israf

| madde | projedeki karsiligi |
|---|---|
| Failure autopsy | error bankasi + FN taksonomisi + `autopsy_dense_part.md` |
| K-fold CV | brand-disi katlar (each olcumde) |
| Guven araligi | grup bootstrap (`geometry-twin-leakage`) |
| Deduplication | %80 geometrik ikiz detection edildi, grup bootstrap SART |
| Checkpoint ensembling | canli urun ZATEN 4 kontrol noktasi topluluğu |
| Stacking / meta-learner | P6_KAFES = iki kademeli (1. kademe skoru 2. kademeye girdi) |
| Robust scaling | part-ici z-score DEPLOYED |
| Metric engineering | MIKRO vs f1w karari verildi ve yazildi |
| Reject / abstain | TIER (AUTO/REVIEW) present -- but **ATIL** olctugu ayri konu |
| Seed / repro | `receipt_hash.damga()` each makbuzda |
| Robust eval set | d6 (gelistirme) / d7 (SINAV, 2 okuma kaldi) |
| Transfer / fine-tune | seg A/B arm tam as buydu |
| Uncertainty | vote sayisi = topluluğun kendi confidence sinyali |

## 3. CANLI — denenmemis, this projede anlamli

Sirasi beklenen etkiye per:

1. **SEG augmentasyonu -- YALNIZ HAFIF OLANI.** *(DUZELTME 2026-08-13:
   ilk yazimda "no denenmedi" demistim, YANLIS.)* `--augment` bayragi VAR
   ve kendi yardim metni soyle diyor: *"1.05 = aggressive (the
   heuristic-tuned one that hurt quality)"* -- i.e. AGRESIF donme (1.05 rad
   ~ 60 derece) denendi ve KALITEYI DUSURDU. Denenmemis which HAFIF setting
   (0.2-0.4 rad). Egitim logu hala `augment=False` diyor, i.e. su an no
   augmentasyon YOK. Kol: `--augment --augment-maxang 0.3`.
2. **Yogun-part UZMANI (specialist)** — NIT tipi parts GT'nin %51'i ve
   uctan uca 0.0089. Ayri egitilmis a uzman + router no denenmedi.
   Rejim kapisi denendi, UZMAN MODEL denenmedi.
3. **EMA / SWA agirlik ortalamasi** — tek kosudan bedava stabilizasyon.
   Denenmedi.
4. **Snapshot ensembling (tek kosudan)** — cosine restart + epoch
   toplulugu. Mevcut ensemble AYRI kosulardan; this ayri a sey.
5. **Label smoothing / yumusak etiket** — denenmedi, ucuz.
6. **Yardimci gorev (multi-task)** — `train_seg_extra.py --aux-wire`
   bayragi VAR but this kampanyada no acilmadi.
7. **Curriculum (kolay→zor)** — denenmedi.
8. **Etiket kalitesi agirliklandirma** — kismi etiketlerin guvenilirligi
   different; agirlik verilmiyor.
9. **Oznitelik secimi / boyut indirgeme** — 100+ sutun present, sistematik
   eleme yapilmadi.
10. **Self-supervised / contrastive on-training** — pahali, but sentetik
    dustukten after "temsil" yolunun remaining tek adayi.
11. **Mixup / CutMix (vertex duzeyi)** — mesh'e uyarlanmasi gerek.
12. **Optimizator denemesi (AdamW/SGD/Ranger)** — only DiffusionNet
    tarafinda anlamli (HGB'de optimizator none).

## 4. INVALID — this projeye uymuyor

| madde | why |
|---|---|
| Canary / A-B uretimde | urun sahaya HENUZ inmiyor |
| Drift monitoring | same |
| Cost-matrix | robot metrigi ZATEN is metrigi |
| Explainability (SHAP) | teshis araci, F1 kaldiraci not |
| Distillation | hedef F1, model boyutu not |
| Annotation guidelines | insan etiketleme durdu |
| MLflow / W&B | receipt + damga sistemi same isi goruyor |

---

## VERDICT

Listenin **%60'i this projede already measured ve dustu ya da already present**.
Gercekten yeni which **12 madde** present ve bunlarin en umutlu ikisi
sunlar, because ikisi de olculmus duvara dogrudan nisan aliyor:

- **segmentasyon egitiminde augmentasyon** (log: `augment=False`)
- **dense-part uzmani** (NIT = GT'nin %51'i, uctan uca 0.0089)

---

# EK OTOPSI — IKINCI LISTE (2026-08-13)

Kod dogrulamasiyla (`train_seg_extra.py` satir 179/203/216/317).

## MEASURED-DUSTU ya da ZATEN VAR

| madde | verdict | evidence |
|---|---|---|
| Probabilistic calibration (Platt/isotonic) | **DUSTU** | onceki listede measured |
| Tversky loss | **ZATEN OPEN** | `--tversky` default 0.25, `alpha 0.3 / beta 0.7` (satir 179, 317) |
| Per-part canonicalization | **MEASURED**: olcumde +0.0151, URETIMDE −0.0138 | Bolum 19.11 |
| Spectral descriptors / Laplacian ozfonksiyon kodlamasi | **ZATEN VAR** | DiffusionNet'in KENDISI spektral (`n_eig=96`) |
| Per-class threshold sweep | **ZATEN VAR** (fold icinde rule aramasi) | `run_p6_kademe2` |
| Per-part loss weighting (esitleyici) | **DUSTU** −0.1120 | `gelistirme_taramasi` |
| Point-sampling stratejileri | large olcude KAPSANMIS | remesh ZATEN uniform izotropik |
| Segmentasyon augmentasyonu (AGRESIF) | **DUSTU** | bayragin kendi yardim metni: "1.05 = the heuristic-tuned one that hurt quality" |

## CANLI — listeye added YENI maddeler

| kod | madde | why umutlu |
|---|---|---|
| Y13 | **Sinir-farkindali loss** (edge/contour IoU) | CP fiziksel as a SINIR (delik mouth cemberi); loss bunu no hedeflemiyor |
| Y14 | **Focal-Tversky** | Tversky OPEN but focal varyanti denenmedi |
| Y15 | Tepe duzeyi threshold taramasi (segmentasyon) | `BH_MESH_ESIK=0.05` elle secilmis, taranmadi |
| Y16 | **Remesh-varyant toplulugu** | tek remesh present; varyant gurultusu no stabilize edilmedi |
| Y17 | Laplacian / spektral koordinat augmentasyonu | topoloji-korumali, geometriye eligible |
| Y18 | Jeodezik-farkindali jitter | gercekci yerel augment |
| Y19 | **Egrilik / normal kanallari girdi as** | `input_features="xyz"` -- yalnizca konum! egrilik ve normal HIC verilmiyor |
| Y20 | **Cok-gorunumlu render fuzyonu** (normal/depth render + CNN) | TAMAMEN YENI kiplik; mevcut hicbir arm goruntu sinyali kullanmiyor |
| Y21 | Gorunum + mesh gec fuzyon toplulugu | Y20'nin dogal devami |
| Y22 | Tepe duzeyi belirsizlik (MC dropout) | `dropout=0.3` already present, cikarimda kapali |
| Y23 | **NIT'i YUKARI agirliklandirma** | esitleyici agirlik dustu (−0.1120) but TERSI (zor parcaya AGIRLIK) denenmedi |
| Y24 | Sinir iyilestirme son-islem (graph-cut / CRF) | mesh uzerinde no post-process none |
| Y25 | Self-supervised on-gorev (normal tahmini / jigsaw) | sentetik dustukten after temsil yolunun adayi |
| Y26 | Sinifa-ozel augmentasyon receteleri | hedefli augment |
| Y27 | Hiyerarsik part duzeyi cozucu | Y2 uzman koluyla sinerjik |

**En umutlu ucu, gerekcesiyle:**
1. **Y19** — model girdisi `input_features="xyz"`, i.e. modele yalnizca KONUM
   veriliyor; egrilik ve normal gibi yuzey bilgisi HIC verilmiyor. Kablo
   girisi a yuzey ozelligidir.
2. **Y13** — CP a SINIRDIR (mouth cemberi) ve loss bunu hedeflemiyor.
3. **Y20** — mevcut butun kollar tek kiplikte (mesh); render fuzyonu yeni
   bilgi getirir.
