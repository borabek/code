# OTOPSI — GENEL ML KONTROL LISTESI, BU PROJEDE

Tarih: 2026-08-13 · Olcut: bu projede **olculmus** evidence.
Hukum uc turlu: **OLCULDU-DUSTU** · **ZATEN VAR** · **CANLI (denenmedi)**

---

## 1. OLCULDU ve DUSTU — yeniden acmak icin YENI rationale sart

| madde | bu projede olculen | receipt |
|---|---|---|
| Focal loss | **−0.3018** | `kayip_denemesi` |
| Sinif agirliklandirma | notr (−0.0058) | ayni |
| Parca-esitleyici agirlik | **−0.1120** | `gelistirme_taramasi` |
| Hard example mining (zor negatif) | **−0.0195** | ayni |
| HPO (lr / yaprak / L2) | notr ya da zararli | ayni |
| Ensemble / kalibrasyon / pseudo-label | hepsi DUSTU | kullanicinin onceki listesi |
| Sentetik veri (kural tabanli) | uretec calisiyor, **segmentasyon sentetikte ateslemiyor (0.92x)** | `sentetik_duman` |
| Veri kaynagi genisletme | stok BITTI: WSCAD olu, SIE'de yeni yalniz 22 part | `wscad-data-lever-dead` |
| Noise injection / B9 augmentation | KAPANDI | `g10-sig-boyama-gecti` |
| Threshold optimizasyonu | ZATEN yapiliyor (fold icinde kural aramasi) | `run_p6_kademe2` |
| Feature engineering | 25+ blok measured, biri bile uretimde kalmadi | Bolum 19 |
| Multi-modal fusion | metadata YOK -- yalnizca geometri | — |

## 2. ZATEN VAR — tekrar kurmak israf

| madde | projedeki karsiligi |
|---|---|
| Failure autopsy | error bankasi + FN taksonomisi + `OTOPSI_YOGUN_PARCA.md` |
| K-fold CV | brand-disi katlar (her olcumde) |
| Guven araligi | grup bootstrap (`geometry-twin-leakage`) |
| Deduplication | %80 geometrik ikiz tespit edildi, grup bootstrap SART |
| Checkpoint ensembling | canli urun ZATEN 4 kontrol noktasi topluluğu |
| Stacking / meta-learner | P6_KAFES = iki kademeli (1. kademe skoru 2. kademeye girdi) |
| Robust scaling | part-ici z-skor DAGITILDI |
| Metric engineering | MIKRO vs f1w karari verildi ve yazildi |
| Reject / abstain | TIER (AUTO/REVIEW) var -- ama **ATIL** olctugu ayri konu |
| Seed / repro | `makbuz_hash.damga()` her makbuzda |
| Robust eval set | d6 (gelistirme) / d7 (SINAV, 2 okuma kaldi) |
| Transfer / fine-tune | seg A/B kolu tam olarak buydu |
| Uncertainty | vote sayisi = topluluğun kendi confidence sinyali |

## 3. CANLI — denenmemis, bu projede anlamli

Sirasi beklenen etkiye gore:

1. **SEG augmentasyonu -- YALNIZ HAFIF OLANI.** *(DUZELTME 2026-08-13:
   ilk yazimda "hic denenmedi" demistim, YANLIS.)* `--augment` bayragi VAR
   ve kendi yardim metni soyle diyor: *"1.05 = aggressive (the
   heuristic-tuned one that hurt quality)"* -- yani AGRESIF donme (1.05 rad
   ~ 60 derece) denendi ve KALITEYI DUSURDU. Denenmemis olan HAFIF ayar
   (0.2-0.4 rad). Egitim logu hala `augment=False` diyor, yani su an hic
   augmentasyon YOK. Kol: `--augment --augment-maxang 0.3`.
2. **Yogun-part UZMANI (specialist)** — NIT tipi parts GT'nin %51'i ve
   uctan uca 0.0089. Ayri egitilmis bir uzman + router hic denenmedi.
   Rejim kapisi denendi, UZMAN MODEL denenmedi.
3. **EMA / SWA agirlik ortalamasi** — tek kosudan bedava stabilizasyon.
   Denenmedi.
4. **Snapshot ensembling (tek kosudan)** — cosine restart + epoch
   toplulugu. Mevcut ensemble AYRI kosulardan; bu ayri bir sey.
5. **Label smoothing / yumusak etiket** — denenmedi, ucuz.
6. **Yardimci gorev (multi-task)** — `train_seg_extra.py --aux-wire`
   bayragi VAR ama bu kampanyada hic acilmadi.
7. **Curriculum (kolay→zor)** — denenmedi.
8. **Etiket kalitesi agirliklandirma** — kismi etiketlerin guvenilirligi
   farkli; agirlik verilmiyor.
9. **Oznitelik secimi / boyut indirgeme** — 100+ sutun var, sistematik
   eleme yapilmadi.
10. **Self-supervised / contrastive on-training** — pahali, ama sentetik
    dustukten sonra "temsil" yolunun kalan tek adayi.
11. **Mixup / CutMix (vertex duzeyi)** — mesh'e uyarlanmasi gerek.
12. **Optimizator denemesi (AdamW/SGD/Ranger)** — yalniz DiffusionNet
    tarafinda anlamli (HGB'de optimizator yok).

## 4. GECERSIZ — bu projeye uymuyor

| madde | neden |
|---|---|
| Canary / A-B uretimde | urun sahaya HENUZ inmiyor |
| Drift monitoring | ayni |
| Cost-matrix | robot metrigi ZATEN is metrigi |
| Explainability (SHAP) | teshis araci, F1 kaldiraci degil |
| Distillation | hedef F1, model boyutu degil |
| Annotation guidelines | insan etiketleme durdu |
| MLflow / W&B | receipt + damga sistemi ayni isi goruyor |

---

## HUKUM

Listenin **%60'i bu projede zaten measured ve dustu ya da zaten var**.
Gercekten yeni olan **12 madde** var ve bunlarin en umutlu ikisi
sunlar, cunku ikisi de olculmus duvara dogrudan nisan aliyor:

- **segmentasyon egitiminde augmentasyon** (log: `augment=False`)
- **dense-part uzmani** (NIT = GT'nin %51'i, uctan uca 0.0089)

---

# EK OTOPSI — IKINCI LISTE (2026-08-13)

Kod dogrulamasiyla (`train_seg_extra.py` satir 179/203/216/317).

## OLCULDU-DUSTU ya da ZATEN VAR

| madde | verdict | evidence |
|---|---|---|
| Probabilistic calibration (Platt/isotonic) | **DUSTU** | onceki listede measured |
| Tversky loss | **ZATEN ACIK** | `--tversky` varsayilan 0.25, `alpha 0.3 / beta 0.7` (satir 179, 317) |
| Per-part canonicalization | **OLCULDU**: olcumde +0.0151, URETIMDE −0.0138 | Bolum 19.11 |
| Spectral descriptors / Laplacian ozfonksiyon kodlamasi | **ZATEN VAR** | DiffusionNet'in KENDISI spektral (`n_eig=96`) |
| Per-class threshold sweep | **ZATEN VAR** (fold icinde kural aramasi) | `run_p6_kademe2` |
| Per-part loss weighting (esitleyici) | **DUSTU** −0.1120 | `gelistirme_taramasi` |
| Point-sampling stratejileri | buyuk olcude KAPSANMIS | remesh ZATEN uniform izotropik |
| Segmentasyon augmentasyonu (AGRESIF) | **DUSTU** | bayragin kendi yardim metni: "1.05 = the heuristic-tuned one that hurt quality" |

## CANLI — listeye eklenen YENI maddeler

| kod | madde | neden umutlu |
|---|---|---|
| Y13 | **Sinir-farkindali loss** (edge/contour IoU) | CP fiziksel olarak bir SINIR (delik mouth cemberi); loss bunu hic hedeflemiyor |
| Y14 | **Focal-Tversky** | Tversky ACIK ama focal varyanti denenmedi |
| Y15 | Tepe duzeyi threshold taramasi (segmentasyon) | `BH_MESH_ESIK=0.05` elle secilmis, taranmadi |
| Y16 | **Remesh-varyant toplulugu** | tek remesh var; varyant gurultusu hic stabilize edilmedi |
| Y17 | Laplacian / spektral koordinat augmentasyonu | topoloji-korumali, geometriye uygun |
| Y18 | Jeodezik-farkindali jitter | gercekci yerel augment |
| Y19 | **Egrilik / normal kanallari girdi olarak** | `input_features="xyz"` -- yalnizca konum! egrilik ve normal HIC verilmiyor |
| Y20 | **Cok-gorunumlu render fuzyonu** (normal/depth render + CNN) | TAMAMEN YENI kiplik; mevcut hicbir arm goruntu sinyali kullanmiyor |
| Y21 | Gorunum + mesh gec fuzyon toplulugu | Y20'nin dogal devami |
| Y22 | Tepe duzeyi belirsizlik (MC dropout) | `dropout=0.3` zaten var, cikarimda kapali |
| Y23 | **NIT'i YUKARI agirliklandirma** | esitleyici agirlik dustu (−0.1120) ama TERSI (zor parcaya AGIRLIK) denenmedi |
| Y24 | Sinir iyilestirme son-islem (graph-cut / CRF) | mesh uzerinde hic post-process yok |
| Y25 | Self-supervised on-gorev (normal tahmini / jigsaw) | sentetik dustukten sonra temsil yolunun adayi |
| Y26 | Sinifa-ozel augmentasyon receteleri | hedefli augment |
| Y27 | Hiyerarsik part duzeyi cozucu | Y2 uzman koluyla sinerjik |

**En umutlu ucu, gerekcesiyle:**
1. **Y19** — model girdisi `input_features="xyz"`, yani modele yalnizca KONUM
   veriliyor; egrilik ve normal gibi yuzey bilgisi HIC verilmiyor. Kablo
   girisi bir yuzey ozelligidir.
2. **Y13** — CP bir SINIRDIR (mouth cemberi) ve loss bunu hedeflemiyor.
3. **Y20** — mevcut butun kollar tek kiplikte (mesh); render fuzyonu yeni
   bilgi getirir.
