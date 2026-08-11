# Desktop JSON -> WSCAD CP: %90 TODO

Tarih: 2026-07-15

Durum: Bu dosya yeni ve otoritatif plandir. Eski TRAINING_VAL90_TODO.md,
WSCAD/CAD pseudo-label egitimini ve eski checkpointleri merkeze aldigi icin bu
hedefte uygulanmayacak.

## 1. Hedefi kilitle

Urun hedefi:

- Supervised CP label kaynagi: C:\Users\DE00024082\Desktop\JSON.
- Giris: daha once egitimde gorulmemis WSCAD STEP.
- Cikis: her gercek connection point icin 3B nokta ve insertion/approach yonu.
- Ana metrik: 5 mm esleme yaricapinda, full-coverage, pooled micro-F1 >= 0.90.
- Zorunlu alt limitler: precision >= 0.90 ve recall >= 0.90.
- Ek kalite: ortalama localization error <= 2.0 mm ve angle error <= 15 derece.
- Nihai kanit: yalniz bagimsiz insan/uretici GT'li, kilitli WSCAD STEP test seti.

Metrik notu:

- Repoda "micro_accuracy" Jaccard'dir: TP / (TP + FP + FN).
- F1=0.90, Jaccard olarak yalniz 0.818'e denk gelir.
- Eger is hedefi gercekten Jaccard >= 0.90 ise gereken F1 yaklasik 0.947'dir.
- CAD-pseudo-GT, train split, ayni katalog parcasi veya selective/abstention
  sonucu nihai %90 diye raporlanmayacak.

Benchmarkta ML-only yol kullanilacak. predict.py icin --auto ve --from-cad
yasak; bunlar CAD sonucunu modele karistirir.

## 2. FBI sorusturmasi: 2026-07-15 sifir noktasi

### 2.1 Kaynak veri

- [x] Gercek kaynak tek dosya degil:
  C:\Users\DE00024082\Desktop\JSON altinda 479 parca JSON'u.
- [x] Toplam boyut 1,099,514,790 byte; 7,279,977 vertex ve 10,534,228 triangle.
- [x] 479/479 parse oluyor; bozuk JSON, eksik ana sema, NaN/Inf ve bbox disi CP yok.
- [x] Sema: PartNr, Graphic3d, BoundingBox, ConnectionPoints.
- [x] 11,927 ham ConnectionPoints, coincident dedup sonrasi gercek egitim hedefi
  3,099 fiziksel CP konumu.
- [x] Dagilim cok dengesiz: ABB 386/479 parca ve 2,520/3,099 benzersiz CP;
  PXC yalniz 21 parca ve 265 benzersiz CP.
- [x] 182/479 parca 8,000 vertex ustunde ve subsample/patch yoluna giriyor.
- [x] Exact vertex-cloud denetiminde 73 duplicate grup ve 303 fazla kopya var.
  Coarse geometry split, kullanilan 470 parcayi yalniz 155 gruba indiriyor.
- [x] Mevcut splitte exact hash/geometry grubu train-val-test arasinda gecmiyor;
  fakat 23 yakin katalog ailesi farkli splitlere dagiliyor.
- [x] SEW.MC07B0015-5A3-4-00 icinde 20 sifir direction ve zit yonlu coincident
  etiket var. Lokasyon etiketi kullanilabilir, direction loss'u oldugu gibi
  kullanilamaz.
- [x] Tum source directionlarin yaklasik %97.9'u axis-aligned; fakat hedef PXC
  alt-kumesinde bu oran yalniz yaklasik %32. Genel istatistik hedef aileyi
  temsil etmiyor.
- [ ] [INSAN QA] Urun semantigini yazili kilitle: hedef 11,927 mantiksal
  electrical terminal mi, yoksa loader'in urettigi 3,099 fiziksel giris/delik
  konumu mu? Bu plan fiziksel CP konumunu varsayiyor.

### 2.2 Hedef kapsam hatasi

- [x] Desktop JSON'daki dokuz gercek PXC terminal block, _bridge_test.txt ile
  real_v1/scratch fitinden tamamen cikarilmis.
- [x] Geri kalan 12 PXC, pxc_out_of_scope.txt'e gore terminal block olmayan
  touch panel, PC, I/O, olcum ve guc kaynagi parcalari.
- [x] Mevcut run bu 12 out-of-scope PXC'yi train/val/testte tutuyor; dokuz gercek
  hedef PXC'yi ise disarida birakiyor.
- [x] Bu nedenle mevcut run "Desktop JSON kullaniyor", fakat gercek target
  terminal-block orneklerini ogrenmiyor.
- [x] Korpus terminal-block-only degil; 479 parcadan yalniz 232'sinin bbox
  diagonali 250 mm veya daha kucuk. Boyut tek basina scope etiketi sayilmayacak.

### 2.3 Split ve provenance

- [x] cp_real_v1 manifesti: train 334, val 80, test 56, excluded PXC bridge 9.
- [x] Train/val/test benzersiz fiziksel CP sayilari: 2,151 / 554 / 376;
  bridge GT 18.
- [x] cp_real_v1, JSON ile fine-tune edilmis olsa da
  checkpoints/cp_hp_pretrain_synth_best.ckpt ile basliyor. Bu nedenle saf
  Desktop-JSON provenance degil.
- [x] cp_real_scratch sentetik init kullanmayan saf Desktop-JSON baseline'idir.
- [x] Eski v31/v28/v24 sonuclari WSCAD CAD/pseudo-label verisi kullandigi icin
  yeni hedefle karsilastirilmayacak.
- [x] 2026-07-15 denetim aninda aktif train_cp/run_real Python prosesi yok;
  real_v1 epoch 85'te, scratch epoch 75'te early-stop ile bitmis.

### 2.4 Dürüst skorlar

| Kosu / benchmark | TP | FP | FN | Precision | Recall | micro-F1 | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| cp_real_scratch en iyi real-val, epoch 35 | 88 | 390 | 466 | 0.184 | 0.159 | 0.171 | 0.093 |
| cp_real_v1 en iyi real-val, epoch 45 | 98 | 189 | 456 | 0.341 | 0.177 | 0.233 | 0.132 |
| cp_real_v1 locked Desktop-JSON test | 64 | 239 | 312 | 0.211 | 0.170 | 0.189 | 0.104 |
| cp_real_v1, excluded 9 PXC'nin JSON mesh'i | 4 | 16 | 14 | 0.200 | 0.222 | 0.211 | 0.118 |
| cp_real_v1, ayni 9 katalogun WSCAD STEP'i | 4 | 64 | 14 | 0.059 | 0.222 | 0.093 | 0.049 |

WSCAD STEP satiri su sabit operating point ile olculdu:

- checkpoint: checkpoints/cp_real_v1_best.ckpt
- heat threshold: 0.30
- min votes: 1
- NMS ve match yaricapi: 5 mm
- 9/9 STEP frame alignment residuali 2 mm altinda
- beklenen 18 CP'ye karsi 68 prediction
- 68 predictionin 48'i tek-vote
- yalniz dort TP'nin localization error'u 3.78-4.46 mm; 5 mm esigin sinirinda
- matched prediction angle error ortalamasi 83.3 derece
- bridge JSON label noktalarinin en yakin mesh vertexine mesafesi 3.2-13.1 mm;
  "opening mouth" ile "recessed center" semantigi acikca kilitlenmeli

Bridge uzerindeki tani-amacli threshold sweep de sorunu cozmuyor: STEP'te en iyi
F1 yaklasik 0.12. Bu bridge artik tuning tarafindan goruldugu icin final test
degil, yalniz regresyon/dev setidir.

Ayni katalogda input yogunlugu farki:

| Katalog | WSCAD STEP vertex | Desktop JSON vertex | STEP/JSON |
|---|---:|---:|---:|
| 3031238 | 26,913 | 639 | 42x |
| 3036550 | 48,361 | 965 | 50x |
| 3048357 | 97,527 | 157 | 621x |
| 3211813 | 42,799 | 762 | 56x |
| 3212140 | 49,316 | 677 | 73x |

Modelin xyz, normal, curvature, concavity ve edge feature'lari yerel kNN
yogunluguna bagli. Yuzlerce vertexlik source ile on-binlerce vertexlik STEP
ayni input dagilimi degil.

### 2.5 WSCAD envanteri

- [x] all_wscad_stp altinda 3,895 STEP, toplam yaklasik 2.76 GB.
- [x] 3,699 benzersiz SHA-256 payload; 196 byte-identical fazla kopya.
- [x] 3,895/3,895 dosyada ISO-10303 STEP imzasi var.
- [x] 3,895/3,895 dosyada B-rep entity var; 10 KB altinda dosya yok.
- [x] Dort eski manifestte 2,066 satir var; 191 listelenen dosya artik yok.
- [x] Klasordeki 2,020 STEP dort manifestin hicbirinde yok.
- [ ] Bu envanter tek, guncel ve hash-pinned manifestte birlestirilecek.

## 3. Degismez kurallar

- [ ] Desktop JSON klasoru read-only source-of-truth kalacak; temizlenmis veri
  baska bir derived klasore/manifeste yazilacak.
- [ ] Supervised CP loss'a wscad_corpus_v*, step_openings --label-corpus,
  rect-slot auto labels veya baska CAD pseudo-label girmeyecek.
- [ ] Dokuz eski bridge parcasi yeni final test diye kullanilmayacak; sonuclari
  artik goruldu.
- [ ] Yeni final WSCAD test kataloglari, payload hashleri ve yakin geometri
  aileleri source/train/dev ile cakismayacak.
- [ ] Split bir kez manifestle kilitlenecek; seed degisince split degismeyecek.
- [ ] Aynı veya yakin aileler union halinde tek splitte tutulacak:
  exact vertex hash + geometry group + normalize edilmis katalog-family.
- [ ] Threshold yalniz dev setinde secilecek; final test bir kez acilacak.
- [ ] Her raporda TP/FP/FN, P/R/F1, Jaccard, loc, angle, coverage, seed ve
  checkpoint SHA yazilacak.
- [ ] Config preflight YAML'daki bilinmeyen/yanlis yazilmis anahtarlarda FAIL
  verecek; typo sessizce yok sayilmayacak.
- [ ] "Best" checkpoint secilecek; last.ckpt otomatik production olmayacak.
- [ ] Selective prediction kullanilirsa coverage ayri yazilacak; full-coverage
  %90 diye sunulmayacak.
- [ ] GPU run baslatmadan once data, split, config ve eval preflight PASS olmali.

## 4. P0 - Once veri ve benchmarki dogru kur

### 4.1 Source audit manifesti

- [ ] [KOD] preflight_audit.py'yi makine-okunur JSON cikisi verecek sekilde
  genislet veya ayri audit_json_source.py yaz.
- [ ] Her 479 parca icin su alanlari kaydet:
  part_nr, source_path, source_sha256, manufacturer, n_vertices, n_faces,
  raw_cp_count, unique_cp_count, bbox, direction_valid_count, exact_mesh_hash,
  geometry_group, catalog_family, scope_tag, split.
- [ ] SEW parcasinda lokasyon targetini tut; sifir/kararsiz direction targetlari
  icin direction-loss maskesi ekle.
- [ ] Negatif face index, non-finite/malformed CP ve bos/sifir direction icin
  loader validation ekle.
- [ ] Diger 48 non-unit direction loader tarafindan normalize ediliyor; hangi
  dosyalarda oldugunu manifestte isaretle.
- [ ] Scope'u prefix veya bbox ile otomatik karar verme. Her source parcasini
  broad_pretrain, target_terminal, out_of_scope, qa_needed olarak etiketle.

Kabul:

- 479/479 satir.
- Source hash degisirse preflight FAIL.
- Invalid direction loss'a girmiyor.
- Splitler arasinda exact/near-family cakismasi sifir.

Mevcut denetim komutu:

~~~powershell
.venv\Scripts\python.exe preflight_audit.py "C:\Users\DE00024082\Desktop\JSON" --exclude-parts-file human_gt_holdout.txt
~~~

### 4.2 WSCAD master manifesti

- [ ] [KOD] results/wscad_master_manifest_v1.json olustur.
- [ ] Her STEP icin absolute/relative path, catalog, manufacturer, bytes,
  sha256, ISO-10303 check, geometry bbox/volume/vertex count, duplicate_group,
  source_manifest, scope, label_status ve split kaydet.
- [ ] Catalog parserini underscore ve nokta iceren numaralarda test et; mevcut
  regex canli klasorde en az 99 filename'i eksik/yanlis parse edebiliyor.
- [ ] Evalde substring glob ile "ilk eslesen STEP" secme. Exact manifest path
  ve SHA kullan; benzer katalog veya duplicate dosyada yanlis geometri riski var.
- [ ] 196 byte-identical kopyadan yalniz bir canonical kayit benchmarka girecek.
- [ ] Manifestte olmayan 2,020 dosyanin provenance'i bulunmadan bunlar final
  benchmark adayi olmayacak.
- [ ] Bos/parse edilemeyen veya yanlis scope STEP, quarantine statusu alacak.

Kabul:

- 3,895/3,895 dosya manifestte.
- Mevcut dosya, manifest ve hash sayilari birebir.
- Missing=0, bad STEP signature=0.

### 4.3 Insan GT'li gercek WSCAD benchmark

- [ ] [INSAN QA] Mevcut 9 bridge'i regression/dev olarak tut.
- [ ] [INSAN QA] En az 21 yeni, benzersiz aile ekleyip dev setini en az
  30 WSCAD ailesine ve 200 GT CP'ye cikar.
- [ ] [INSAN QA] Ayrica hicbir deneyde acilmayacak en az 30 benzersiz aile ve
  200 GT CP'li final test hazirla.
- [ ] Final icin 200 CP minimum, 300+ CP tercih edilen istatistik hedefidir.
- [ ] Phoenix Contact ve Wago; rect-slot, round/push-in, screw, PE, fuse,
  multi-level ve zor clutter tiplerini stratify et.
- [ ] GT'yi STEP'in kendi koordinat frame'inde ABB JSON semasiyla kaydet.
- [ ] Her CP icin Point, InsertDirection, type/family ve QA reviewer yaz.
- [ ] En az %20 parcayi ikinci kisi bagimsiz kontrol et; uyusmazligi kapatmadan
  benchmarka alma.
- [ ] Final test parcalarini catalog-family, geometry ve SHA ile source/dev'den
  ayir; listeyi read-only kilitle.

Kabul:

- Dev >=30 unique aile ve >=200 CP.
- Final >=30 unique aile ve >=200 CP.
- Her iki sette duplicate payload/family leakage=0.
- Label QA PASS=100%.

## 5. P1 - Modelden once paired-domain sanity

Bu bolum bilerek "leaky diagnostic"tir; raporlanan accuracy degildir. Amac ayni
fiziksel parcayi iki farkli mesh yolunda modele verince nerede bozuldugunu bulmak.

- [ ] [KOD] Dokuz PXC JSON ile dokuz STEP twinini esleyen
  paired_domain_probe.py yaz.
- [ ] Ayni katalog icin JSON ve STEP'i cad_eval frame alignment ile esle.
- [ ] JSON-mesh -> remesh/simplify -> STEP-mesh arasinda su degerleri raporla:
  vertex density, nearest-neighbor spacing, bbox, normals/curvature distribution,
  heatmap peak sayisi, prediction count ve aligned prediction consistency.
- [ ] Bilgi-ceiling testi yap: 157-965 vertexlik PXC JSON meshlerinde CP
  cevresinde ayirt edici geometri yoksa yalniz bu inputtan %90 beklenemez.
- [ ] Es katalog icin Desktop JSON labelini aligned STEP mesh inputuna tasiyan
  kontrollu A/B kur. Label provenance Desktop JSON kalir; alignment ve
  opening-mouth/direction semantigi insan QA'dan gecmeden train verisi sayilmaz.
- [ ] 9 PXC JSON üzerinde kucuk overfit modeli egit; ayni 9 JSON'da F1 >=0.95
  olmadan mimari deneyi kabul etme.
- [ ] Ayni model ayni kataloglarin WSCAD STEP twinlerinde F1 >=0.85 gormeden
  479-parcalik uzun run baslatma.
- [ ] Bu skorlari final accuracy tablosuna koyma; yalniz input-pipeline gate'i.

Neden: mevcut model ayni dokuz partta JSON mesh F1~0.21 iken STEP'te F1~0.09.
Once temel ogrenme ve domain adapter duzelmeden daha cok epoch anlamsiz.

## 6. P2 - Domain gap'i tek degiskenli ablationlarla kapat

### E1. Eksen/permutasyon equivariance

- [ ] [KOD] aug_axis_permute ekle: 24 proper signed-axis permutationdan birini
  vertex, CP point ve direction'a birlikte uygula.
- [ ] Arbitrary SO(3) ile ayni deneyde karistirma. Ilk ablation yalniz cube
  rotations olsun.
- [ ] Cube rotationdan sonra ayri bir controlled run ile arbitrary SO(3)
  rotation acik/kapali A/B yap; PXC direction dagilimi genel korpustan farkli.
- [ ] JSON ve STEP frame'lerinin axis permutation farkina karsi prediction
  equivariance unit testi ekle.
- [ ] Direction geri donusumunu test et; point dogruyken 90 derece direction
  hatasi kabul edilmesin.

Gerekce: tum source directionlarin %97.9'u eksen hizali olsa da PXC'de oran
yaklasik %32. WSCAD STEP frame'leri de JSON frame'ine gore signed axis
permutation tasiyor. Mevcut aug_rotate=false bu iki hedef farkini modele
gostermiyor.

### E2. Mesh yogunlugu parity

- [ ] [KOD] Source JSON ve STEP'i ayni fiziksel sampling politikasina sok.
- [ ] Raw vertex sayisini model ipucu olarak kullanma; sabit mm spacing/voxel
  veya area-weighted surface sampling dene.
- [ ] 0.3/0.5/1.0 mm STEP deflection ile tahmin kararliligini label'a bakmadan
  olc; ayni parcanin CP sayisi/konumu tessellationla degismemeli.
- [ ] Train ve inference icin ayni sampler, seed ve patch politikasini kullan.
- [ ] Kücük CP deliklerinin samplingde kaybolmadigini coverage testiyle kanitla.
- [ ] Once retrain yapmadan full-patch, patch=False uniform-14k ve fixed-N/voxel
  512/1k/2k/4k/8k/14k inputlarini ayni checkpointte karsilastir.
- [ ] Coverage diagnostic'i gercek inference yolu ile eslestir: PXC/WSCAD
  spatial patch, diger aileler uniform subsample.
- [ ] One-peak CenterNet targetini K-nearest/radius-positive ve voxel/area
  normalized negative encoding ile A/B yap; remesh yogunlugu degisince target
  semantigi degismemeli.

### E3. Patch merge ve false-positive kontrolu

- [ ] [KOD] Patch bazli outputlari part seviyesinde tek heatmap/cluster olarak
  birlestir; overlap'tan gelen kopyalari fiziksel NMS ile bastir.
- [ ] Patch sayisi degisince prediction sayisi degismemeli.
- [x] Mevcut tani: 68 WSCAD predictionin 48'i tek-vote. Train tarafinda ise
  pratikte yalniz tek buyuk PXC ve augmentation klonlari patch olarak gorulmus;
  inference'ta 9/9 WSCAD full-density patch. Bu train/infer temsil uyusmazligidir.
- [ ] WSCAD devde FP'leri screw hole, rail, cosmetic slot, duplicate patch,
  wrong surface ve far clutter olarak etiketle.
- [ ] Hard-negative vertex/region mining'i yalniz gercek Desktop JSON negatif
  bolgelerinden yap; CAD pseudo-CP ekleme.
- [ ] --hard-mining ablationini domain parity duzeldikten sonra tek basina dene.

### E4. Local geometry feature parity

- [ ] [KOD] Absolute xyz'ye bagimliligi azaltan local relative-coordinate,
  normal ve curvature feature ablationi yap.
- [ ] Aynı parcayi cube rotation ve yeniden tessellation sonrasi ayni embedding/
  heatmap vermeye zorlayan consistency loss dene.
- [ ] WSCAD STEP'ler label kullanmadan self-supervised geometry pretrainingde
  kullanilabilir; supervised CP head yine yalniz Desktop JSON labeli gorecek.
- [ ] Sentetik supervised init kullanilacaksa ayri provenance kolu olarak raporla;
  "yalniz Desktop JSON" diye adlandirma.

### E5. Direction label hijyeni

- [ ] [KOD] Direction-valid mask'i loss ve metric yoluna ekle.
- [ ] Sifir, non-finite veya coincident-cancel directionlar point loss'unda
  kalabilir; direction loss/angle metricten cikar.
- [ ] Axis-permutation augmentation direction'i birebir transform etmeli.
- [ ] Point F1 ile direction pass rate'i ayri raporla.
- [ ] Best-checkpoint seciminde point F1 yaninda direction gate uygula; mevcut
  point esleme F1'i 90/180 derece yanlis direction'i cezalandirmiyor.

P2 kabul kapisi:

- Paired 9 JSON train F1 >=0.95.
- Ayni-katalog STEP transfer F1 >=0.85.
- STEP deflection/sampling degisiminde median CP konum farki <=1 mm.
- Ayni labelled mesh 1x/4x/16x/64x remeshlenince F1 ve prediction-count drift
  3 puandan az.
- Canonical sampling A/B, recall'i dusurmeden WSCAD dev F1'i en az 10 puan
  artiriyor ve prediction/part sayisini GT dagilimina yaklastiriyor.
- Aligned-STEP-input A/B, coarse-JSON-input koluna gore WSCAD dev F1'i en az
  20 puan artirmiyorsa input bilgisi/label kontrati yeniden inceleniyor.
- Patch sayisi degisiminde FP patlamasi yok.
- Bu kapilar gecmeden hyperparameter sweep veya uzun multi-seed run yok.

## 7. P3 - Sadece dogru JSON labeliyle training tasarimi

### 7.1 Development asamasi

- [ ] [RUN] Saf provenance baseline: scratch init + kilitli group split.
- [ ] [RUN] Synthetic-init kolunu ayri A/B olarak tut; saf JSON sonucu diye
  birlestirme.
- [ ] [RUN] Once tum manufacturer JSON ile broad representation train et.
- [ ] [RUN] Sonra insan-onayli target_terminal source subsetinde specialist
  fine-tune et. Prefix=PXC veya bbox<250 otomatik target etiketi degildir.
- [ ] Duplicate katalog varyantlarini ornek sayisi gibi oversample etme; her
  geometry/family grubunun toplam loss agirligini esitle.
- [ ] Out-of-scope 12 PXC'yi target specialist stage'inden cikar; broad
  pretrainingde tutulup tutulmayacagini kontrollu A/B ile olc.
- [ ] Mevcut automatic family augmentationin out-of-scope PXC'yi buyutmesine
  izin verme.
- [ ] Part-balanced sampler ve hard-negative mining'i ayri ablationlar yap.
- [ ] En az 3 model seed'i ayni frozen splitte calistir.

### 7.2 Tum 479 parcayi kullanma

- [ ] Yeni bagimsiz WSCAD final test kilitlendikten ve mimari/config tamamen
  secildikten sonra production modeli tum 479 Desktop JSON ile yeniden fit et.
- [ ] Bu final refitte eski dokuz PXC JSON da kullanilabilir; onlar artik
  dev/paired sanity'dir, final test degildir.
- [ ] Yeni final WSCAD testte bu dokuz katalogun aynisi veya yakin geometry
  ailesi bulunmayacak.
- [ ] Final refit sonrasi threshold, sampler veya config degistirme; yalniz daha
  once devde kilitlenen operating point'i kullan.

Bu sira hem "JSON'un icindekilerin tamami ile train" sartini yerine getirir hem
de ayni veriyi accuracy kaniti olarak kullanma hatasini engeller.

## 8. Kontrollu deney matrisi

Her satir tek ana degisken tasiyacak:

| ID | Degisiklik | Source labels | Zorunlu rapor |
|---|---|---|---|
| R0 | Scratch, mevcut pipeline | Desktop JSON | source val/test + WSCAD dev |
| R1 | R0 + 24 cube rotation | Desktop JSON | paired transfer + WSCAD dev |
| R2 | R1 + sampling parity | Desktop JSON | density stability + WSCAD dev |
| R3 | R2 + patch merge/NMS fix | Desktop JSON | FP bucketleri + WSCAD dev |
| R4 | R3 + local geometry features | Desktop JSON | rotation/remesh consistency |
| R5 | R4 + target specialist fine-tune | Desktop JSON target tags | family metrics |
| R6 | R5 + hard mining | Desktop JSON negatives | P/R ve FP bucket delta |
| R7 | En iyi config, 3 seed | Desktop JSON | mean/std ve worst seed |
| PROD | Kilitli config, tum 479 refit | Desktop JSON | final WSCAD test bir kez |

Her deney icin kaydet:

- git commit ve dirty diff ozeti
- source/split/manifest SHA
- exact command ve YAML
- init checkpoint provenance
- seed
- train/val/test part ve unique-family sayisi
- TP/FP/FN, P/R/F1/Jaccard
- loc/angle error
- per-manufacturer, per-slot-type ve per-CP-count bucketleri
- inference time, peak VRAM ve prediction/part dagilimi

Stop kurali:

- Ayni hata sinifi iki ardışık kontrollu deneyde iyilesmiyorsa LR/epoch
  kurcalamayi birak; veri/representation gate'ine geri don.
- WSCAD dev kazanci olmadan source-val kazanci production ilerlemesi sayilmaz.
- Threshold sweep model/representation duzelmeden yeni run gerekcesi olamaz.

## 9. Milestone kapilari

### M0 - Pipeline ogrenebiliyor

- [ ] 9-parca paired JSON overfit F1 >=0.95.
- [ ] Ayni-katalog STEP transfer F1 >=0.85.
- [ ] Direction error <=15 derece.

### M1 - Unseen WSCAD dev

- [ ] En az 30 aile / 200 CP dev setinde P ve R ayri ayri >=0.70.
- [ ] FP/part ve FN/part dagilimi kayitli.
- [ ] Hicbir tek aile toplam skoru domine etmiyor.

### M2 - Promotion adayi

- [ ] 3 seed WSCAD dev mean micro-F1 >=0.85.
- [ ] Worst seed F1 >=0.82.
- [ ] Per-manufacturer ve rect-slot F1 >=0.80.
- [ ] loc <=2 mm, angle <=15 derece.
- [ ] Macro/per-part F1 >=0.85; birkac cok-CP parca pooled skoru tasiyamaz.

### M3 - Nihai %90

- [ ] Config, weights, threshold ve manifest freeze.
- [ ] Final WSCAD test ilk kez acilir.
- [ ] Full-coverage micro-F1 >=0.90.
- [ ] Precision >=0.90 ve recall >=0.90.
- [ ] Jaccard ayrica raporlanir.
- [ ] Per-manufacturer F1 >=0.85.
- [ ] loc <=2 mm ve angle <=15 derece.
- [ ] Alignment/parse/skipped parca=0 ve coverage=%100.
- [ ] Sonuc TP/FP/FN ve confidence interval ile RESULTS.md'ye yazilir.

Final test gecmezse ayni set uzerinde tuning yapilmaz. O set regression/dev olur;
yeni bir sealed final test gerekir.

## 10. Simdi yapilacak ilk isler

### Tamamlanan baseline komutlari

Real-val threshold kilidi:

~~~powershell
.venv\Scripts\python.exe sweep_val.py checkpoints\cp_real_v1_best.ckpt --corpus "C:\Users\DE00024082\Desktop\JSON" --list _real_val.txt --device cuda --thresholds "0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.50"
~~~

WSCAD 9 ML-only prediction:

~~~powershell
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe predict.py checkpoints\cp_real_v1_best.ckpt _cad_eval_pxc --out results\json_restart_cp_real_v1_wscad9_preds.json --device cuda --heatmap-thresh 0.30 --min-votes 1 --max-gpu-verts 14000
~~~

Human/uretici GT skoru:

~~~powershell
$env:PYTHONUTF8="1"
.venv\Scripts\python.exe cad_eval.py --preds results\json_restart_cp_real_v1_wscad9_preds.json --gt-dir "C:\Users\DE00024082\Desktop\JSON" --step-dir _cad_eval_pxc --out results\json_restart_cp_real_v1_wscad9_eval.json
~~~

### Siradaki uc teslimat

- [ ] 1. Source + WSCAD master manifest ve leakage-safe frozen split.
- [ ] 2. paired_domain_probe + direction mask + 24 cube rotation.
- [ ] 3. Sampling parity ablationi; ancak M0 gectikten sonra yeni full run.

## 11. Yapilmayacaklar

- [ ] cp_hp_v31/v28/v24 skorunu yeni %90 kaniti diye kullanma.
- [ ] WSCAD CAD pseudo-labellarini supervised source'a ekleme.
- [ ] Dokuz bridge uzerinde threshold secip onu final test diye raporlama.
- [ ] Tum 3,895 STEP'i label varmis gibi accuracy hesabina sokma.
- [ ] Duplicate dosya/katalog varyantlarini bagimsiz veri sayma.
- [ ] Once domain gap'i kapatmadan LR, weight decay veya epoch sweep yapma.
- [ ] --auto ile CAD fallback sonucunu ML sonucu diye sunma.
- [ ] Sadece macro-F1 veya sadece en iyi seed ile promotion yapma.
- [ ] %90'i garanti etme: mevcut dürüst WSCAD F1 0.093; bu plan kapili ve
  kanita dayali ilerleme yoludur.
