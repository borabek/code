# Scheffler master tezine bağlı 7 günlük F1 / accuracy %90 planı

Tarih: 16 Temmuz 2026  
Süre: 16-22 Temmuz 2026, 7 takvim günü  
Tez: `C:\Users\DE00024082\Desktop\Neuer Ordner\Masterarbeit_Scheffler.pdf`  
Tez SHA-256: `6B554F65A40355CC9FAD3A02E92828D5F739609A84C32095077F5E61DFA85A02`

Bu dosya bir başarı garantisi değil, yedi günlük kapılı bir saldırı planıdır. Mevcut
kanıtla bir haftada hem F1 hem gerçek accuracy/Jaccard %90 sözü bilimsel olarak
verilemez. Ama aşağıdaki kapılar geçilirse sonuç dürüst biçimde ölçülebilir; bir kapı
geçilmezse yanlış modeli daha uzun eğitmek yerine kök neden raporlanır.

## 1. En başta hedefi doğru tanımla

Bu projede `micro_accuracy`, klasik sınıflandırma accuracy'si değil Jaccard'dır:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
Jaccard   = TP / (TP + FP + FN)
Jaccard   = F1 / (2 - F1)
```

Bunun sonucu:

- `F1 = 0.90` yalnız `Jaccard = 0.8182` eder.
- `Jaccard = 0.90` için `F1 = 0.9474` gerekir.
- Kullanıcının istediği iki metriğin de %90 olması için gerçek nihai kapı
  `Jaccard >= 0.90` ve dolayısıyla `F1 >= 0.9474` olacaktır.
- Ek zorunlu kapılar: `precision >= 0.90`, `recall >= 0.90`, full coverage ve
  5 mm Hungarian/tekil eşleme.
- Klasik vertex accuracy yalnız yardımcı metrik olacaktır. Housing/background
  baskınlığıyla kolayca yüksek göründüğü için başarı başlığı yapılmayacaktır.

İki ölçüm katmanı ayrı tutulacaktır:

1. Semantik yüzey: sınıf başına ve macro Dice/Jaccard.
2. Nihai connection point: TP/FP/FN, precision, recall, F1, Jaccard, konum ve yön.

Tezdeki yüzey Dice/Jaccard değeri, tek başına connection-point başarısı değildir.

## 2. Tezin gerçekten söylediği

Tezin uygulanacak omurgası şudur:

- DMME: problem ve metrik sözleşmesi, veri anlama, hazırlama, modelleme,
  değerlendirme ve deployment sırasıyla kapılanır (`Masterarbeit_Scheffler.md:950-976`).
- Geometrik duplicate'ler SHA-256 ile kaldırılır; metadata ile kapsam filtresi
  uygulanır (`:1098-1138`).
- STEP, hataları onarılmış yaklaşık watertight 2-manifold meshe çevrilir; tüm
  örnekler yaklaşık 6.000 vertex / 12.000 triangle yoğunluğuna getirilir
  (`:1227-1269`, `:1823-1829`). Bu sayı teorik optimum değil GPU/MeshCNN
  kısıtıdır; asıl kural train ve inference'ın aynı dönüşümü kullanmasıdır.
- Her vertex tek bir semantik sınıf alır; tezde beş sınıf Blender vertex-group
  etiketiyle elle hazırlanmıştır (`:1323-1369`).
- Ana model DiffusionNet'tir. En iyi tez başlangıcı `XYZ`, `lr=1e-3`, decay
  `100 x 0.75`, weighted NLL, 3 diffusion block, width 64, 64 eigenvector ve
  dropout 0.3'tür (`:1490-1502`, `:1943-1959`).
- Augmentation: sürekli rastgele rotasyon; lognormal gürültü `mu=0,
  sigma=0.005` ve smoothing; foreground üzerinde 1-10 noktalı ARAP deformasyonu
  (`:1271-1321`). Tez olasılık ve ARAP şiddeti vermediği için bunlar uydurulmayacak,
  tek-değişkenli ablation ile seçilecektir.
- Instance ayrımı k-means yerine graph connectivity, boundary peeling ve
  çözünürlüğe bağlı küçük-cluster filtresiyle yapılır. CP merkezi/yönü cluster,
  opening-boundary ve surface merkezlerinden çıkarılır (`:1548-1612`).
- HPO Ray Tune + ASHA ile başarısız koşuları erken durdurur (`:1681-1699`,
  `:1921-1929`).
- Final test tuning için kullanılmaz; bir kez açılır (`:694-702`).

Tezin dürüst referans sonucu:

| Ölçüm | Sonuç |
|---|---:|
| Klasik vertex accuracy | yaklaşık %80,5; dengesizlik yüzünden yanıltıcı |
| Ortalama Dice | %65,60 |
| Ortalama Jaccard | %51,41 |
| Weighted Jaccard | %70,52 |
| YOLO mAP@0.5 | %93,11; CP F1/Jaccard değildir |

Tez sonuçları `Masterarbeit_Scheffler.md:1967-2005` ve `:2155-2163` içindedir.
Yazarın otonom kullanım için önerdiği eşik bile her sınıfta Jaccard %50'dir.
Dolayısıyla tez %90'ı göstermemiştir; yalnız doğru yolu tarif etmektedir.

## 3. FBI teşhisi: mevcut repo neden henüz tez yöntemi değil

### 3.1 Yanlış başlık altında farklı görev eğitilmiş

`train_cp.py --backbone diffusionnet`, tezdeki beş sınıflı semantik segmentasyonu
çalıştırmıyor. `cp_regressor.py` içindeki model her vertex için yedi kanal
`(heat, offset x/y/z, direction x/y/z)` üretiyor ve CenterNet tipi loss + NMS
kullanıyor. `run_real_dn.yaml` ayrıca augmentation'ı kapatmış durumda.

Bu nedenle `cp_real_dn` düşük skoru DiffusionNet tez yaklaşımını çürütmez; yalnız
doğrudan sparse keypoint-regression başlığının bu veride çalışmadığını gösterir.

Teze uygun parçalar repoda zaten mevcut:

- `diffusionnet.py`: gerçek vertex-semantic DiffusionNet, weighted NLL/CE ve tez
  başlangıç hiperparametreleri.
- `augment.py`: rotasyon, lognormal `sigma=0.005`, smoothing ve ARAP.
- `connector3d.py`: graph-connected instance, boundary peeling, merkez ve normal.
- `infer_pipeline.py`: semantic label -> instance -> connector graph hattı.
- `metrics.py`: Dice/Jaccard ve CP değerlendirme yardımcıları.

Eksik ana köprü, Desktop JSON CP point/direction etiketlerinden güvenilir semantic
surface maskesi üretmektir. JSON dosyalarında hazır beş sınıflı vertex label yoktur.

### 3.2 Aynı geometriye çelişkili doğru cevaplar var

Canlı denetimde ham `cp_real_v1` manifestindeki 470 parçanın yalnız 167 benzersiz
vertex hash'i olduğu görüldü. Yetmiş üç duplicate-geometri grubu 376 parçayı
kapsıyor. Bunların 13 grubu / 86 parçası aynı geometriye rağmen farklı CP sayısı
taşıyor; örnekler arasında aynı ABB/ACS380 geometrisinin farklı CP sayıları var.

Salt geometri modeli aynı girdiden iki farklı doğru cevap üretemez. Bu bir tuning
problemi değil, label sözleşmesi/Bayes ceiling problemidir. İlk gün şu ayrım
yapılmadan hiçbir yeni ana koşu başlatılmayacak:

- Aynı fiziksel açıklığın birden fazla mantıksal terminal olarak tekrarıysa fiziksel
  CP seviyesinde canonical dedup yapılır.
- Etiket/version hatasıysa insan incelemesiyle düzeltilir.
- Gerçekten aynı geometri fakat ürün metadata'sına göre farklı aktif CP varsa,
  tezde önerilen geometry+metadata hibriti kullanılır. İlgili metadata WSCAD
  inference sırasında yoksa bu örnekler geometry-only eğitimden karantinaya alınır.

Kapı: raw geometry hash başına tek ve tutarlı fiziksel CP hedef kümesi. Çelişki
kaldığı sürece full-coverage Jaccard %90 iddiası yasaktır.

### 3.3 Bugünkü remesh kontrollü ablation değil

`thesis_remesh.py` yaklaşık hedef yazmasına rağmen üretilen corpus sabit yoğunlukta
değildir:

| Corpus | Dosya | Vertex median | p95 | max |
|---|---:|---:|---:|---:|
| `_remeshed` | 478 | 8.154 | 36.181 | 92.469 |
| `_remeshed12k` | 479 | 14.295 | 36.996 | 134.694 |

`_remeshed` üretiminde bir dosya başarısızdır. Daha önemlisi, remesh her duplicate'i
nümerik olarak farklılaştırdığı için derived `geometry_key` değişmiş ve split de
değişmiştir:

| Girdi | train / val / test |
|---|---:|
| Ham JSON v1 | 334 / 80 / 56 |
| `_remeshed` | 353 / 62 / 54 |
| `_remeshed12k` | 342 / 54 / 74 |

Bu nedenle bugünkü remesh koşuları tek-değişkenli karşılaştırma değildir. Split,
remesh öncesi `raw_parent_id + raw geometry hash + catalog family` ile dondurulacak;
tüm derived meshler bu split'i miras alacaktır.

### 3.4 Bugünkü dürüst skor fotoğrafı

| Koşu / benchmark | Pooled micro-F1 | Jaccard | Not |
|---|---:|---:|---|
| Saf JSON scratch, en iyi real-val | 0,1705 | yaklaşık 0,093 | saf source başlangıcı |
| `cp_real_v1`, en iyi real-val | 0,2331 | 0,1319 | synthetic init içerir |
| `cp_real_v1`, 9 PXC WSCAD STEP | 0,0930 | 0,0488 | artık tuning görmüş diagnostic |
| 6k-remesh koşusu, test | 0,1939 | 0,1074 | split drift; doğrudan kıyaslanamaz |
| direct-DiffusionNet, test | 0,0801 | 0,0417 | semantic tez görevi değildir |
| 12k-remesh, en iyi val | 0,1429 | 0,0769 | split drift |
| 12k-remesh, final test | 0,1205 | 0,0641 | split drift |

Dokuz exact PXC STEP twinini doğrudan eğitimde gösteren M0 memorization deneyi
`F1=0,941 / Jaccard=0,889` görmüştür. Bu yalnız pipeline'ın öğrenebildiği bir
leaky sanity-check'tir; unseen WSCAD accuracy kanıtı değildir.

### 3.5 Hedef-domain örneği eksik

Doğrulanmış hedef-benzeri Desktop JSON kapsamı şu an yalnız dokuz PXC terminal
block ve 18 fiziksel CP'dir; bunlar `_bridge_test.txt` ile gerçek koşulardan
çıkarılmıştır. Kalan 12 PXC parça terminal block değildir. Model bu durumda hedef
sınıfın doğrulanmış örneğini görmeden 3.895 WSCAD STEP'e genellenmeye zorlanmaktadır.

Tez, eğitimde benzer feature geometrileri olan parçaların daha iyi segmentlendiğini
açıkça söyler (`Masterarbeit_Scheffler.md:2163`). Bu kapsam açığı çözülmeden yalnız
daha uzun eğitim %90'a götürmez.

## 4. Değişmez veri ve kanıt kuralları

- Tek supervised training kaynağı `C:\Users\DE00024082\Desktop\JSON` olacaktır.
- JSON'dan türetilen surface maskeleri de provenance olarak JSON sayılır; her
  türetme parametresi manifestte tutulur.
- WSCAD insan GT'si yalnız development/final ölçümde kullanılır; gradient,
  pseudo-label veya supervised fine-tune için kullanılmaz.
- Bütün 479 dosya korunur ama tezdeki metadata filtresine uygun olarak in-scope,
  conflict, out-of-scope ve broken gruplarına ayrılır. Her dosyayı körlemesine loss'a
  sokmak teze uygun değildir.
- Source split remesh öncesi raw identity/family/hash ile bir kez dondurulur.
- Aynı katalog ailesi, near-duplicate geometri ve byte/shape duplicate farklı
  splitlere giremez.
- Dokuz PXC bridge yalnız pipeline diagnostic/dev olabilir. Nihai test olamaz.
- Yeni WSCAD dev: en az 30 geometrik farklı terminal block ve en az 200 insan GT CP.
- Yeni untouched final: farklı en az 30 terminal block ve en az 200 insan GT CP.
  Daha güçlü istatistiksel iddia için 500-1.000 CP tercih edilir; Wilson güven
  aralığı da raporlanır.
- Final test dosyaları ve GT hashleri Gün 1'de kilitlenir, Gün 7'ye kadar açılmaz.
- CAD pseudo-label, weighted-only skor, YOLO mAP veya selective coverage %90 diye
  sunulmaz.

## 5. Hedef mimari

```text
Desktop JSON mesh + CP point/direction
        |
        v
raw-ID manifest -> scope/conflict/dedupe -> frozen family split
        |
        v
deterministic manifold repair + gerçekten sabit 6k/12k remesh
        |
        v
JSON CP -> connected semantic opening/contact surface mask + ignore boundary
        |
        v
tez-konfigürasyonlu semantic DiffusionNet (weighted NLL)
        |
        v
graph connectivity -> boundary peel -> min-cluster filter
        |
        v
opening center / surface center / normal -> CP list
        |
        v
5 mm full-coverage CP F1 + Jaccard

WSCAD STEP -> AYNI repair/remesh/config ----------------------^
```

JSON'da tezdeki beş vertex sınıfı bulunmadığı için ilk gerçekçi sınıf sözleşmesi:

- `0`: housing/background
- `1`: CP opening/contact surface
- `ignore`: mask sınırında belirsiz vertexler; loss'a girmez

CP adı/metadata, contact ve cable-entry ayrımını en az %95 doğrulukla veriyorsa
sonra ayrı sınıflara bölünür. Güvenilir olmayan isimlerden sahte beş sınıf üretmek
teze bağlılık değildir.

Semantic DiffusionNet ana detector olur. Mevcut heat/offset/direction başlığı ancak
semantic instance içinde çalışan yerel bir refiner olarak ablation'a girebilir;
tüm mesh üzerinde ana detector olarak kullanılmaz.

## 6. Yedi günlük yürütme planı

### Gün 1 - 16 Temmuz: sözleşme, conflict temizliği ve donmuş benchmark

Yapılacaklar:

- [ ] `physical CP` ile logical terminal tekrarının kesin sözleşmesini yaz.
- [ ] 479 JSON için source manifest üret: raw hash, vertex/face sayısı, CP sayısı,
  fiziksel dedup CP sayısı, üretici, aile, scope ve reason.
- [ ] 13 conflict grubunu/86 parçayı görsel ve metadata destekli incele; canonical
  target, metadata-hybrid veya quarantine kararı ver.
- [ ] Train/val/test'i raw parent geometry + family üzerinden 70/20/10 dondur.
- [ ] Mevcut `_remeshed*` splitlerini deney kanıtı olarak emekliye ayır; checkpointleri
  silme, yalnız `non-comparable` olarak işaretle.
- [ ] Remesh kalite sözleşmesi kur: manifold/watertight QA, target-vertex bandı,
  median edge, CP->surface mesafesi, topology ve başarısız parça raporu.
- [ ] Yeni WSCAD dev/final kataloglarını seç, human-GT manifestlerini ve hashlerini
  kilitle. Bridge 9 finalden çıkarılır.
- [ ] Sıfır noktası raporuna bütün formülleri ve TP/FP/FN'i kaydet.

Teslimatlar:

- `[NEW] results/thesis_week/source_manifest.json`
- `[NEW] results/thesis_week/conflicting_geometry_labels.json`
- `[NEW] results/thesis_week/frozen_split.json`
- `[NEW] results/thesis_week/wscad_dev_manifest.json`
- `[NEW] results/thesis_week/wscad_final_locked_manifest.json`
- `[NEW] results/thesis_week/baseline.json`

Gün 1 geçiş kapısı:

- 479/479 dosya bir reason ile hesapta.
- Cross-split raw hash/family leakage sıfır.
- Aynı geometry-only girdiye çelişkili fiziksel CP hedefi sıfır; değilse metadata
  planı veya quarantine tamamlanmış.
- En az 30-part/200-CP dev hazır ve final manifest kilitli.

Kapı geçmezse training başlamaz.

### Gün 2 - 17 Temmuz: sabit remesh ve JSON'dan semantic label üretimi

Yapılacaklar:

- [ ] `thesis_remesh.py` çıktısını gerçekten hedef banda zorla. 6k kolu için
  `5.500-6.500`, 12k kolu için `11.000-13.000` vertex dışında kalan örneği kabul
  etme. Tekrar üret veya açıkça quarantine et.
- [ ] JSON ve WSCAD'e aynı repair/remesh sürümünü ve parametre hashini uygula.
- [ ] Derived corpus splitini yeniden hesaplama; Gün 1 raw split ID'sini miras al.
- [ ] `[NEW] cp_semantic_labels.py` ile her fiziksel CP için en yakın yüz/ray seed,
  graph-connected geodesic patch, local-normal filtresi ve belirsiz boundary ring
  üret. Euclidean kürenin karşı yüzeyi yanlış etiketlemesine izin verme.
- [ ] CP point ve InsertDirection'ın mesh/frame ile uyumunu ölç; off-surface ve
  ters-yön vakalarını bucketla.
- [ ] Üretici/aile/CP sayısına göre stratified en az 50 JSON parçayı ve bridge 9'u
  görsel QA et.
- [ ] Aynı CP için 6k/12k maskenin yüzey alanı ve instance sayısının stabilitesini
  ölç.

Teslimatlar:

- `[NEW] cp_semantic_labels.py`
- `[NEW] results/thesis_week/remesh_qa.json`
- `[NEW] results/thesis_week/label_audit.json`
- `[NEW] results/thesis_week/label_viz/`

Gün 2 geçiş kapısı:

- Kabul edilen parçaların %100'ü manifold/çözünürlük bandında.
- Fiziksel CP'lerin en az %95'i tam bir connected foreground maskesine map oluyor.
- İki farklı CP'nin yanlış tek maskede birleşmesi <%1.
- İnsan QA'da kabul >=%95.
- CP'lerin >%5'i yüzeyden 5 mm'den uzaksa otomatik weak-label yolu durur; insan
  vertex-mask düzeltmesi gerekir ve bir haftalık %90 takvimi yeniden değerlendirilir.

### Gün 3 - 18 Temmuz: gerçek tez DiffusionNet'i ve M0 ceiling testi

Yapılacaklar:

- [ ] `[NEW] train_cp_semantic.py` wrapper'ı ile `diffusionnet.py` içindeki gerçek
  semantic modeli kullan; `train_cp.py --backbone diffusionnet` kullanma.
- [ ] İlk koşu tez-best config: XYZ, lr 1e-3, decay 100/0.75, weighted NLL,
  3x64 block/width, 64 eigenvector, dropout 0.3.
- [ ] Sınıf ağırlıklarını yalnız frozen train split ve seçilen remesh çözünürlüğünden
  yeniden hesapla; tez sayılarını körlemesine kopyalama.
- [ ] Önce 12-20 stratified parçada augmentation kapalı overfit testi yap.
- [ ] Aynı checkpointi aynı parçaların JSON meshinde, sabit remeshinde ve exact
  WSCAD twin diagnostic'inde değerlendir.
- [ ] Sonra yalnız rotasyon ekle; tüm CP yönlerini ve maskeleri aynı dönüşümle taşı.
- [ ] Lognormal noise+smoothing ve ARAP'ı henüz ana koşuya topluca koyma.

Teslimatlar:

- `[NEW] train_cp_semantic.py`
- `[NEW] run_thesis_semantic_m0.yaml`
- `[NEW] results/thesis_week/m0_ceiling.json`

Gün 3 geçiş kapısı:

- Memorized JSON CP: `F1 >= 0.98`, `Jaccard >= 0.96`.
- Aynı-katalog, aynı deterministic-remesh STEP diagnostic:
  `F1 >= 0.92`, `Jaccard >= 0.85`.
- 6k/12k prediction-count farkı <=%5 ve CP konum drift medyanı <=1 mm.

Kapı geçmezse full training/HPO yapılmaz. Hata model kapasitesi değil label transfer,
remesh veya frame sözleşmesindedir.

### Gün 4 - 19 Temmuz: tam source eğitimi ve dar ASHA araması

Yapılacaklar:

- [ ] Frozen 70/20/10 source split üzerinde tez-best semantic baseline'ı eğit.
- [ ] En fazla 6-8 ASHA trial kullan. Bir GPU'da yüzlerce kombinasyon deneme.
- [ ] Arama alanını dar tut: XYZ/HKS; lr `5e-4, 1e-3, 2e-3`; width `64/128`;
  block `3/4`; eig `64/128`; dropout `0.2/0.3`.
- [ ] Önce weighted NLL. NLL + overlap/Tversky kolu yalnız tek-değişkenli ablation.
- [ ] Her koşuda source macro class Jaccard ve WSCAD-dev CP Jaccard raporla.
- [ ] Best seçiminde klasik vertex accuracy veya weighted-only skor kullanma.
- [ ] Rotasyon, noise+smoothing ve ARAP'ı E1/E2/E3 olarak tek tek ekle. Her
  augmentation'da point/direction/label transform testleri geçsin.

Deney sırası:

| ID | Tek değişken | Promotion şartı |
|---|---|---|
| E0 | tez-best semantic baseline | referans |
| E1 | sürekli rotation | dev Jaccard +>=0,02, source çökmesin |
| E2 | E1 + lognormal 0,005 + smoothing | dev Jaccard +>=0,02 |
| E3 | en iyi kol + ARAP | dev Jaccard +>=0,02, topology/label QA geçsin |
| E4 | 6k yerine 12k | instance recall artsın, FP patlamasın |
| E5 | local offset/direction refiner | yalnız instance içinde, CP Jaccard artsın |

Gün 4 geçiş kapısı:

- Source val foreground macro Jaccard >=0.75.
- WSCAD dev CP `F1 >= 0.80`, `Jaccard >= 0.67`.
- Hiçbir üreticide recall sıfır değil.
- Train/val farkı ve üç ana hata bucket'i raporlanmış.

### Gün 5 - 20 Temmuz: graph instance ve FP/FN avı

Yapılacaklar:

- [ ] `connector3d.py` graph connectivity hattını semantic maskeye bağla.
- [ ] Boundary peeling'i 0/1/2 iteration ablation ile seç.
- [ ] Min-cluster threshold'unu tezdeki sabit sayılardan değil, bu train splitteki
  gerçek foreground-instance lower quartile'ından hesapla ve çözünürlüğe ölçekle.
- [ ] Opening boundary center `v_o`, surface/depth center `v_s`, convex-hull/area
  merkezlerini ayrı değerlendir; CP noktası için devde en iyi ama sabit kuralı seç.
- [ ] Normal/yönü `v_o-v_s` ve JSON InsertDirection ile denetle; nokta F1 ve direction
  pass rate ayrı raporlanır.
- [ ] FP'leri screw hole, rail slot, cosmetic opening, duplicate fragment ve patch
  merge olarak etiketle.
- [ ] Hard negative'leri yalnız Desktop JSON'dan seç. WSCAD dev GT'sini loss'a sokma.
- [ ] Threshold/min-size/boundary parametrelerini yalnız WSCAD devde kalibre et.

Teslimatlar:

- `[NEW] cp_instances.py` veya mevcut `connector3d.py` için ince adapter
- `[NEW] eval_cp_semantic.py`
- `[NEW] results/thesis_week/error_buckets_day5.json`

Gün 5 geçiş kapısı:

- WSCAD dev `precision >= 0.87`, `recall >= 0.87`.
- WSCAD dev `F1 >= 0.90`, `Jaccard >= 0.82`.
- Full coverage %100; parça reddetme/selective prediction yok.
- Median matched-point hata <=2,5 mm; yön metriği ayrıca görünür.

### Gün 6 - 21 Temmuz: tez feedback loop'u, üç seed ve promotion

Yapılacaklar:

- [ ] Gün 5 FN/FP bucketlerinden en benzer JSON feature ailelerini bul.
- [ ] JSON-derived maskelerin hatalı olanlarını düzelt; bütün düzeltmelerin raw JSON
  provenance ve insan onayı manifestte olsun.
- [ ] WSCAD-dev yalnız hata yönünü gösterir; WSCAD vertex/CP GT gradient görmez.
- [ ] Kilitli en iyi configi seed 0/1/2 ile baştan eğit.
- [ ] Tek global operating point seç; üreticiye/test parçasına özel threshold yasak.
- [ ] Mean/std, worst seed, macro per-part ve per-manufacturer sonuçlarını raporla.
- [ ] Son seed bitince model/config/code/data hashlerini dondur.

Final-test promotion kapısı:

- Üç seed mean WSCAD-dev `Jaccard >= 0.90` ve `F1 >= 0.9474`.
- Worst seed `Jaccard >= 0.88`.
- Her ana üretici/aile `Jaccard >= 0.85`.
- Pooled ve macro/per-part skor birbirinden en fazla 0,05 uzak.
- Precision ve recall ayrı ayrı >=0.90.
- Conflicting-label, leakage ve remesh QA kapıları hâlâ yeşil.

Bu kapılardan biri geçmezse final test açılmaz. Bir haftada %90 başarılmamış kabul
edilir ve gerçek skor/gap raporlanır.

### Gün 7 - 22 Temmuz: production refit ve finali yalnız bir kez aç

Yapılacaklar:

- [ ] Gün 6'da config tamamen kilitlendikten sonra bütün onaylı in-scope Desktop
  JSON'u production refit için kullan. Conflict/out-of-scope dosyaları loss'a sokma.
- [ ] Checkpoint, config, code commit, source manifest, remesh sürümü ve label-adapter
  hashlerini kaydet.
- [ ] Yeni untouched WSCAD final setini ilk ve tek kez aç.
- [ ] Aynı global preprocessing ve operating point ile prediction üret.
- [ ] `cad_eval.py` ile pooled/macro/per-part/per-manufacturer tabloyu üret.
- [ ] Bootstrap/Wilson güven aralığı ve coverage'i ekle.
- [ ] Final başarısızsa threshold/configi final üstünde değiştirme. Bu set artık final
  değildir; sonraki sprintte dev olur ve yeni blind final gerekir.

Nihai başarı kapısı:

- Full-coverage pooled CP `Jaccard >= 0.90`.
- Aynı TP/FP/FN ile CP `F1 >= 0.9474`.
- Precision >=0.90 ve recall >=0.90.
- Macro/per-part F1 >=0.90.
- Ana üretici/aile F1 >=0.85.
- Sonuç human-GT'dir; CAD pseudo-label değildir.
- Tez semantic class Dice/Jaccard, CP lokasyon ve direction metrikleri ayrıca verilir.

Bu şartların hepsi geçmeden sonuç `%90` diye adlandırılmaz.

## 7. Zaman ve hesap bütçesi

| İş | Bütçe |
|---|---:|
| Veri/scope/conflict/GT ve visual QA | haftanın %40'ı |
| Semantic label adapter + remesh QA | %20 |
| Model ve kontrollü ablation | %20 |
| Instance/postprocess/error mining | %15 |
| Final rapor/reproducibility | %5 |

Yanlış head üzerinde 90 epoch daha çalıştırmak yerine önce M0 kapısı geçilecektir.
ASHA başarısız koşuyu erken keser. Her deney bir öncekiyle yalnız tek ana değişkende
farklı olur; split ve benchmark sabit kalır.

## 8. Bir haftanın sonunda üç olası dürüst çıktı

1. **Tam başarı:** untouched human-GT WSCAD finalinde Jaccard >=0.90 ve
   F1 >=0.9474. Yalnız bu durumda iki metrik de %90 denir.
2. **F1-only başarı:** F1 >=0.90 fakat Jaccard <0.90. `%90 F1` denebilir,
   `%90 accuracy` denemez.
3. **Kapı başarısızlığı:** gerçek skor, hata bucket'i ve data/label ceiling raporlanır.
   Özellikle JSON CP noktalarından yüzey maskesi güvenilir üretilemiyorsa yeni insan
   vertex label veya JSON'a hedef-benzeri yüksek kaliteli mesh eklemek gerekir.

Tezin kendi kanıtı ve bugünkü repo skorları nedeniyle üçüncü olasılık halen en
yüksek riskli senaryodur. Planın amacı sahte %90 üretmek değil, yedi gün içinde
%90'ın gerçekten mümkün olup olmadığını en hızlı ve denetlenebilir biçimde sınamaktır.

## 9. Kesinlikle yapılmayacaklar

- [ ] YOLO `mAP@0.5=0.9311` değerini CP F1/accuracy diye kullanma.
- [ ] Housing-dominant klasik vertex accuracy'yi headline yapma.
- [ ] Weighted Jaccard'ı unweighted/macro Jaccard gibi gösterme.
- [ ] WSCAD CAD pseudo-labelini human ground truth sayma.
- [ ] Bridge 9 veya final test üstünde threshold seçme.
- [ ] Remesh sonrası yeniden hesaplanan geometry hash ile split kurma.
- [ ] Aynı geometri/farklı target çatışmasını loss'a bırakıp modelden çözmesini bekleme.
- [ ] Farklı splitli 6k/12k koşularını tek-değişkenli ablation diye kıyaslama.
- [ ] Semantic tez yöntemi yerine yalnız direct heat/offset regressor'u daha uzun eğitme.
- [ ] Selective coverage veya yalnız en iyi seed ile full-coverage %90 ilan etme.
- [ ] Başarıyı garanti etme; final kapısı sonucu belirler.
