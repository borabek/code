# Tanınmayan üreticide robot-hazır F1 = 0.75

> Karar ve yüksek-kaldıraçlı TODO — 2026-08-05 gece denetimi  
> Kapsam: Claude proje geçmişinin 35 JSONL oturumu, 133 Claude memory dosyası,
> workspace'teki ilgili Markdown planları, sonuç makbuzları ve çalışan kod yolu.

## Net hüküm

**0.75; mevcut eşik, gate veya yalnız silindir-ekseni ayarıyla çıkmaz.** Hedef ancak aşağıdaki
iki duvar aynı anda kaldırılırsa gerçekçi olur:

1. **Tespit/assignment:** komşu CP'leri tek adaya birleştirmeden tespit F1'ini yaklaşık `0.90`'a çıkar.
2. **Pose:** tespit edilen CP'lerin en az `%84`'ünü `2 mm + işaretli 10°` koşulunda robot-hazır yap.

Hedef denklem:

```text
robot F1 = tespit F1 × robot-hazır dönüşüm
0.90 × 0.84 = 0.756
```

Bugünkü en önemli karar, yeni bir büyük model kurmak değil; önce ölçüm hattını düzeltmek,
çok-CP adaylarının birleşmesini önlemek, temiz ve morfoloji-hedefli temsili büyütmek, sonra
silindir dışı pose adaylarını ortak bir robot-hazır sıralayıcıya vermektir.

---

## 1. Şu an gerçekten nerede duruyoruz?

| Durum | Tespit F1 | Robot F1 | Hüküm |
|---|---:|---:|---|
| Temiz D6 referans ölçümü | `0.4481` | `0.1479` | 468 parça / 8 üretici; fakat evaluator tam canlı ürün zincirini çalıştırmıyor |
| Gate v5 | `0.5375` | `0.1834` | Olumlu geliştirme sonucu; dağıtılmadı |
| Gate v5 + göreli eşik | `0.5727` | `0.2060` | D6 üzerinde eşik seçildi; artık bağımsız test değil |
| + B-rep snap, tekrar kullanılan yarı | `0.5425` | `0.2330` | Geliştirme sonucu |
| + öğrenilmiş silindir ekseni seçici, aynı yarı | `0.5425` | `0.2676` | Geliştirme sonucu; test yarısı tekrar kullanıldı |
| Full-D6 geriye dönük birleşik aday | `0.5715` | `0.2807` | Ürüne alınmadı; bağımsız final sayı değil |
| **Tam canlı `robot_cp.extract` tabanı** | **ölçülmedi** | **ölçülmedi** | P0 tamamlanmadan yeni manşet yazma |

Önemli ayrımlar:

- Eski `0.6050 / 0.3183` “görülmemiş üretici” sonucu geçersizdir. 250 test parçasının
  169'u segmentasyon verisine; test üreticilerinden 958 parça segmentasyona ve 1.502 parça
  gate korpusuna sızmıştı.
- D6 ilk kurulduğunda temizdi. Gate, eşik, B-rep ve seçici kararlarında tekrar tekrar
  kullanıldığı için **artık DEV'dir**, final/locked değildir.
- `d6_temiz_olc.py` gate sonrası kayıtları doğrudan eşliyor; canlı üründe açık olan pose,
  açı, üye ve yön başlıklarının tamamını `robot_cp.extract` gibi yürütmüyor. Bu nedenle
  `0.1479` temiz bir referanstır ama kesin tam-ürün tabanı değildir.
- Full-D6 eksen seçici ortalamayı yükseltirken ONV'yi yaklaşık `0.4289 → 0.1516` düşürdü;
  high-CP robot F1 yalnız `0.0351` kaldı. Global deploy edilemez.

### D6 üretici dağılımı

MOR 80 · NIT 50 · ONV 8 · S+S 9 · SE 8 · SUPU 165 · UPUN 117 · UTL 31.
Küçük `n`'li üreticiler oynaktır; her sonuçta üretici-makro, medyan, aralık ve güven aralığı
zorunludur.

---

## 2. Tavan denetimi planı nasıl değiştirdi?

Eski rapor `0.9134` aday tavanını her GT için bağımsız en yakın adayı sayarak hesaplıyor.
Aynı aday birden fazla komşu GT'ye kredi alabildiği için bu iyimserdir.

| Tavan | Tümü | Düşük-CP | Yüksek-CP |
|---|---:|---:|---:|
| Eski bağımsız-kapsama hesabı | `0.9134` | `0.9418` | `0.6718` |
| **Tek-eşlemeli aday oracle — denetim yeniden hesabı** | **`0.8806`** | **`0.9290`** | **`0.4685`** |

Tek-eşlemeli hesapta 2.672 GT'nin 323'ü eski yöntemde mükerrer kredi alıyordu. Bu değer
P0'da makbuzlu ve ortak matcher ile yeniden üretilmeden resmî tavan yapılmayacaktır.

Daha da kritik olan: gate v5'in seçtiği mevcut proposal havuzunun kusursuz gate+pose
tek-eşlemeli tavanı yalnız yaklaşık `0.7393`. Yani **mevcut sırayla gate'ten sonra pose
iyileştirerek 0.75'e çıkmak matematiksel olarak mümkün değil.** Daha geniş proposal havuzu
korunmalı, aday ve pose birlikte sıralanmalıdır.

Mevcut p3c pose sözlüğü için ayrıca one-to-one oracle denetimi yapıldı:

| Pose sözlüğü havuzu | Kusursuz seçicili robot oracle | High-CP |
|---|---:|---:|
| Gate v5 seçili havuz | `0.5951` | `0.0590` |
| Geniş gate `0.4/0.2` | `0.6234` | — |
| Tüm 7.736 raw aday | `0.6686` | `0.0915` |

Dolayısıyla p3c seçicisi kusursuz olsa bile hedefe erişemez. **Önce pose seçenek sözlüğü
yeni doğru cevap üretmeli; selector kapasitesi ikinci iştir.**

### 0.75 için gereken dönüşüm

| Tespit F1 | Gereken robot-hazır dönüşüm |
|---:|---:|
| `0.80` | `%93.8` |
| `0.85` | `%88.2` |
| `0.8806` — bugünkü tek-eşlemeli oracle | `%85.2` |
| `0.90` | `%83.3` |
| `0.95` | `%78.9` |

Bugünkü full-D6 geliştirme adayının dönüşümü yaklaşık `%49.1`. Bu yüzden çalışma hedefleri:

- tek-eşlemeli aday oracle: **genel `≥0.96`, high-CP `≥0.85`**;
- seçilmiş tespit: **`≥0.90`**;
- pose-seçenek oracle dönüşümü: **`≥0.92`**;
- gerçek robot-hazır dönüşüm: **`≥0.84`**.

---

## 3. Değişmeyecek ölçüm sözleşmesi

- **Robot metriği:** yanal `≤2 mm`, işaretli yön açısı `≤10°`, eksenel `≤40 mm`.
- **Tespit metriği:** mevcut `max(3 mm, bbox diyagonalinin %6'sı)` yanal koşulu ve `≤40 mm`
  eksenel koşul; açı serbest.
- Toleransı gevşeterek 0.75 yazmak yasaktır. Fiziksel ürün spesifikasyonu değişirse ayrı bir
  ikincil operating point olarak raporlanır.
- Tek, deterministik, one-to-one matcher kullanılacak. Greedy/Hungarian ayrılığı kapatılacak.
- Her deney şu tabloyu çıkaracak: rejim-ağırlıklı P/R/F1, ham micro P/R/F1,
  üretici-makro/medyan/worst, üretici başına P/R/F1 ve `1–3 / 4–7 / 8+ CP` kırılımı.
- Candidate presence, one-to-one assignment, lateral, angle ve sign oracle'ları **ortak ve
  kümülatif** ölçülecek; bağımsız tavanlar toplanmayacak.
- Artifact hash, config, checkpoint, veri mührü, seed, süre, RAM/VRAM ve fallback sayısı makbuza
  yazılacak. Sessiz fallback başarı sayılmayacak.

### Final kabul kapısı

Yeni bir **D7** kümesi, hiçbir etiket/model/eşik/branch kararında kullanılmamış üreticilerden
kurulacak. Model ve matcher dondurulduktan sonra tek atış ölçülecek.

- D7 rejim-ağırlıklı robot F1 `≥0.75`.
- Manufacturer-block bootstrap `%95` alt sınırı `≥0.70`.
- Ağırlıklı precision ve recall ayrı ayrı `≥0.70`.
- Düşük-CP robot F1 `≥0.80`, yüksek-CP robot F1 `≥0.50`;
  bu ikisi mevcut `0.895/0.105` ağırlıkla `0.7685` verir.
- En az 20 parçalı hiçbir üreticide robot F1 `<0.50` olmayacak.
- Üç seed veya önceden dondurulmuş ensemble; final sonucu gördükten sonra seçim yok.
- D7 sonucuna göre kod/eşik/veri değiştirilirse D7 anında DEV olur ve final için D8 gerekir.

---

## 4. Önceliklendirilmiş TODO

Her sırada önce ucuz oracle/kanıt, sonra pahalı eğitim vardır. Bir kol GO kapısını geçmeden
bir sonraki pahalı aşamaya kaynak ayrılmaz.

### P0 — Ölçümü ve final sınavını mühürle

**Kaldıraç:** Dolaylı ama zorunlu · **Maliyet:** düşük/orta · **Durum:** hemen

- [ ] STEP'ten başlayıp gerçek `robot_cp.extract` yolunu çağıran tek canonical evaluator yaz.
- [ ] Cache evaluator ile canlı evaluator arasında aday, gate, pose, açı, üye ve yön aşamalarını
  tek tek karşılaştıran parity testi ekle.
- [ ] `sina_kume.py` greedy matcher ile `metrics.py` one-to-one/Hungarian ayrılığını kapat.
- [ ] Eski bağımsız-kapsama tavanını bırak; maximum bipartite one-to-one detection ve
  robot-pose oracle makbuzlarını üret.
- [ ] Her aşamadaki proposal sayısını ve hangi nedenle düştüğünü kaydet:
  segmentasyon → component → vote merge → class dedupe → gate → assignment → pose.
- [ ] D6'yı `DEV` olarak yeniden etiketle. D7 için yeni üretici/parça edinme listesini çıkar;
  üretici, geometri ikizi ve yakın katalog varyantı sızıntısını engelle.
- [ ] D7 yoksa manufacturer-outer CV kullan; bunun final tek-atış kanıtı olmadığını açıkça yaz.

**Çıkış koşulu:** aynı frozen artifact iki evaluator'da aynı sonucu verir; one-to-one tavanlar,
hash'ler ve D7 veri sözleşmesi kayıtlıdır. Bundan önce hiçbir sayı “ürün F1” diye yayımlanmaz.

### P1 — Çok-CP'yi ezen merge/dedupe yarıçaplarını düzelt

**Kaldıraç:** çok yüksek · **Maliyet:** düşük · **Teze sadık:** evet

Kodda üç yarıçap komşu gerçek giriş aralığıyla çakışıyor:

- vote pooling: `5 mm`;
- component cluster: `3 mm` single-linkage;
- CableEntry ↔ CableTermination dedupe: hard-coded `10 mm`.

Kutup pitch'i yaklaşık `3.5–6 mm` olduğunda iki gerçek CP tek adaya çökebilir. High-CP'de
bağımsız coverage `0.6718` iken tek-eşlemeli tavanın `0.4685` olması bunun güçlü imzasıdır.

- [ ] Vote radius için `{0,1,2,3,5}` mm; cluster için `{0,1,2,3}` mm; class-dedupe için
  `{0,1,2,3,10}` mm kontrollü ablasyon yap.
- [ ] Her aday dağılımında gate'i yeniden fit et; eski gate'i yeni pool üzerinde yargılama.
- [ ] Duplicate FP, split TP, candidate count, latency ve one-to-one oracle'ı birlikte raporla.
- [ ] Sabit global yarıçap yerine yalnız kanıt gelirse mesh çözünürlüğü/yerel açıklık çapına bağlı
  ölçekli bir yarıçap dene.

Salt `_vote2: 5→0 mm` denetimi şimdiden genel detection oracle'da `0.8806→0.8926`,
high-CP'de `0.4685→0.4888` ve strict-pose oracle'da yaklaşık `+0.062` alan açtı. Bu tek başına
hedef değildir ama mekanizmanın gerçek olduğunu doğrular.

**GO:** high-CP one-to-one candidate oracle `+≥0.015`, genel oracle `+≥0.010`, low-CP gerçek
F1 `≥−0.005` ve genel görülmemiş-üretici tespit `+≥0.010`. **KILL:** high-CP oracle açılmıyor veya FP maliyeti net
robot F1'i götürüyorsa. Bu grid yalnız D6/outer-DEV'de seçilir.

### P2 — Çalışan temiz ağız-etiket üretimini tamamla ve doğrula

**Kaldıraç:** yüksek · **Maliyet:** mevcut compute zaten harcanıyor · **Tez omurgası:** korunur

Bu dosya yazılırken `_label_auto` içinde **2.022 NPZ** vardı ve üretim süreçleri çalışıyordu.
Kirli 958 etiket `_label_auto_KIRLI` altında ayrı tutuluyor; karıştırılmayacak.

- [ ] Çalışan süreçleri kesme; shard'ların tamamlanmasını bekle.
- [ ] Ortak `results/g5_agiz_etiket.json` dosyasının shard'larca ezilme riskine karşı gerçek NPZ
  envanterinden tek birleşik receipt üret.
- [ ] D5/D6/D7 üreticileri, yasak parça kimlikleri ve geometri ikizleri için fail-hard sızıntı testi.
- [ ] Mevcut `%60` pseudo-label QC'si bağımsız nearest saydığı için onu one-to-one coverage,
  yanal hata ve yön kalitesiyle değiştir.
- [ ] Üretici/morfoloji/CP yoğunluğuna göre tabakalı en az 100 parçayı insan gözüyle denetle;
  yalnız sayı büyüdü diye eğitime sokma.
- [ ] Aynı DiffusionNet, aynı 5 sınıf, aynı `v_o` mantığı ve masked partial loss ile üç seed eğit.
- [ ] Yeni candidate dağılımında gate ve sonraki tüm aşamaları sıfırdan yeniden fit et.

**GO:** clean manufacturer-outer one-to-one oracle genel `+≥0.03`, high-CP `+≥0.05` ve
refit sonrası tespit `+≥0.02`; low-CP kayıp `≤0.005`. **KILL/karantina:** QC, sızıntı veya
outer-manufacturer genellemesi geçmezse.

Not: Mimari tez sadakati korunur. Otomatik kısmi etiket, orijinal tez deneyinin birebir tekrarı
değil; dürüst adı **“tez-omurgalı ürün genişletmesi”** olmalıdır.

### P3 — Rastgele veri değil, hedefli aktif etiketleme

**Kaldıraç:** yüksek · **Maliyet:** kontrollü insan saati · **Teze sadık:** evet

SE GT'lerinin yaklaşık `%95`'inde kullanılabilir silindir yok; NIT'in `%70`'inde görülen
silindirler giriş eksenine dik. Rastgele TOGI hacmi bu bilgi açığını kapatmaz.

- [ ] Train üreticilerindeki FN/yanlış-pose parçalarını geometri ve DiffusionNet embedding'iyle
  kümelendir.
- [ ] Square/spring-cage, non-round mouth, slot/push-in, yüksek-CP ve NIT/SE-benzeri kümeleri
  çeşitlilik gözeterek seç.
- [ ] Pseudo-label'i ön-doldur, insan yalnız hatalı 5-sınıf bölgeleri ve ağız component'ini
  düzeltsin. Aynı geometri ailesinden kopya label satın alma.
- [ ] Önce 50, sonra en fazla 100 parçalık partilerle öğrenme eğrisi çıkar.
- [ ] Etiket saati başına candidate-oracle ve robot F1 kazancını kaydet.

**GO:** bir parti manufacturer-outer robot F1'de `+≥0.015` veya one-to-one oracle'da
`+≥0.025` getiriyorsa devam. **STOP:** iki ardışık parti bu sınırların altında kalırsa.

### P4 — Silindir dışı pose sözlüğü kur; önce oracle ölç

**Kaldıraç:** çok yüksek · **Maliyet:** orta · **Teze sadık:** fiziksel/geometrik post-process

Her tespit için tek eksen dayatmak yerine küçük bir fiziksel pose seçenek sözlüğü üret:

1. mevcut tez yönü: `v_o − v_s` ve bbox hizalama;
2. güvenilir analitik silindir ekseni;
3. CableEntry mouth-boundary loop düzleminin normali;
4. karşılıklı planar yüz/slot ekseni;
5. lokal boşluk/free-space ray veya channel/corridor yönü;
6. mevcut üye/yön başlıklarının adayları;
7. **düzeltme yok** seçeneği.

- [ ] Önce her kaynağın kapsama, unique-oracle katkısı, yanal ve işaretli açı dağılımını çıkar.
- [ ] Konum sözlüğü × yön sözlüğü Cartesian product'unu kur; doğru yönü yanlış konuma veya
  doğru konumu yanlış yöne kilitleme.
- [ ] Silindir yüzeyi dışında B-rep circular edge loop, rectangular boundary loop ve paired-plane
  seçeneklerini dahil et.
- [ ] Silindir branch'ini MOR/UTL/ONV-benzeri geometride uzman olarak tut; her parçaya uygulama.
- [ ] NIT/SE için boundary/planar/free-space branch'lerini ayrı ölç.
- [ ] Her düzelticiye güven skoru ve abstention/fallback ekle.
- [ ] Mevcut p3c etiketlemesini düzelt: önce ham konuma greedy GT atayıp bütün pose seçeneklerini
  aynı GT'ye etiketleme. Pose seçenekleri üretildikten sonra robot maliyetiyle one-to-one eşle.
- [ ] Dünya XYZ'sine bağlı `max(abs(axis))`, mutlak çap/uzunluk ve ham aday sayısı gibi üretici
  CAD-frame imzalarını ablate et; OBB/B-rep-relative, bbox-normalized ve part-içi rank kullan.

**İlk eğitim kapısı:** raw one-to-one pose-dictionary robot oracle `≥0.82`, high-CP `≥0.35`.
Bu geçmeden yeni selector eğitme. **Nihai GO:** birleşik pose-option dönüşüm oracle'ı `≥0.92`
veya `D=0.90` altında robot potansiyeli `≥0.80`; yeni bir branch full oracle'da `+≥0.03`
veya high-CP'de `+≥0.05` verir. **KILL:** yalnız mevcut
adayları farklı adlandırıp oracle açmayan branch.

### P5 — Geniş proposal havuzu → ortak robot-hazır ranker → tek-eşlemeli assignment

**Kaldıraç:** hedefe giden ana algoritmik kol · **Maliyet:** orta · **Önkoşul:** P1/P2/P4 oracle

Mevcut sıra `gate → p3c pose` olduğu için gate, doğru pose'a sahip alternatifi daha pose görülmeden
atabiliyor. Yeni sıra:

```text
geniş proposal pool → candidate × pose seçenekleri → robot-hazır skor → conflict/assignment
```

- [ ] Raw veya geniş (`0.4/0.2` benzeri) proposal pool'u koru; threshold'u final karar yapma.
- [ ] Hedefi loose-detection değil doğrudan `2 mm + işaretli 10°` robot-hazır etiketi yap.
- [ ] Üretici kimliğini özellik olarak verme. Geometri, segmentasyon güveni, component,
  B-rep, free-space ve pose-tutarlılık özelliklerini kullan.
- [ ] Manufacturer-outer OOF eğitim; inner fold'da eşik. Ortalama yanında worst-manufacturer
  ve high-CP'yi optimize et.
- [ ] Bir proposal'ın birden fazla GT'yi kazanmasını engelleyen conflict graph / bipartite
  assignment kullan.
- [ ] Mevcut cylinder selector'ı bir uzman olarak içeri al; düşük güvende eski pose'a dön.

Eski SetNet/listwise kolu eski scalar/detection candidate uzayında başarısızdı. Bu madde onu
aynen yeniden açmaz; yalnız P4'ün **yeni ve ölçülmüş pose-option oracle bilgisi** varsa doğrudan
robot hedefli küçük bir ranker kurulmasına izin verir.

**Ara kapılar:**

- A: fresh outer üreticide `tespit ≥0.70`, `robot ≥0.35`;
- B: `tespit ≥0.80`, `robot ≥0.55`;
- C: `tespit ≥0.90`, dönüşüm `≥0.84`, `robot ≥0.75`.

**Deploy GO:** en az 6/8 DEV üreticide pozitif, worst-manufacturer'da anlamlı gerileme yok,
high-CP robot `+≥0.03`, üç seed aynı yönde. Ortalama yükselip ONV-benzeri bir domain çökerse
deploy yok.

### P6 — Yalnız iz kanıt gösterirse diğer tavan/limitleri aç

**Kaldıraç:** koşullu · **Maliyet:** orta/yüksek

- [ ] `MM_MAX=8`, B-rep aday sayısı, arama menzili ve top-k sınırlarını trace et. Doğru seçenek
  cap yüzünden kesiliyorsa artır; aksi halde dokunma.
- [ ] Pseudo-label `EKSEN_ONCE=1`, `EKSEN_SONRA=6`, radius payı `1.15` sınırlarını yalnız
  train/inner-DEV'de kalibre et.
- [ ] P1 sonrası high-CP candidate oracle hâlâ düşükse yaklaşık 6k uniform mesh'e karşı
  9k/12k **uniform** vertex pilotu yap. Tezin graph-instance çözünürlüğü önerisiyle uyumludur,
  fakat compute bedeli ayrıca raporlanır.
- [ ] Eski adaptive-density A1 deneyi negatiftir. Aynı deneyi yeniden açma; yalnız yeni trace
  gerçek component kaybının örnekleme yoğunluğünden geldiğini kanıtlarsa farklı, ön-yazılı bir
  high-CP pilotuna izin ver.
- [ ] Ensemble'ı en sona bırak; yalnız manufacturer-out candidate recall ve robot F1'i üç seed'de
  artırırsa tut.

**GO:** ilgili cap/resolution değişimi one-to-one oracle'ı genel `+≥0.02` veya high-CP
`+≥0.05` açar ve uçtan uca kazanç compute maliyetine değer. Kör limit büyütme yok.

### P7 — 0.75 gelmeden de para kazandıran güvenli ürün yolu

Araştırma metriğiyle ürün çalışma kipini karıştırma:

- [ ] **AUTO:** kalibre edilmiş precision `≥0.95`; yalnız yüksek güvenli CP'ler robotta otomatik.
- [ ] **REVIEW:** düşük güven, non-cylinder veya yoğun parçalar operatöre açıklanabilir adaylarla
  gider; düzeltmeleri aktif öğrenme kuyruğuna döner.
- [ ] AUTO coverage, yanlış-pozitif maliyeti, operatör başına kazanılan dakika, latency ve
  başarısız STEP oranını ayrı KPI yap.
- [ ] WSCAD/catalog expected-count varsa yalnız ayrı **metadata-assisted** ürün kipinde dene.
  STEP-only `0.75` manşetine karıştırma; temiz DEV kazancı `<0.02` ise kapat.
- [ ] Fiziksel çakışma/erişilebilirlik güvenlik metriğini F1'den ayrı tut. Güvenlik filtresi F1'i
  düşürse bile ürün kararı olarak raporla.

---

## 5. Artırılacak tavanlar; değiştirilmeyecek metrikler

| Nesne | Bugünkü işaret | Hedef | Nasıl |
|---|---:|---:|---|
| One-to-one candidate oracle | `0.8806` | `≥0.96` | P1 merge/dedupe + P2/P3 temsil |
| High-CP candidate oracle | `0.4685` | `≥0.85` | komşu CP koruma + gerekirse vertex çözünürlüğü |
| Gate sonrası seçili-pool oracle | `~0.7393` | `≥0.93` | geniş proposal + ortak ranker |
| Seçilmiş tespit F1 | `~0.57` geliştirme | `≥0.90` | temsil + ranker + assignment |
| Gerçek pose dönüşümü | `~%49` geliştirme | `≥%84` | non-cylinder pose bank + güvenli seçici |
| Pose-dictionary robot oracle | raw `0.6686`; high `0.0915` | ilk kapı `≥0.82/0.35`, nihai dönüşüm `≥%92` | cylinder + boundary + planar + free-space |
| Robot toleransı | `2 mm / 10°` | **aynı** | değiştirme |

---

## 6. Yeniden açılmayacak ölü kollar

Yeni bir bilgi kaynağı ve önceden yazılmış oracle gerekçesi olmadıkça:

- [x] `2 mm / 10°` toleransını gevşetmek veya metriği yeniden ağırlıklandırmak.
- [x] Eski kirli D5, pooled/tanıdık veya D6-tuned sayıları final unseen sonucu diye kullanmak.
- [x] İşaret flip'i: kusursuz işaret oracle kazancı yalnız yaklaşık `+0.0084`.
- [x] Parça-içi baskın eksen uzlaşısı: robot yaklaşık `0.2089 → 0.1773`.
- [x] En yakın silindire herkesi global snap etmek; güven/uzmanlık olmadan p3c deploy etmek.
- [x] Eski gate uzayında kapasite, manufacturer weighting, twin weighting, pruning ve tekrar
  tekrar threshold taramak.
- [x] Threshold'u kör düşürmek; geniş proposal ancak sonra ortak ranker varsa anlamlıdır.
- [x] Saf CAD/geometri dedektörü; eski scalar topoloji/lattice/grid özellikleri.
- [x] Eski candidate uzayında SetNet/listwise deneyi.
- [x] Genel count prior/expected-count'u STEP-only tabana karıştırmak.
- [x] `k_eig=128`; eski adaptive-density A1; `r=2 mm` high-CP disk pseudo-label.
- [x] Rastgele veya tek baskın üreticiden toplu etiket eklemek; çeşitlilik ve hata morfolojisi
  seçilmeden insan saati harcamak.
- [x] Yalnız aday-seviyesi/AUC artışını uçtan uca kazanç saymak.

---

## 7. Kaynak kullanım freni

- Oracle açılmadan yeni ranker için bir GPU-gece harcama.
- Birleştirme/dedupe izi ölçülmeden mesh/model büyütme.
- İlk 50/100 hedefli etiketin marjinal kazancı ölçülmeden büyük manuel etiket bütçesi açma.
- Yeni candidate pool'da gate'i refit etmeden eski gate ile kol öldürme.
- D7 mühürlenmeden “final test” çalıştırma; sonucu gördükten sonra tuning yapma.

Önerilen yürütme sırası:

```text
P0 ölçüm + D7
    ├── P1 merge/dedupe
    └── P2 temiz G5
          ├── P3 hedefli etiket
          └── P4 pose-option oracle
                    ↓
             P5 ortak ranker
                    ↓
          P6 kanıtlı limit artışı
                    ↓
              D7 tek atış
```

---

## 8. Kanıt haritası

Ana yerel kaynaklar:

- Temiz durum ve eski tavan: [`DURUM_2026-08-05.md`](DURUM_2026-08-05.md)
- Temiz sınav: [`results/d6_sinav_kumesi.json`](results/d6_sinav_kumesi.json)
- Referans ölçüm: [`results/d6_temiz_olc.json`](results/d6_temiz_olc.json)
- Gate/eşik/B-rep/seçici: [`results/p1_gate_v5.json`](results/p1_gate_v5.json),
  [`results/p1c_esik.json`](results/p1c_esik.json),
  [`results/p3_brep_oturt.json`](results/p3_brep_oturt.json),
  [`results/p3c_eksen_secici.json`](results/p3c_eksen_secici.json)
- Eski oracle/otopsi: [`results/d6_tavan_merdiveni.json`](results/d6_tavan_merdiveni.json),
  [`results/d6_otopsi.json`](results/d6_otopsi.json),
  [`results/d6_isaret_odulu.json`](results/d6_isaret_odulu.json)
- Evaluator ve matcher: [`d6_temiz_olc.py`](d6_temiz_olc.py),
  [`sina_kume.py`](sina_kume.py), [`metrics.py`](metrics.py)
- Canlı ürün hattı: [`robot_cp.py`](robot_cp.py), [`cp_config.json`](cp_config.json)
- Temiz pseudo-label hattı: [`g5_agiz_etiket.py`](g5_agiz_etiket.py)
- Tez omurgası: [`THESIS_FINAL.md`](THESIS_FINAL.md),
  [`Masterarbeit_Scheffler.md`](Masterarbeit_Scheffler.md)

Claude geçmişinde kronolojik tek kaynak:

`C:\Users\DE00024082\.claude\projects\c--Users-DE00024082-Desktop-code\03658826-77fa-4d1b-a970-7ee8ae02d4db.jsonl`

Bayat/yanıltıcı manşetler için özellikle `DURUM_R.md`, `PRODUCT_MODEL.md`,
`OPERASYON_ROBOT_080.md`, `ROTA_PLANI.md`, `PLAN_URETICI_070.md`, `CP_F1_SCOREBOARD.md`
ve `FINAL_STATUS.md` güncel kanıt yerine kullanılmamalıdır.
