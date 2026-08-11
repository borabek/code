# ADLİ TIP RAPORU — Kaldıraçlar Neden Seri Halde Ölüyor?
**2026-07-28 · Vaka: CP-F1 darboğazı (0.75'te kilitli) · İncelenen: ~20 kaldıraç ölümü, 3 hayatta kalan**

> Bu rapor F1'i artırma planı DEĞİL. Soru şu: neden denediğimiz hemen her şey ölüyor?
> Ortak ölüm nedenleri neler, ve o SİSTEMİK nedenler düzeltilirse ne değişir?

---

## 1. ÖLÜM ŞEKLİ SINIFLANDIRMASI (~20 vaka)

| Ölüm şekli | Vakalar | Tipik bulgu |
|---|---|---|
| **A. Fazlalık (redundancy)** — "yeni" sinyal mevcutla korelasyonlu | eksen-kümeleme (+0.0036 AUC), skor-havuzlama (+0.002 F1), bağlam feat. (+0.003), embedding (+0.008), per-mfg eşik (±0.01) | hipotez DOĞRU çıkıyor ama model zaten biliyor |
| **B. Bilgi yokluğu** — aranan sinyal veride hiç yok | huni/koni (B-rep AUC 0.504 + mesh negatif), FP-arketip (0 saf küme), insan-adjudication ekonomisi | sinyal her ölçekte yok, etiket de üretmez |
| **C. Gürültü tabanı / transfer başarısızlığı** — WORK'te kazanan, holdout'ta kaybolan | SetNet (+0.016 WORK → 0.000 holdout), listwise, aile-out fold farkları | n=541'de ±0.015 çözülemiyor |
| **D. Seyrelme** — recall kazancı precision kaybından yavaş | agresif aday (tavan 0.965 ama F1 0.697), min_v düşürme | gate 6.5x havuzu taşıyamıyor |
| **E. Yanlış-teşhis karışımı** — iki ayrı hastalık tek istatistikte | high-CP (tavan 0.370, ÇÖZÜNÜRLÜK açlığı) tipiklerle (tavan 0.900) aynı ALL'da | kaldıraçlar "ölü" görünür çünkü yanlış alt-nüfusta test edilir |
| **F. Süreç boğulması** — kısıtların bileşkesi arama uzayını kapatıyor | 2D-eye (tez-sadakat), etiket turu (plan kapsamı), rank-ensemble (deploy kuralı) | tek tek makul, bileşke: sadece "aynı damarı yeniden kaz" kalıyor |

**Hayatta kalanlar (kontrol grubu):** zengin temsil (+0.020, holdout-doğrulamalı), k_eig 96, izotropik remesh, union+gate.
**Adli imza:** hayatta kalanların HEPSİ *gerçekten yeni bilgi enjeksiyonu ya da temsil-uyumu düzeltmesi*.
Ölenlerin HEPSİ *mevcut bilginin yeniden karılması*. **Damar tükenmiş; biz kazmayı suçluyorduk.**

---

---

# ⚠️ REVİZYON 1 (aynı gün, rapor sonrası ölçümler) — İKİ TEŞHİS ÇÜRÜDÜ

Raporu yazdıktan sonra R1'in ve "Bayes hatası" tezinin öncüllerini test ettim. **İkisi de yanlış çıktı.**
Yanlış teşhisi raporda bırakmamak için burada düzeltiyorum.

### ÇÜRÜYEN 1 — "R1: backbone kör" YANLIŞ
family-out AUC ile ölçüldü:
| kaynak | AUC |
|---|---|
| **sadece backbone embedding'leri** | **0.9272** |
| sadece 13 el-yapımı feature | 0.8997 |
| zengin 46d | 0.9355 |
Backbone, birleştirilmiş `Contact` etiketiyle eğitilmiş olmasına rağmen wire/tool ayrımını **taşıyor** ve
el-yapımı feature'ları **geçiyor**. "Model ayrımı silmek üzere eğitildi" tezi düştü → wire-farkındalıklı
retrain (Tedavi-3) beklenen değerini büyük ölçüde kaybetti, **öncelik listesinden düşürüldü.**

### ÇÜRÜYEN 2 — "kalan hata indirgenemez / FP≡TP / bilgi geometride yok" YANLIŞ
Tüm bilgi kaynakları birleşik uzayda, farklı parçalardan gelen **neredeyse-özdeş aday çiftleri**:
| komşu mesafesi | zıt etiketli |
|---|---|
| en yakın %5 | **0/137 = %0.0** |
| en yakın %10 | **0/267 = %0.0** |
| en yakın %20 | 8/591 = %1.4 |
Bayes hatası pratikte **yok**. Neredeyse-özdeş adaylar aynı etikete sahip → **sınıflar ayrılabilir.**
Gecelerdir tekrarladığım "geometride bilgi yok" sonucu **hatalıydı**; doğrusu: sınır VAR, biz onu
**öğrenemiyoruz**.

### YENİ ANA TEŞHİS — VERİ AÇLIĞI (R2'nin ağır formu)
Öğrenme eğrisi (family-out) **düzleşmiyor**:
| eğitim ailesi | AUC | F1 | adım kazancı |
|---|---|---|---|
| 25% (119) | 0.8892 | 0.7029 | — |
| 50% (239) | 0.9143 | 0.7280 | +0.025 |
| 75% (359) | 0.9284 | 0.7428 | +0.014 |
| **100% (479)** | **0.9381** | **0.7516** | **+0.010 — hâlâ artıyor** |
Katlama başına ≈ **+0.024 F1**. Tavan yok. **Kaldıraçlar öldü çünkü hepsi ~500 aileyle doymuş bir
modele "daha iyi feature" vermeye çalışıyordu; eksik olan feature değil, ÖRNEK.**

**Kullanılmamış veri (elimizde, etiketli, bedava):** Desktop/JSON'da **386 ABB parçası = 11,228 üretici CP**
(mevcut 1888 GT'nin **6 katı**), ortalama **29 CP/parça** — yani tam da modelin çöktüğü high-CP rejimi.
JSON mesh'i remesh edilebiliyor ve **CP'ler aynı frame'de** → hizalama hatası bile ortadan kalkıyor.

**ENGEL (ölçüldü):** ABB'de mevcut ayarla aday-tavan recall sadece **0.107** — 6000 vertex koca gövdeye
yayılınca tel ağızları `min_v=30`'a takılıyor, aday HİÇ oluşmuyor. Yani **R5 (çözünürlük açlığı), veri
kolunun ön koşulu.** İkisi aynı kapı: **alan-ölçekli vertex bütçesi.**

**Revize edilmiş ana hat:** çözünürlüğü düzelt → ABB verisini aç → gate'i 7× veriyle eğit.
Öğrenme eğrisi ekstrapolasyonu (≈2.8 katlama × +0.024): **0.75 → ~0.82** aralığı; 0.85 için bunun üstüne
metadata (R4) meşrulaşması gerekir. Bu, gece boyunca ölçülen ilk **büyük ve mekanizması kanıtlı** kaldıraç.

---

## 2. KÖK PATOLOJİLER (sistemik — kaldıraçları öldüren asıl hastalıklar)
> Revizyon 1 ışığında R1'in gücü düştü, R2 ana teşhis oldu, R5 ön-koşul mertebesine yükseldi.

### R1 — ETİKET UZAYI HASTALIĞI (en derin)
Tezin `Contact` sınıfı **tasarımı gereği** "Kontaktierung *bzw.* Werkzeugeinschub" — tel ve alet TEK sınıf.
Segmentasyon modeli bu etiketlerle eğitildi → **backbone, bizim ayırmak istediğimiz iki şeyi BİRLEŞTİRMEK
üzere eğitildi.** Sonra downstream gate'ten, feature-çıkarıcının silmek için eğitildiği ayrımı geri
kazanmasını istiyoruz. Fazlalık-ölümlerinin (A sınıfı) kökü bu: gate'e ne eklersek ekleyelim, hepsi aynı
kör backbone'dan türetiliyor. Embedding'in +0.008'de kalması bunun kanıtı — modelin KENDİ temsili de kör.
**Düzeltme:** wire-farkındalıklı yardımcı süpervizyonla backbone'u yeniden eğitmek (üretici-eşleşmeli
CE/CT bileşenleri = wire; eşleşmeyenler = tool adayı → segmentasyon eğitimine auxiliary loss).
Kullanıcının Faz-2 listesi buna izin veriyor ("önce candidate auxiliary head"). **DENENMEMİŞ tek yapısal
ML düzeltmesi budur.** Maliyet: 1 GPU günü. Risk: orta (etiket türetme gürültüsü).

### R2 — ÖLÇÜM ÇÖZÜNÜRLÜĞÜ HASTALIĞI
541 parça / 1888 GT ile family-out gürültüsü ±0.015; holdout CI genişliği **±0.06**. Test ettiğimiz
kaldıraçların vaadi +0.01–0.03 = **gürültü tabanının altında/dibinde.** Bu yüzden: (a) iyi kaldıraçlar
ölü görünebilir, (b) kötüler canlı görünüp holdout'ta buharlaşır (SetNet). Aparat, ölçmeye çalıştığı
etkiyi çözemiyor.
**Düzeltme:** değerlendirme kümesini büyüt (ABB 386 parça zero-shot kolu; JSON korpusunun kalanı),
her kararda 3-seed + CI zorunlu, kaldıraç başına ön-kayıtlı eşik. F1'i artırmaz ama **yanlış idam ve
yanlış beraatları durdurur.**

### R3 — TEK GÖZ HASTALIĞI
Aday üretici, gate feature'ları, embedding, güven skoru — HEPSİ aynı DiffusionNet'ten. Sistemin tek gözü
var; "çeşitlilik" diye eklediklerimiz aynı gözün farklı gözlükleri (multires dahil). Fazlalık-ölümlerinin
ikinci kökü. İkinci göz adayları (B-rep step_openings, geo-detektör) zayıf ölçüldü ve füzyonda öldü —
ama **aday-recall seviyesinde** (gate değil) sistematik füzyon hiç tam denenmedi.
**Düzeltme (düşük güven):** B-rep adaylarını sadece "6k'nın hiç aday bulamadığı bölgelerde" ekle
(recall-tamamlayıcı, seyreltme-korumalı).

### R4 — GÖREV TANIMI HASTALIĞI (bilgi-yokluğunun kaynağı)
Benchmark "SADECE tel girişleri" istiyor; bu ayrım kısmen **katalog bilgisi** (parçanın işlevi), geometri
değil. Üretici bunu datasheet'ten bilir; bizim girdimiz sadece geometri. B-sınıfı ölümlerin kökü.
**Düzeltme (SIFIR hesaplama maliyeti — kullanıcı işyerinde 30 dk):** WSCAD/şirket artikel veritabanında
**Anschlusszahl / kutup sayısı / giriş yönü alanı var mı** kontrol et. P5 STEP-PRODUCT'ta bulamadı ama
şirket DB'sinde olabilir. VARSA: metadata modu (0.775) "varsayımsal" etiketinden kurtulur, yön-prior
yasağı kalkar, count-assisted +0.02–0.03 MEŞRU olur. Bu, listedeki en ucuz gerçek kazanç.

### R5 — KARIŞIM HASTALIĞI (high-CP)
ALL istatistiği iki ayrı hastalığı karıştırıyor: tipik parçalar (ayrım-sınırlı, tavan 0.90, F1 0.825)
ve high-CP parçalar (ÇÖZÜNÜRLÜK-sınırlı, tavan 0.370, F1 0.458). Mekanik neden: 6000-vertex bütçesi
200mm'lik marshalling gövdesine yayılınca kenar ~2–3mm → 1.5–2mm'lik tel ağzı 10–20 vertexe düşer →
min_v=30 filtresi bileşeni siler → açıklık HİÇ aday olamaz. Bu bilgi-yokluğu DEĞİL, mühendislik açlığı.
**Düzeltme:** (a) kuyruktaki hibrit (agresif havuz sadece high-CP'ye) — kullanıcı işe geçince koşacak;
(b) kalıcısı: **alan-ölçekli vertex bütçesi** (sabit ~0.7mm kenar hedefi) + min_v'nin yoğunlukla
ölçeklenmesi. Beklenti: ALL +0.02–0.03 ve robotun bilinen OOD çöküşünün onarımı.

### R6 — SÜREÇ BOĞULMASI (benim de payım var)
Donmuş ürün + tez-sadakat + etiket-yasağı + deploy-kuralı: her biri tek tek doğru karar; bileşkesi,
arama uzayını "downstream'de mevcut bilgiyi yeniden karıştır"a indirgedi — tam da fazlalık-ölümlerinin
yaşandığı bölge. Süreç, tükenmiş damarı kazmayı GARANTİLEDİ.
**Düzeltme:** kısıt envanterini kullanıcıyla açıkça yeniden müzakere et (hangileri kalkabilir: 2D yardımcı
göz? hedefli etiket? tez-sadakat sadece raporlama düzeyinde mi kalsın?). Bu teknik değil, karar işi.

### R7 — SKORBOARD BORCU (büyük ölçüde ödendi)
BA_ALLOW_SEEN şişkinliği (−0.027), GT-türevli metadata, boş REVIEW katmanı, bayat MEMORY sayıları —
her biri kaldıraç triyajını saptırdı (erken "büyük lever" yanılsamaları buradandı). P0 bunları temizledi.
**Kalan iş:** tek skorboard (`split_lock.json` + kilitli protokol) dışında sayı ÜRETMEMEK; REVIEW
eşiğini yeniden kalibre etmek.

---

## 3. TEDAVİ PLANI — önceliklendirilmiş (beklenen kazanç × olasılık ÷ maliyet)

| # | Tedavi | Maliyet | Dürüst beklenti |
|---|---|---|---|
| 1 | **R5-a: high-CP hibrit** (kuyrukta, kullanıcı başlatacak) | 40 dk GPU | ALL 0.775→~0.79 (WORK) |
| 2 | **R4: WSCAD DB'de Anschlusszahl kontrolü** (kullanıcı, işyerinde) | 30 dk, 0 hesap | count meşrulaşırsa +0.02–0.03 |
| 3 | **R1: wire-farkındalıklı backbone retrain** (auxiliary loss) | 1 GPU günü | tek yapısal umut; oracle-boşluğun (0.89−0.78) bir kısmı; aralık +0.00–0.05, GERÇEKTEN belirsiz |
| 4 | **R5-b: alan-ölçekli remesh** (kalıcı high-CP çözümü) | yarım gün | R5-a'yı ürünleştirir |
| 5 | **R2: eval büyütme + CI disiplini** | yarım gün | sayı artırmaz, kararları güvenilir kılar |
| 6 | **R6: kısıt müzakeresi** (kullanıcı kararı) | toplantı | uzayı yeniden açar |
| 7 | R3: B-rep recall-tamamlayıcı füzyon | 1 gün | düşük güven, +0.00–0.01 |

**Bileşik dürüst projeksiyon:** 1+2+3 hepsi tutarsa 0.748 → **~0.80–0.83** (kilitli protokolde).
0.85, ancak R4 gerçek çıkarsa VE R1 üst aralığında çalışırsa erişilir — **mümkün ama söz verilemez.**

## 4. TEK CÜMLELİK OTOPSİ SONUCU
> Kaldıraçlar tek tek öldürülmedi; **hepsi aynı beş sistemik hastalığın komplikasyonuydu** — kör-eğitilmiş
> backbone (R1), çözünürlüğü yetmeyen ölçüm aparatı (R2), tek bilgi kaynağı (R3), geometriye sığmayan görev
> tanımı (R4) ve iki hastalığı karıştıran istatistik (R5). Bunlar düzeltilmeden yeni kaldıraç denemek,
> aynı mezarlığa yeni mezar kazmaktır.
