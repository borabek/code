# GEOMETRİYİ TEZE SADIK GELİŞTİRME — to-do (2026-07-29)

## Çerçeve değişikliği

Üç bağımsız ölçüm aynı sonucu verdi: **geometri açıklığı bulur, tel girişini ayırt edemez**
(`geo_cp` 0.400 · B-rep 0.332 · yarık+süzgeç 0.302). Rakip dedektör arm **kalıcı kapalı**.

Ama tezin boru hattı şu:

```
STEP → izotropik remesh (~6000) → DiffusionNet segmentasyon → açıklık türetme (v_o) → CP
```

Burada **ikinci ve dördüncü adım tamamen geometrik** ve ikisi de bugüne up to hiç geliştirilmedi.
Geometri, segmentasyonla *yarışmak* yerine ona *hizmet* ettiğinde teze sadık kalır — çünkü tezin
kendi tarifi already geometrik türetmeye dayanıyor.

**Bu listedeki hiçbir madde segmentasyon ağını, korpusunu veya bölünmesini değiştirmez.**

---

## A. REMESH — tezin kendi kuralını uygula (en yüksek beklenti)

Tez *izotropik* remesh diyor: **uniform vertex YOĞUNLUĞU**. Biz sabit vertex *SAYISI* kullanıyoruz.
Parçalar farklı boyuttayken this, tezin kuralını **ihlal ediyor**. Ölçüldü: çok-CP parçaları %51 more
büyük but aynı 6000 vertex'i alıyor → yoğunluk 0.531 vs 0.790 /mm².

- [ ] **A1 — Adaptif hedef.** `target = alan_mm² × 0.790` (sabit yoğunluk). Çok-CP medyanı için
      ~8900. **Ölçüm:** candidate-recall + CP-F1, regime ayrımlı. **Kill:** corpus-temsili katkı < 0.02.
      *Not: türetildi but HİÇ ÖLÇÜLMEDİ — min_v10 sorunu maskeleyince ertelendi.*
- [ ] **A2 — Delik-koruyan remesh.** CC-A ölçtü: çok-CP açıklıklarının **%83'ünde 30 vertex bile
      none**. İzotropik remesh küçük açıklıkları eziyor. B-rep açıklıkların **nerede** olduğunu
      biliyor → o bölgelerde yerel yoğunluk artırılabilir (pymeshlab seçici refine).
      **Tez tartışması:** uniform yoğunluktan deviation. Gerekçe: tezin amacı JSON–STEP tessellation
      eşleşmesiydi; delik korumak o amaca zarar vermiyor. **Makbuzda ayrıca işaretlenir.**
      **Kill:** candidate-recall katkısı < 0.03.
- [ ] **A3 — Remesh kalite denetimi.** Kaç açıklık remesh'te *kayboluyor*? B-rep'teki each silindir/
      yarık için remeshlenmiş mesh'te karşılık present mı (ışın testiyle). Bu a **teşhis**, kaldıraç
      değil — A1/A2'nin ne up to yer açtığını gösterir.

## B. AÇIKLIK TÜRETME (v_o) — tezin dördüncü adımı

Tez CP'yi açıklığın **ağız merkezi** (v_o) as tanımlıyor. Bizim türetmemiz mesh bileşenlerinin
ağırlık merkezinden geliyor; dik error 0.1–2.3 mm. Robot hedefi <2mm. B-rep **tam** merkez/axis verir.

- [ ] **B1 — B-rep'e yapıştırma (snap).** ML'in bulduğu CP'yi, 3mm içindeki B-rep silindirinin
      **tam eksenine ve ağız merkezine** yapıştır. Tespit ML'de kalır — geometri sadece **konumu
      düzeltir**. **Kill:** medyan dik error azalmıyorsa (≥%20) ölü.
      *Bu, F1'i değil ROBOT İSABETİNİ hedefler — ayrı ve ölçülebilir a değer.*
- [ ] **B2 — Eksen düzeltmesi.** Insert yönünü mesh normal ortalamasından değil, B-rep silindir
      ekseninden al. **Ölçüm:** GT InsertDirection with açı hatası (şu an ölçülmüyor bile).
- [ ] **B3 — Yuva↔ağız dönüşümü kesinleştir.** Ölçüldü: GT bazen katı içinde (3270115: 16/16),
      bazen yüzeyde (3273112: 0/19). B-rep with ağız **deterministik** hesaplanabilir; şu an ışın
      taramasıyla prediction ediliyor. Eşleşme toleransını daraltmayı mümkün kılar.

## C. GEOMETRİ SEGMENTASYONA GİRDİ OLARAK (tez sınırında — dikkatli)

- [ ] **C1 — Yarık maskesi girdi kanalı.** G2 yarık dedektörü çalışıyor (**recall 0.860**).
      Bunu DiffusionNet'e ek vertex özelliği (0/1 "yarık yüzeyinde mi") as vermek, ağın hiç
      görmediği a ipucu ekler.
      **DİKKAT:** ağın girdi boyutu değişir → **yeniden eğitim gerekir** → tez korpusu/bölünmesi
      korunsa bile "tezin ağı" değişir. **Bu madde however A ve B tükendikten after ve açık
      gerekçeyle açılır.** Şimdilik yalnız fizibilite notu.

## D. NEGATİF SONUCU TEZE YAZ (ölçüm değil, yazım — but değerli)

- [ ] **D1 — "Geometri why yetmez" bölümü.** Elimizde bunun için sağlam kanıt present ve this, tezin
      *segmentasyon kullanma gerekçesini* güçlendirir:
      - saf geometrik dedektör 3 varyantta 0.30–0.40
      - hayatta kalan FP'ler mevcut 41 özellikle AUC 0.36/0.61
      - kanal geometrisi +0.003 F1, B-rep özellikleri +0.005 F1
      - **sızıntılı üst sınır** dürüst skoru sadece +0.003 geçiyor → bilgi duvarı
      - terminal bloklarında 69 gerçek girişe karşı **289 aynı görünen delik**
- [ ] **D2 — Ölçüm protokolü bölümü.** Rejim ayrımı, corpus ağırlıklandırma, kill kriteri,
      üretici-dışı sağlamlık testi. Bunlar this projede *hatalardan* öğrenildi ve yazılmaya değer.

---

## Sıra ve gerekçesi

| # | iş | why this sırada | prediction |
|---|---|---|---|
| 1 | **A3** teşhis | ucuz, A1/A2'nin tavanını gösterir | 1-2 sa |
| 2 | **A1** adaptif hedef | türetilmiş, ölçülmemiş, tezin kendi kuralı | 3-4 sa |
| 3 | **B1+B2** snap/axis | F1'den bağımsız değer (robot isabeti), düşük risk | 3-4 sa |
| 4 | **A2** delik-koruyan | en büyük potansiyel, en yüksek tez-riski | 5-6 sa |
| 5 | **B3** ağız kesinleştirme | A2'den after anlamlı | 2 sa |
| 6 | **D1+D2** yazım | ölçümler bitince | 3-4 sa |
| — | C1 | **kapalı** — A ve B tükenene up to açılmaz | — |

## Bu listede OLMAYANLAR (ve nedeni)

- ~~Rakip geometrik dedektör~~ → üç kez ölçüldü, kalıcı kapalı
- ~~Yeni FP süzgeci~~ → bilgi duvarı kanıtlandı (sızıntılı sınır +0.003)
- ~~Segmentasyon yeniden eğitimi~~ → çok-CP için insan etiketi none (CAD sözde-etiketi GT with F1~0.46)

## Kurallar (öncekilerle aynı, pazarlıksız)

1. Kill kriterleri **CP-F1 veya robot isabeti** üzerinden, baştan yazılı — AUC değil
2. Her kaldıraç **üretici-dışı** bölünmede de sınanır (ExtraTrees dersi)
3. Kilitli holdout **harcandı** — yeni a aile bölünmesi kurulmadan hiçbir sayı "doğrulanmış" denemez
4. Tezden each deviation makbuzda **ayrıca işaretlenir** ve gerekçesi yazılır

---

## S. ATLANAN SONDALAR (6 saat kısıtı yüzünden koşulmadı — şimdi ekleniyor)

Eski `GEO_070_PLAN.md`'de dört probe vardı; **S1 ve S4 koşuldu, S2 ve S3 atlandı.**
S2 özellikle önemli: geo kolunun **yeniden açılma şartlarından biri** ("AP214 renk kaydı") tam da
this sondayla sınanacaktı ve hiç bakılmadı.

- [ ] **S2 — AP214 renk/malzeme kaydı.** STEP'lerde yüzey rengi present mı (plastik gövde vs metal
      kontak)? Varsa this, "kanal metalde bitmeli" testinin **çok-katı olmadan** yapılabilmesi
      demektir — C3'ün ölü sayılmasının sebebi ortadan kalkar.
      **Sonuç kapalı a arm YENİDEN AÇABİLİR** → bulgu ne olursa olsun makbuza yazılır.
- [ ] **S3 — DIN-ray izi.** Arka yüzde 35mm standart ray oluğu B-rep'ten bulunabiliyor mu?
      Bulunursa **kanonik çerçeve** kurulur (arka/ön/üst) ve montaj delikleri yüzeyine göre
      elenebilir. Bu, B1 snap'inin de doğruluğunu artırır.
- [ ] **S5 — 4 kollu tablo (yarım kalmıştı).** (baseline/mv10) × (küçük/büyük corpus), regime ayrımlı.
      Veriler hazır; kazançların toplanıp toplanmadığını tamamlar.

**Sıraya eklenişi:** S2 ve S3 **en başa** — ucuzlar (15'er dakika) ve sonuçları A/B/C sırasını
değiştirebilir. S5 en sona, ölçüm bütünlüğü için.
