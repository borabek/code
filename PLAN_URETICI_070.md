# PLAN: görülmemiş üreticide F1 ≥ 0.70 — 3 günlük kol listesi

> Bu liste **update-todos'tan ayrıdır**. Önce bu, sonra o.
> Ölçüm kümesi: `results/d5_4_sinav_kumesi.json` (250 parça / 13 üretici, mühür `1c26a66e4c4e6160`).
> Her kol **iki kümede birden** raporlanır: görülmemiş-üretici sınavı **ve** 194'lük küme.

## Başlangıç noktası (bugün ölçüldü)

```
                   194'lük küme   görülmemiş 13 üretici
DAĞITILMIŞ ürün       0.7584*         0.3680
TABAN (1190 parça)    0.6778          0.3475
v3    (2955 parça)    0.6836          0.4010
*resmî manşet, bekçili alt küme — diğerleriyle kıyaslanmaz
```
Üretici yayılımı: en iyi 0.6154 — en kötü 0.0541. **Asıl problem bu yayılım.**

## VERİ KOLU KAPANDI — bu bir talep değil, bir eleme

Bugün ölçülen eğim: korpus ×2.48 → +0.0535, yani `b ≈ 0.0588`.
0.4010 → 0.70 için `ln(oran) = 0.299 / 0.0588 = 5.08` → korpusun **161 katı** gerekirdi.

**Bu hesap daha fazla veri istemek için değil, veri fikrini masadan kaldırmak için yapıldı.**
Aşağıdaki dört kolun hiçbiri yeni parça istemiyor; hepsi **eldeki 4405 parçayla** çalışıyor.
Yapılan şey korpusu büyütmek değil, `F1 = a + b·ln(grup)` denkleminin **a ve b'sini
yükseltmek** — yani aynı veriden daha çok genelleme çıkarmak:

| kol | neyi yükseltiyor | mekanizma |
|---|---|---|
| G2 üretici-dengeli | **b** | eğitim satırlarının %69'u WEI+PXC; ağırlık eşitlenir |
| G3 ikiz-ağırlıklı | **b** | grup başına ~2.5 kopya ezberletiyor; grup eşit ağırlık alır |
| G4 öznitelik budama | **b** | ezberleyen sütunlar atılır, transfer eden kalır |
| G5 oto-etiket | **a** | temsilin kendisi düzelir → her korpus boyunda kazanç |

---

## ⚠ G1 SONUCU — PLAN DEĞİŞTİ (2026-08-04 gecesi)

```
KÖTÜ YARI (6 üretici, 466 FN)      İYİ YARI (361 FN)
  ADAY_YOK        %76.2               %57.6    ← ayıran şey BU
  GATE_REDDI      %12.9               %14.1    ← AYNI, ayırt etmiyor
  ADAY_KALABALIK  %10.3               %27.4
```
Üreticiler arası yayılımın sebebi **gate değil temsil**. Tavan hesabı:

```
ŞU AN                                        F1 0.4010  (FN: ADAY_YOK 563 = %68.1)
MÜKEMMEL GATE (tüm reddi kurtar + SIFIR FP)     0.5964
+ MÜKEMMEL ÇÖZÜNÜRLÜK                            0.7038   ← hedefin TAM DİBİ
+ ADAY_YOK'un yarısı kurtarılırsa                0.8692
```
**G2/G3/G4 matematiksel olarak 0.70'i getiremez.** Fiziksel olarak imkânsız biçimde
mükemmel olsalar bile tavan 0.7038. Gün-1 kapısı ateşledi → **G5 öne alındı.**
G2/G3/G4 iptal değil; 0.5964 tavanının altında ek katkı için G5 koşarken paralel yürür.

## GÜN 1 (revize) — G5 başlatıldı + ucuz gate kolları paralel (hepsi gate düzeyi, eğitim dakikalar sürer)

### G1 · Üretici otopsisi — NEDEN 0.054 ile 0.615 arası?  *(2 saat, eğitim yok)*
F0-4'ün FN taksonomisini **sınav kümesine** uygula, üretici kırılımıyla:
`ADAY_YOK` (temsil) / `GATE_REDDI` (kapı) / `ADAY_KALABALIK` (çözünürlük) / `ATAMA`.
- Kötü üreticilerde baskın kova hangisi? Hepsinde aynı mı?
- **Bu, Gün 2'nin hangi kola gideceğini belirler.** Önce bu koşmadan başka kol açma.
- Ayrıca: kötü üreticilerin parça ölçeği / CP yoğunluğu / çerçeve residual'i dağılımı.
- **Çıktı:** üretici × kova tablosu + "erişilebilir hata" sayısı.

### G2 · Üretici-dengeli eğitim  *(1 saat)*
Eğitim satırlarının **%69'u WEI+PXC** (8978 + 5840 / 21377). Gate doğal olarak onlara uyuyor.
- Üretici başına eşit ağırlık (`sample_weight`), routing YOK, mimari YOK.
- **Beklenti +0.02…+0.05** — dengesizlik çok sert olduğu için en olası ucuz kazanç.
- **GO:** görülmemiş ≥ +0.020 **ve** 194'lük küme ≥ −0.010.

### G3 · İkiz-ağırlıklı eğitim  *(1 saat)*
Geometri grubu başına ~2.5 parça; ağırlıklandırılmazsa kalabalık aileler ezberletiyor.
Her **grup** eşit ağırlık.
- **Beklenti +0.01…+0.03.** **GO:** görülmemiş ≥ +0.015.
- **Tuzak:** etkin N düşer; tek nokta yanıltır → G2 ile birlikte değil, **ayrı** ölç.

### G4 · Transfer eden özniteliğe budama  *(2 saat)*
13 gate özniteliğinden yalnız `votes` genelleşiyor ([[gate-memorizes-not-learns]]).
Düşük kapasiteli gate (`max_depth` sınırlı, `min_samples_leaf` yüksek) + transfer eden alt küme.
- **Beklenti +0.02…+0.04 görülmemişte, −0.02…−0.04 havuzlanmışta.** Bu bir **takas**.
- **GO:** görülmemiş ≥ +0.025 **ve** kesinlik ≥ 0.60 (dejenere "hepsini kabul et" modelini eler).

> **Gün 1 sonu kapısı:** G2+G3+G4 en iyi kombinasyonu ≥ **0.47** vermezse
> Gün 2'de temsil koluna geçilir ve gate kolları bırakılır.

---

## GÜN 2 — temsil (kök neden, tek gerçek büyük kol)

### G5 · Ağız-çevrimi oto-etiket + seg yeniden eğitimi  *(gündüz hazırlık, GECE eğitim)*
Seg ağı **71 parçayla** eğitildi ve o 71 parça birkaç üreticiden. Üreticiye özgü çöküşün
kök nedeni burada; gate katmanındaki hiçbir şey bunu düzeltemez.
- Üretici CP'sinden ekleme yönü boyunca ışın → **açıklığın sınır çevrimi** CableEntry boyanır.
- **h3'ün r=2mm DİSKİ DEĞİL** — o etiket ateşlemeyi %78 → %24 çökertmişti.
- Seg korpusu 71 → ~1500, **13+ üretici**. 5 sınıf ve `v_o` DEĞİŞMEZ (tez-sadık).
- Üç seed, `train_seg_extra.py --train-dir`.
- **BAĞLAYICI KILL:** ateşleme %78 altına düşerse **veya** görülmemiş F1 −0.005'ten fazla düşerse geri al.
- **Beklenti +0.05…+0.15 görülmemişte** — geniş bant, çünkü ölçülmemiş.

### G6 · Yeni ağ geçerse tüm downstream yeniden türetilir  *(~3 saat, otomatik)*
Aday havuzu değişti → eski ölçümler geçersiz. Sınav + 194'lük küme yeniden.

---

## GÜN 3 — birleştir, doğrula, mühürle

### G7 · En iyi kombinasyon + grup-bootstrap güven aralığı  *(3 saat)*
- Geçen kolları **tek tek** ekleyerek ölç (ablasyon), hangisinin ne getirdiği belli olsun.
- Görülmemiş-üretici F1 için grup-bootstrap GA; alt sınır 0.65'in üstünde mi?
- Üretici yayılımı: en kötü üretici ≥ 0.30 olmalı, yoksa "ortalama 0.70" aldatıcıdır.

### G8 · Ürün kararı + manşet  *(2 saat)*
- İki kümeli manşet tablosu: 194'lük + görülmemiş, yan yana, kapsam etiketleriyle.
- Dağıtım yalnız her iki kümede de gerileme yoksa.

### G9 · Rapor  *(1 saat)*
Ne çalıştı, ne çalışmadı, hangi sayı hangi kapsamda. Stajın teslim edilebilir çıktısı.

---

## Dürüst olasılık

| senaryo | görülmemiş F1 | olasılık |
|---|---|---|
| G2+G3+G4 tutar, G5 üst banttan gelir | **0.65 – 0.72** | düşük |
| G2+G3+G4 tutar, G5 orta bant | **0.52 – 0.60** | **en olası** |
| gate kolları null, G5 orta bant | 0.45 – 0.52 | orta |
| hepsi null | 0.40 – 0.43 | düşük |

**0.70 ancak temsil kolu üst banttan gelirse mümkün.** Merkez beklenti ~0.55.
Bunu 3 gün boyunca her akşam güncelleyeceğim; 0.70 uzaklaşırsa saklamayacağım —
o durumda teslim edilebilir sonuç şudur ve zayıf değildir:

> "Ürün, hiç görmediği 13 üreticide F1 0.3680'den 0.55'e çıkarıldı (+%49 göreli);
> aynı dönemde havuzlanmış performans korundu. Üretici genellemesinin ölçülebilir
> hale getirilmesi (250 parçalık mühürlü sınav kümesi) bu çalışmanın kalıcı çıktısıdır."

## Değişmeyecekler (tez sadakati)

DiffusionNet mimarisi · 5 sınıf · ~6000 uniform izotropik remesh · `v_o` ağız-ortası türetmesi.
Hiçbir kol bunlara dokunmaz. G5 yalnız **etiket sayısını** artırır.
