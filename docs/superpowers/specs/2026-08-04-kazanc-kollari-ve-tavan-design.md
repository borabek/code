# Kazanç kolları ve Bayes tavanını yükseltme — tasarım (2026-08-04)

## Neden bu tasarım var

Tespit F1 günlerdir 0.7584'te duruyor. Dokuz gate mekanizması, fiziksel skalerler ve
profil öznitelikleri kapandı. Bugün T1 ölçümü fiziksel ölçümlerin ayrılamaz payı
%12.8 → %12.8 (göreli azalma %0.0) bıraktığını gösterdi.

Kod tabanına bakınca sebep göründü:

```
scheffler_dataset.EXPECTED_COUNTS = {'train': 71, 'val': 20, 'test_locked': 11}
wire_gate.py                       → komşu / periyodiklik özniteliği: 0
diffusionnet.predict               → gate'e giden tek şey: 5 sınıf olasılığı
```

**Öğrenilen temsil 71 parçayla eğitildi; türetme korpusu 4432.** Kapanan dokuz kolun
hepsi bu 71 parçanın ürettiği beş olasılığın *üstünde* oynuyordu. Bayes tavanı modelin
değil öznitelik uzayının özelliğidir; uzay beş olasılık + el yapımı skalerlerse tavan
oradan gelir.

Bu tasarım üç kol açar. Üçü de **tez-değişmezlerine dokunmaz**: DiffusionNet mimarisi,
5 sınıf, ~6000 uniform izotropik remesh ve `v_o` ağız-ortası türetmesi aynen kalır.

## Kollar

### C — Ağı derinden oku (gate'e gizli öznitelik)

Gate şu an adayı beş olasılıkla tanıyor. Ağın son difüzyon bloğunun çıktısı çok daha
zengin bir geometri kodlaması taşıyor ve hiç kullanılmadı.

- **Nasıl:** son bloğa forward hook. Adayın `UYE` tepelerinde ortalama + maksimum
  havuzlama → 2·c_width boyut. TRAIN üzerinde fit edilen PCA ile ~16 boyuta indirilir
  (PCA'nın VAL/LOCKED görmesi sızıntıdır; fit yalnız TRAIN'de).
- **Tez etkisi:** yok. Ağırlık, mimari, sınıf sayısı, kayıp fonksiyonu değişmez;
  checkpoint'ler yeniden eğitilmez. Sadece zaten hesaplanan bir ara tensör okunur.
- **Tavan sondası T2:** T1 ile aynı kNN protokolü, bu sütunlar eklenmiş halde.
- **GO:** ayrılamaz pay ≥%20 göreli azalsın **ve** üretici-dışı ≥ +0.010.
- **Risk:** [[extratrees-does-not-transfer]] deseni — havuzlanmışı iyileştirip görülmemiş
  üreticide çökebilir. Bu yüzden üretici-dışı şartı GO'nun içinde, sonradan bakılacak
  bir kontrol değil.

### B — Skaler yerine yapı (kafes / periyodiklik / komşuluk)

`wire_gate`'de komşu özniteliği yok: her aday tek başına karar veriliyor. Ama klemens
blokları düzenli dizidir. On kutbun sekizi ateşlediyse eksik ikisinin nerede olduğu
komşularından bellidir.

- **Öznitelikler (aday başına, ilişkisel):** baskın adımın (pitch) ana eksen
  izdüşümünden tespiti; adayın en yakın kafes düğümüne uzaklığı; aynı sırada eş-doğrusal
  komşu sayısı; sıra içi rank; ayna simetrisi tutarlılığı.
- **Tez etkisi:** yok. Yalnız gate/son-işlem katmanı; `v_o` türetmesi ve aday üretici
  aynen kalır.
- **Tavan sondası T3:** aynı kNN protokolü. **Bedava** — `_der_tam.pkl` içindeki `P`/`Pd`
  yeterli, ağ çıkarımı gerekmez.
- **GO:** ayrılamaz pay ≥%20 göreli azalsın **ve** havuzlanmış ≥ +0.015.
- **Yan ürün (AYRI ölçülür): kafes tamamlama.** Komşuları ateşlemiş boş kafes
  düğümlerine aday öner. 366 yüksek-CP FN'nin doğal hedefi. Ayrı GO: yüksek-CP
  ≥ +0.020 ve düşük-CP ≥ -0.005. C/B ile karıştırılmaz.

### A — Temsili büyüt (ağız-çevrimi oto-etiket)

En yüksek tavanlı, en pahalı kol. Seg korpusu 71 → ~1500+.

- **Nasıl:** her üretici CP'sinden ekleme yönü boyunca ışın; yüzeye çarptığı açıklığın
  **sınır çevrimi** CableEntry boyanır.
- **h3 dersi:** [[h3-highcp-finetune-dead]] aynı fikirde çöktü çünkü CP çevresine
  **r=2mm disk** boyadı ve ateşlemeyi %78 → %24 düşürdü. Disk gövde yüzeyini de
  boyuyordu; ağız *çevrimi* o hatanın düzeltmesidir.
- **Tez etkisi:** yok. 5 sınıf ve `v_o` değişmez; yalnız etiketli parça sayısı artar.
  Eğitim `train_seg_extra.py --train-dir` ile, üç seed.
- **KILL (bağlayıcı):** ateşleme oranı %78'in altına düşerse **veya** görülmemiş üretici
  F1'i -0.005'ten fazla düşerse **geri al**.
  - Neden bu ölçüt: h3'ün gerçek çöküşü ateşlemeydi ve F1 o çökerken bir süre sabit
    görünür — "sadece F1" geç yakalar. Seg IoU ise zayıf etiketle doğal olarak
    düşebileceği için yanlış negatif üretir, iyi kolu boşuna öldürür.

## Sıra ve gerekçesi

```
ŞİMDİ    T3  yapısal tavan ölçümü          bedava, önbellekten, GPU kullanmaz
~3 saat  türetme biter → D5-3 → F2-12      ölçülmüş kazancı olan tek kol (veri)
sonra    T2  gizli öznitelik tavan ölçümü   ağ boşalınca
         geçen kolu TEK BAŞINA dağıt        atıf korunur
GECE     A   oto-etiket + seg eğitimi       uzun eğitim uykuya
```

İki kolu aynı anda **ölçmek** sorunsuz (ölçüm dağıtım değildir); aynı anda **dağıtmak**
atfı bozar. Bu yüzden ölçümler paralel, dağıtımlar sıralı.

Kritik yol aç bırakılır: veri kolu şu an koşuyor ve ölçülmüş kazancı olan tek şey o
([[ogrenme-egrisi-fiyat-etiketi]]: hacimden +0.030, [[cesitlilik-ve-fn-profili]]:
çeşitlilikten ayrıca +0.0443).

## Dürüst beklenti

- Veri kolu: +0.03 ~ +0.05 (eğriden, ölçülmüş)
- B: havuzlanmış katkısı sınırlı olabilir — yüksek-CP rejiminin ağırlığı %10.5
- C: ölçülmemiş; tavan sorusunun gerçek cevabı burada
- A: ölçülmemiş; düşük-CP'de aday tavanında +0.129 alınmamış pay var ve ona **yalnız
  temsil** dokunabiliyor

0.85'i gören senaryo: veri + A + C. B tek başına 0.85 getirmez ama FN kovasını açar.

## Doğrulama

- Her kol için önce tavan sondası; ≥%20 göreli azalma yoksa **kurulmaz**
- Dağıtım kararları grup-bootstrap ile, `protokol.dogrula()` zorunlu
- LOCKED'e dokunulmaz; [[locked-sinav-kirliligi-yakalandi]] bekçisi açık kalır
- `pytest tests/` yeşil kalır
