# ETİKETLEME PLANI — insan etiketiyle segmentasyonu iyileştirme (2026-07-29)

## Evet, senin `label_tool.html` ile

`label_tool.html` = **"WSCAD 5-class labeler"**. Her parça için `<pid>.obj` mesh'ini açıyorsun,
vertexleri 5 sınıftan biriyle boyuyorsun, `<pid>.labels.txt` çıkıyor. `_label_targets/` altında
22 parçalık bir kuyruk zaten hazır.

---

## Neden bu sefer farklı olmalı — önceki tur ZARAR VERDİ

Depodaki kendi ölçümlerimiz:

| önceki deneme | sonuç |
|---|---|
| 77 → 103 insan-etiketli parça | **"did not help"** |
| Kısmi etiket (yalnız CableEntry, maskeli BCE) | val Connection IoU **0.622 → 0.586** |
| İşaretleri korpus kapsamına genişletme | val IoU **0.676 → 0.630**, arbiter CP-F1 **0.520 → 0.351** |

**Sebep etiketlerin yanlış olması değildi — konvansiyon çelişkisiydi:**

1. **Kapsam:** insan açıklığın *çekirdeğini* boyadı (vertexlerin **%1.45**'i), korpus *bütün açıklık
   bölgesini* boyuyor (**%4.99**). Ağ iki çelişen tanım öğrendi.
2. **Hedef:** insan 5 sınıftan yalnız 1'ini işaretledi → o parçalar **maskeli kayıpla**, korpus tam
   **5-sınıf kayıpla** eğitildi. Farklı hedef fonksiyonu.

Bu plan bu iki sebebi ortadan kaldırmak üzerine kurulu. **Bunlar çözülmeden etiketleme yapılmamalı**
— yoksa üçüncü kez emek harcanıp model bozulur.

---

## L0 — KALİBRASYON (etiketlemeden ÖNCE, 1-2 saat) ← en kritik adım

Amaç: senin boyaman ile korpusun konvansiyonunun **aynı** olduğunu ölçmek.

- [ ] **L0.1** Korpustan 3 parça seç (zaten insan-etiketli, doğru cevabı biliyoruz).
      Etiketlerini gizle, sana `label_tool.html` ile boyat.
- [ ] **L0.2** Ölç: senin işaretlediğin vertex oranı vs korpusun oranı; sınıf bazında IoU.
      **Hedef: CableEntry+Contact oranı korpusun ±%30'u içinde, IoU ≥ 0.60.**
- [ ] **L0.3** Sapma varsa **talimatı düzelt, seni değil** — örnek görsellerle "bu kadar boyanır"
      referansı üret (korpus parçalarından kesitler).
- [ ] **KAPI:** L0.2 tutmadan tek yeni parça etiketlenmez. *(Önceki turun tam olarak atladığı adım.)*

## L1 — ÖLÇEK KARARI (ölçüyle, tahminle değil)

- [ ] **L1.1** Öğrenme eğrisi: mevcut 71 korpus parçasından **20 / 40 / 71** ile eğit, val IoU ölç.
      Eğri düzse yeni veri de düz kalır — o zaman **etiketleme yapılmaz**, karar burada verilir.
- [ ] **L1.2** Eğri canlıysa gereken parça sayısını eğriden **ekstrapole et** (körlemesine
      "100-150 parça" deme).
- [ ] **KAPI:** 71→N için beklenen IoU kazancı < 0.03 ise etiketleme **yapılmaz**.

## L2 — PARÇA SEÇİMİ (rastgele değil, hedefli)

- [ ] **L2.1** Ölçülmüş boşluğa göre seç: manşeti **düşük-CP belirliyor** (%89.5) ve orada tavan
      ağın **%86 ateşleme oranı**. Yani **düşük-CP parçalar** öncelikli.
- [ ] **L2.2** Ağın *ateşlemediği* GT noktalarına sahip parçaları önceliklendir (T5 ölçümü bunları
      zaten işaretliyor) — model orada yanılıyor, en çok bilgi orada.
- [ ] **L2.3** Aile çeşitliliği zorunlu: aynı aileden ≤2 parça (aile-dışı genelleme için).
- [ ] **L2.4** Kilitli holdout ailelerinden **hiçbiri** seçilmez.

## L3 — ETİKETLEME (asıl emek)

- [ ] **L3.1** **5 sınıfın hepsi** boyanır (Housing/Contact/SnapPoint/CableEntry/LabelSurface).
      Yalnız CableEntry boyamak ölçülmüş şekilde zarar veriyor.
- [ ] **L3.2** Kapsam **korpus konvansiyonu**: açıklığın çekirdeği değil, **bütün açıklık bölgesi**.
- [ ] **L3.3** CP tanımı `_label_targets/README.md`'deki cp-v3 kuralı: CableEntry **veya** gerçek
      derinliği olan Contact (kare/push-in kelepçe de CP'dir).
- [ ] **L3.4** Her 10 parçada bir L0 kalibrasyonu tekrarlanır (kayma kontrolü).

## L4 — DOĞRULAMA (etiketler eğitime girmeden önce)

- [ ] **L4.1** `ingest_batch2_labels.py`'nin katı doğrulaması: satır sayısı = OBJ vertex sayısı,
      uyuşmazlık = **reddet** (sessizce kullanma).
- [ ] **L4.2** Sınıf oranı denetimi: her parça korpus dağılımının ±%50'si içinde mi.
- [ ] **L4.3** Görsel denetim: sertifikalı GLB aracıyla (`glb_audit` mantığı) rastgele 5 parça.

## L5 — EĞİTİM ve ÖLÇÜM

- [ ] **L5.1** Korpus + yeni parçalar **tek ve aynı 5-sınıf kayıpla** eğitilir (maskeli BCE YOK).
- [ ] **L5.2** Ölçüm **rejim ayrımlı** (düşük-CP / çok-CP) + **üretici-dışı** sağlamlık.
- [ ] **L5.3** Tez metriği (val mean IoU) **ve** ürün metriği (CP-F1) birlikte raporlanır.
- [ ] **KILL:** val IoU'da kayıp VEYA düşük-CP CP-F1'de −0.01'den fazla düşüş → yeni veri **alınmaz**.

---

## Dürüst emek ve beklenti

| | |
|---|---|
| Parça başına etiketleme | 6000 vertex × 5 sınıf — **saatler**, dakikalar değil |
| L1 kapısı geçerse gereken | eğriden çıkacak; körlemesine tahmin **30–80 parça** |
| Beklenen kazanç | **bilinmiyor** — L1 tam da bunu ölçmek için var |
| Risk | önceki tur zarar verdi; L0/L4 kapıları bunun için |

## En önemli nokta

**Sıra: L0 → L1 → (kapı) → L2 → L3.**

İlk iki adım **etiketleme içermiyor** ve toplam 2-3 saat. İkisi de "bu iş değmez" diyebilir — o
zaman saatlerce boyama yapılmadan karar verilmiş olur. Önceki turun hatası bu iki adımı atlayıp
doğrudan boyamaya başlamaktı.
