# YARIN — LİSTE 2: ETİKETLEME KARARI (L0 + L1)

> Bu **ayrı bir liste**. Bugünün listesi (H1/H3, S5, D1/D2) ayrı yürüyor.
> Bu listeye **yarın** bakılacak.

## Amaç

İnsan etiketlemesi yapılmalı mı? Karar **ölçüyle** verilecek, tahminle değil.
**İki adım da hiç boyama içermiyor** ve toplam 2-3 saat.

Önceki tur bu iki adımı atladı, doğrudan boyamaya başladı ve **model bozuldu**
(val Connection IoU 0.622 → 0.586; işaretleri genişletince 0.676 → 0.630, CP-F1 0.520 → 0.351).

---

## L0 — KALİBRASYON (1-2 saat, senden ~3 parça boyama)

**Soru:** Senin boyaman korpusun konvansiyonuyla aynı mı?

- [ ] **L0.1** Korpustan 3 parça seç (zaten insan-etiketli, doğru cevap elimizde).
      Etiketlerini gizle, `label_tool.html` ile sana boyat.
- [ ] **L0.2** Ölç: işaretlediğin vertex oranı vs korpusun oranı, sınıf bazında IoU.
      **KAPI: CableEntry+Contact oranı korpusun ±%30'u içinde VE IoU ≥ 0.60**
- [ ] **L0.3** Sapma varsa **talimatı düzelt, kullanıcıyı değil** — korpus parçalarından
      "bu kadar boyanır" görsel referansı üret.
- [ ] **L0.4** Kapı tutmadan **tek yeni parça etiketlenmez**.

**Neden kritik:** önceki turun kök sebebi buydu — insan açıklığın *çekirdeğini* boyadı (%1.45),
korpus *bütün açıklık bölgesini* boyuyor (%4.99). Ağ iki çelişen tanım öğrendi.

## L1 — ÖĞRENME EĞRİSİ (1 saat, senden hiçbir şey)

**Soru:** Daha fazla veri bu ağa gerçekten yarıyor mu?

- [ ] **L1.1** Mevcut 71 korpus parçasından **20 / 40 / 71** ile eğit, val mean IoU ölç.
- [ ] **L1.2** Eğriyi çiz. **Düzse yeni veri de düz kalır.**
- [ ] **L1.3** Canlıysa gereken parça sayısını **eğriden ekstrapole et**
      (körlemesine "100-150 parça" deme).
- [ ] **KAPI: 71 → N için beklenen IoU kazancı < 0.03 ise ETİKETLEME YAPILMAZ.**

---

## Karar tablosu (yarın bu doldurulacak)

| L0 | L1 | karar |
|---|---|---|
| geçti | geçti | **Etiketle** — L2 parça seçimine geç |
| geçti | kaldı | **Etiketleme** — veri bu ağa yaramıyor, başka kaldıraç ara |
| kaldı | geçti | **Önce talimatı düzelt**, L0'ı tekrarla |
| kaldı | kaldı | **Etiketleme** — iki sebepten de değmez |

---

## Kapılar geçerse (L2-L5, ayrıca planlanacak)

`ETIKETLEME_PLANI.md` içinde detaylı. Özet:
- **L2** hedefli parça seçimi: düşük-CP öncelikli (manşeti o belirliyor, %89.5), ağın
  ateşlemediği GT'ler öncelikli, aile çeşitliliği zorunlu, kilitli holdout aileleri hariç
- **L3** **5 sınıfın hepsi** boyanır (yalnız CableEntry ölçülmüş şekilde zarar veriyor),
  kapsam korpus konvansiyonu
- **L4** katı doğrulama: satır sayısı = OBJ vertex sayısı, uyuşmazlık **reddedilir**
- **L5** tek ve aynı 5-sınıf kayıpla eğitim; rejim-ayrımlı + üretici-dışı ölçüm;
  **KILL:** val IoU kaybı VEYA düşük-CP CP-F1'de −0.01'den fazla düşüş

## Dürüst emek

Parça başına **6000 vertex × 5 sınıf** — saatler, dakikalar değil.
Gereken sayı L1 eğrisinden çıkacak; şu an bilinmiyor.

---

## Yarın nereden başlanacak

**L1 önce** — çünkü senden hiçbir şey istemiyor ve tek başına "etiketleme yapılmaz" diyebilir.
L1 geçerse L0 için 3 parça boyaman istenecek.
