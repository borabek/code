# ÖN KAYIT — veri büyütme kaldıracı (2026-07-30)

Sonuç görülmeden yazıldı. Amaç: ölçüm bittikten sonra barajı sonuca göre kaydırmamak.

## Hipotez

Eksik STEP'leri indirmek CP-F1'i artırır, çünkü:

1. **Miktar.** Kullanılabilir havuz 1906 parça. Census'ün bulduğu kadarı eklenirse ~%40-45 büyüme.
2. **Çeşitlilik (asıl gerekçe).** Havuz şu an SADECE 2 üretici (WEI 1010 / PXC 896).
   Eksik listede A-B 190, SIE 310, KLM 101, CWT 68 var. Ölçülmüş zayıf eksenimiz
   üretici-dışı transfer (ExtraTrees orada −0.061 yemişti, kilitli holdout'ta düşük-CP
   −0.029 ile doğrulandı). Yeni üretici = o eksende gerçek eğitim sinyali.

## ÖLÇÜLEN gerçek büyüklük *(sayım bitti — beklenti aşağı çekildi)*

Sayım 873 eksik parçanın hepsini WSCAD'de tek tek aradı (null testi 6/6 geçti):

| kapı | kalan |
|---|---|
| eksik parça | 873 |
| WSCAD'de **birebir** parça-no eşleşmesi | 350 (%40) |
| + **aynı marka** olma şartı | 325 *(25'i çapraz-marka çakışması: KLM→Legrand/Pilz/Gira…)* |
| + kapsam (`*ElectricalTerminal*`) | **325 / 325 — hepsi terminal** |

Havuz **1906 → 2231 (×1.17)**, beklediğim ×1.45 değil. Log ekstrapolasyonu:

    +0.0107 × ln(1.17)/ln(1.92) ≈ **+0.0026**

**Bu kendi kill eşiğimin (+0.005) ALTINDA.** Yani ham miktar gerekçesi TEK BAŞINA yetmiyor —
dürüst olan bunu şimdiden yazmak.

**Asıl gerekçe miktar değil ÇEŞİTLİLİK:** havuzda şu an yalnız 2 üretici var (WEI, PXC).
Gelen 325 parçanın 186'sı **Allen-Bradley (1492 klemens serisi)**, 106'sı **Siemens (8WH2000
klemens serisi)**, 6'sı **WAGO (2022 TOPJOB S)** — yani üretici sayısı **2 → 6**. Ölçülmüş zayıf
eksenimiz tam da bu (ExtraTrees: aile-dışı +0.012 → üretici-dışı −0.061; kilitli holdout'ta
düşük-CP −0.029 ile doğrulandı). Bu kaldıraç F1'i değil **transferi** hedefliyor.

**Bu yüzden karar kuralına ikinci bir kol ekleniyor** (aşağıda).

## Karar kuralı (ÖNCEDEN)

| ölçülen | karar |
|---|---|
| ≥ +0.005 ağırlıklı | ÜRÜNE GİRER, korpus kalıcı büyür |
| +0.001 … +0.005 | veri tutulur (bedava), ürün iddiası GÜNCELLENMEZ |
| ≤ +0.001 veya negatif | veri kaldıracı ÖLÜ ilan edilir, yazılır, bir daha açılmaz |

**İkinci kol — transfer (asıl hedef).** Ham F1 barajı geçmese bile, **üretici-dışı sonda**
(WEI-dışarıda / PXC-dışarıda) eski korpusa göre **≥ +0.01** iyileşirse veri kalıcı olarak alınır:
ölçülmüş kırılganlığımız orada, ve holdout bir kez o yüzden yandı. İki kol AYRI raporlanır,
biri diğerinin yerine geçmez.

**Not (dürüstlük):** gelen 325 parçanın 321'i düşük-CP. Düşük-CP için "daha çok AYNI dağılımdan
veri" tükendi diye ölçmüştük (sızıntılı üst sınır yalnız +0.003). Bu partinin farkı aynı
dağılımdan olmaması — yeni üreticiler. Eğer transfer kolu da kıpırdamazsa, **veri kaldıracı
gerçekten kapanır** ve öyle yazılır.

Ek koşul: **düşük-CP gerilerse (−0.005'ten fazla) kabul edilmez** — nüfusun %89.5'i orada.

## Ölçüm protokolü (P9 ile birebir aynı olmalı)

- 100 parça (70 düşük / 30 çok-CP), test aileleri gate eğitiminden ÇIKARILIR
- eşikler büyük veride aile-dışı OOF ile seçilir, test kümesine bakılmadan
- korpus-ağırlıklı: düşük 0.895 / çok 0.105
- **kilitli holdout HARCANDI (2026-07-29) — kullanılamaz**
- karşılaştırma AYNI test parçaları üzerinde: eski gate vs yeni (büyük korpus) gate

## Bilinen tuzaklar (bu projede yaşandı)

- **Elma-armut havuz kıyası:** gate'i 823 parçalık havuzda ölçüp 1903'lük havuzla
  kıyaslamak kaldıracı ÖLÜ göstermişti; aynı havuzda ölçünce +0.1273 çıktı.
- **Tek-türetme vekili geçersiz:** union'ın değeri modeller arası çeşitlilikten gelir,
  union olmadan görünmez (3 yanlış cevap, 2'si TERS işaretli).
- **S5 dersi:** aday-arzı değişikliği küçük korpusta yargılanmaz (min_v10 tek başına
  −0.001, çift korpusla çift +0.037).

---

# SONUÇ — KALDIRAÇ ÖLDÜ (2026-07-30, ölçüldü)

İndirme koştu: **325 deneme → 7 dosya.** Korpus **1906 → 1913 (×1.004)**.

| kapı | kalan | kayıp nedeni |
|---|---|---|
| eksik parça | 873 | |
| WSCAD'de aranabilir | 350 | 523'ü katalogda hiç yok |
| aynı marka | 325 | 25 çapraz-marka çakışması |
| **STEP olarak yayınlanmış** | **7** | **318'i yalnız WSCAD verisi + EPLAN `.edz` olarak var** |

## Kök neden — ve kendi ölçüm hatam

Sayımım **aranabilirliği** ölçtü, **indirilebilirliği** değil. İkisi aynı sanmıştım; değiller.
Parçanın kendi metadata'sı (`partType 0 → Data/part.json`) bunu açıkça söylüyor:

    formats: [0, 1]        # 0 = WSCAD verisi, 1 = EPLAN .edz. Format 2 (STEP) YOK.

Allen-Bradley 1492 klemenslerinin **186'sı da** aranabilir ve **hiçbirinin** STEP'i yok.
Siemens 8WH ailesinde de aynı: 8WH2000/8WH3000 serilerinin tamamı STEP'siz, yalnız
`8WH6000-0AF00` yayınlanmış. `.edz` tescilli EPLAN biçimi — gmsh okumaz, kullanılamaz.

**Ders:** bir kaynağın "var" demesi "alınabilir" demek değil. Erişilebilirlik sayımı,
gerçekten kullanacağın uç noktaya sorulmalı — arama uç noktasına değil.

## Karar (ön kayıttaki kurala göre, kural değiştirilmeden)

Beklenti +0.0026 idi ve o bile kill eşiğinin altındaydı. Gerçekleşen büyüme ×1.004,
yani **ölçülebilir etki sıfır**. Ön kayıt "≤ +0.001 → ÖLÜ ilan edilir" diyordu.

**WSCAD üzerinden veri büyütme kaldıracı KAPANDI.** Ölçüm bile koşulmadı: koşmanın anlamı yok,
7 parça hiçbir metriği kıpırdatmaz. Ölçüm koşup "+0.001 çıktı" demek dürüst olmazdı — hesap
zaten önden yapılabilirdi ve yapıldı.

**Çeşitlilik kolu da kapandı:** yeni üretici gelmedi (7 parçanın 6'sı WAGO, 1'i Siemens).
Üretici sayısı 2 → 4 oldu ama 7 parçayla üretici-dışı transfer eğitilemez.

## Yine de kalıcı kazanç (bedava değildi, ama duruyor)

* `verify_downloads.py` — indirilen STEP gerçekten o parça mı? Eksen-oranı parmak izi
  (20/20 doğru geçer, 15/20 yanlış yakalanır; kaçan 5'i geometrisi BİREBİR aynı kardeş
  varyantlar — hiçbir geometrik test ayıramaz). 7/7 yeni dosya doğrulandı.
* Çapraz-marka kapısı — 25 parça korpusa girmeden elendi (KLM→Legrand/Pilz/Gira…).
  Bunlar sessizce GT'yi zehirleyecekti.
* `gate_regrow.resume_partial` — devam kaydı artık parça numarasına bağlı (indeks kaymasıyla
  metriği sessizce bozuyordu), 4 test.
* Doğru Allen-Bradley üretici ID'si (tabloda 4470 yazıyordu, doğrusu 5).
