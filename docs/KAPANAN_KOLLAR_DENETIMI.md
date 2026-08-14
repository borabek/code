# "KAPANDI" hukumlerinin denetimi — 2026-08-11

## Neden bu belge var

Yon yelpazesi kolunu 64 yonlu bir sondayla olcup **OLU** ilan ettim
(NIT'te +0.0000 recall). Sonra fark ettim ki 64 yonun kure uzerindeki komsuluk
araligi ~25 derece, measurement toleransi ise 10 derece — **probe, aradigi etkiyi
fiziksel olarak olcemezdi.** 256 yonle ayni arm NIT'te **+0.0676 recall** verdi.

Yani verdict kolun degil, SONDANIN kusuruydu. Bu, hafizada "KAPANDI" diye duran
diger kollar icin de sorulmasi gereken bir soru aciyor.

## Kapatma karari vermeden once sorulacak uc soru

1. **Sondanin cozunurlugu, olcumun toleransini tutturabiliyor mu?**
   (yelpaze: 25 derece adim vs 10 derece tolerans — HAYIR)
2. **Sonda, etkinin BEKLENDIGI yerde mi kosuldu?**
   (yelpazeyi once ilk 100 D6 parcasinda kostum; orada banka zaten 0.9293
   veriyordu ve olculecek pay yoktu. Bosluk NIT'teydi.)
3. **Taban, o kumede zaten guclu muydu?**
   (D6'da P6 +0.1681 gorunuyordu cunku D6'nin tabani zayifti; `tam` brand
   katlarinda gercek kazanc +0.0230.)

## Yeniden sinanmasi gereken kollar (oncelik sirasi)

Asagidaki kollar hafizada KAPALI. Her biri icin "probe bu etkiyi olcebilir
miydi?" sorusu isaretlendi. **Bu bir yeniden-acma karari DEGIL**, sinama
listesidir; her biri once ucuz bir ceiling sondasiyla yoklanir.

| arm | kapanma gerekcesi | supheli mi | neden |
|---|---|---|---|
| K2.1 periyodiklik | tespit +0.0126 ama robot -0.0100 | **EVET** | direction KAYNAKTAN KOPYALANIYORDU; direction bankasi varken tekrar denenmeli — `lattice.py` + `order_stamp.py` bunun icin yazildi |
| mesh tepeleri havuzu | "uc bagimsiz olcumde ZARAR" | **KAPANDI DEGIL** | bu oturumda ACILDI: direction bankasi yoksa mesh yalniz FP uretiyor; ikisi birlikte konum recall'u 0.54 -> 0.98 |
| axis boyu ornekleme | "tavani cok az oynatti, 11x candidate" | **EVET** | measurement YON BANKASI OLMADAN yapilmisti; bugun candidate sayisi regime kapisiyla yonetiliyor |
| ensemble (g7+g10) | birlesim -0.0068 | belki | tek measurement, confidence araligi yok |
| ExtraTrees transferi | "transfer ETMIYOR" | hayir | mekanizma acik, kapasite sorunu degil |
| few-shot gate | NULL | belki | ornek sayisi cok kucuktu |
| TEL-B kanal profili | OLU | **EVET** | yelpazeyle ayni aile (isin tabanli); cozunurluk kontrolu yapilmadi |
| B-rep snap | DEAD | hayir | kok neden bulunmustu (yaricap/yay hatasi) |
| GEO 0.70 | kalici kapali | hayir | uc bagimsiz gate |

## Kural

Bundan sonra bir arm kapatilirken makbuza su uc alan YAZILIR:

```
sonda_cozunurlugu : olcumun toleransina gore yeterli mi
sonda_kumesi      : etkinin beklendigi yerde mi kosuldu
taban_gucu        : o kumede baseline ne veriyordu
```

Bu alanlar olmadan "KAPANDI" yazilmaz.
