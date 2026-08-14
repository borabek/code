# "CLOSED" hukumlerinin denetimi — 2026-08-11

## Neden this belge present

Yon yelpazesi kolunu 64 yonlu a sondayla olcup **OLU** ilan ettim
(NIT'te +0.0000 recall). Sonra diff ettim ki 64 yonun kure uzerindeki komsuluk
araligi ~25 derece, measurement toleransi ise 10 derece — **probe, aradigi etkiyi
fiziksel as olcemezdi.** 256 yonle same arm NIT'te **+0.0676 recall** verdi.

Yani verdict kolun not, SONDANIN kusuruydu. Bu, hafizada "CLOSED" diye stopped
diger kollar for de sorulmasi gereken a soru aciyor.

## Kapatma karari vermeden before sorulacak uc soru

1. **Sondanin cozunurlugu, olcumun toleransini tutturabiliyor mu?**
   (yelpaze: 25 derece adim vs 10 derece tolerans — HAYIR)
2. **Sonda, etkinin BEKLENDIGI yerde mi kosuldu?**
   (yelpazeyi before ilk 100 D6 parcasinda kostum; orada banka already 0.9293
   veriyordu ve olculecek pay yoktu. Bosluk NIT'teydi.)
3. **Taban, o kumede already guclu muydu?**
   (D6'da P6 +0.1681 gorunuyordu because D6'nin tabani zayifti; `tam` brand
   katlarinda gercek kazanc +0.0230.)

## Yeniden sinanmasi gereken kollar (oncelik sirasi)

Asagidaki kollar hafizada KAPALI. Her biri for "probe this etkiyi olcebilir
miydi?" sorusu isaretlendi. **Bu a yeniden-acma karari DEGIL**, sinama
listesidir; each biri before ucuz a ceiling sondasiyla yoklanir.

| arm | kapanma gerekcesi | supheli mi | why |
|---|---|---|---|
| K2.1 periyodiklik | detection +0.0126 but robot -0.0100 | **EVET** | direction KAYNAKTAN KOPYALANIYORDU; direction bankasi varken tekrar denenmeli — `lattice.py` + `order_stamp.py` bunun for yazildi |
| mesh tepeleri pool | "uc independent olcumde ZARAR" | **CLOSED DEGIL** | this oturumda ACILDI: direction bankasi otherwise mesh only FP uretiyor; ikisi birlikte konum recall'u 0.54 -> 0.98 |
| axis boyu ornekleme | "tavani very az oynatti, 11x candidate" | **EVET** | measurement YON BANKASI OLMADAN yapilmisti; bugun candidate sayisi regime kapisiyla yonetiliyor |
| ensemble (g7+g10) | birlesim -0.0068 | belki | tek measurement, confidence araligi none |
| ExtraTrees transferi | "transfer ETMIYOR" | hayir | mekanizma acik, kapasite sorunu not |
| few-shot gate | NULL | belki | ornek sayisi very kucuktu |
| TEL-B kanal profili | OLU | **EVET** | yelpazeyle same aile (isin tabanli); cozunurluk kontrolu yapilmadi |
| B-rep snap | DEAD | hayir | kok why bulunmustu (yaricap/yay hatasi) |
| GEO 0.70 | persistent kapali | hayir | uc independent gate |

## Kural

Bundan after a arm kapatilirken makbuza su uc alan YAZILIR:

```
sonda_cozunurlugu : olcumun toleransina per yeterli mi
sonda_kumesi      : etkinin beklendigi yerde mi kosuldu
taban_gucu        : o kumede baseline ne veriyordu
```

Bu alanlar olmadan "CLOSED" yazilmaz.
