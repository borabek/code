# SABAH RAPORU -- 2026-08-12 05:24

Butun sayilar `tam` MARKA KATLARINDA (LOMO). D7 SINAVINA BAKILMADI. Manset metrik MIKRO robot F1.

## 1. GATE A -- tam-acik havuzun yonlu recall'u
| pool | cluster | konum | **yonlu** | F1 tavani | candidate/part | GATE A |
|---|---|---|---|---|---|---|
| tam-acik, ceiling 12 | d6 | 0.8997 | **0.7347** | 0.8470 | 266 | gecmedi |
| onceki pool | d6 | 0.8713 | **0.7264** | 0.8415 | 251 | gecmedi |
| onceki pool | tam | 0.9789 | **0.7352** | 0.8474 | 239 | gecmedi |

GATE A esigi: yonlu recall >= 0.85.

Kapi gecmezse pool genisletme arm KAPANIR: ceiling yetmiyorsa selector ne up to iyilesirse iyilessin hedefe ulasilamaz. CEILING, mukemmel a secicinin alacagi F1'dir -- VAAT DEGIL, UST SINIR.

## 1b. SECENEK TAVANI (MAX_SEC) BAGLIYOR MU?
**orneklem: ? / 18 part, yelpaze 256**

| ceiling | yonlu recall | secenek maliyeti |
|---|---|---|
| 12 | 0.5913 | 1.00x |
| 24 | 0.8755 | 1.33x |
| 48 | 0.8755 | 1.35x |

**orneklem: ? / 30 part, yelpaze 256**

| ceiling | yonlu recall | secenek maliyeti |
|---|---|---|
| 12 | 0.8462 | 1.00x |
| 24 | 0.9077 | 1.20x |
| 40 | 0.9077 | 1.23x |

> Bugunku ceiling **12**. Tavan bagliyorsa direction kaynagi eklemek (yelpaze cozunurlugu) recall'u ARTIRMAZ -- yeni yonler tavana takilip mevcutlarin yerini alir. `YB_MAX_SEC` with ayarlanir.

## 2. EK OZNITELIK BLOKLARI (gate +0.01)
Henuz receipt none.

## 3. KADEME2 KOL KIYASLARI
- **SIRA kapali (u25 baseline)** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.2938
- **SIRA acik (u25)** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.3041
- **B1 zor negatif (tam3)** (results/_p6_oz_tam3) secilen=P6 -> P6 0.3195
- **son kosu (uzerine yazilan)** (results/_p6_oz_tam3) secilen=P6 -> P6 0.3195

## 3b. SECICI VERIMLILIGI -- 0.50 nereden gelebilir?
- pool F1 TAVANI (`tam`, mukemmel selector): **0.8474**
- GERCEKLESEN (P6 arm): **0.3091**
- **selector verimliligi = 36.5%**

0.50'ye iki yoldan gidilebilir:
1. **Havuzla:** verimlilik sabit kalirsa tavanin **1.3707** olmasi gerekir  -> 1.0'i asiyor, TEK BASINA IMKANSIZ
2. **Seciciyle:** ceiling sabit kalirsa verimliligin **59.0%** olmasi gerekir (1.62x iyilesme)

> Havuz arm tek basina hedefe goturmuyor; SECICI arm zorunlu. Bu, EK bloklarina ve candidate-set modeline (D2) verilen onceligi belirler.

## 4. GECE FAZLARI
- [00:14:14] BITTI: A1b -- 141 dosya
- [04:12:23] BITTI: A1b -- 3051 dosya
- [04:12:31] BITTI: A2_havuz_tavani (8s)
- [04:58:24] BITTI: B1_zor_negatif (2752s)

## 4b. SAHA -- AUTO KATMANI (tier cokusu)
Dagitilan AUTO esigi = **0.6**
- `d7_p6`: AUTO payi **1.0000**, precision **0.3471**  <- REVIEW KATMANI BOS
- `d7_taban`: **OLCULMEMIS_varsayilan_dolgu** (2001 isaretin hepsi same skor)

> Gorulmemis markada robot HER isarete otonom guveniyor. Esigi yukseltmek kurtarmiyor (0.95'te bile precision ~0.47). Oneri: gorulmemis brand for AUTO katmani KAPATILSIN.

## 5. D7 OKUMA #2 KARARI
- kapiyi gecen blok sayisi: **0**
- this bloklarin total kazanci: **+0.0000** (gate +0.10)

**DECISION: D7 OKUNMAZ.** Kumulatif kazanc kapinin altinda; okuma HARCANMAZ. Butcede kalan okuma sayisi degismez.

> Kazanclar TOPLANARAK prediction edilir; gercek birlesik kazanc genellikle DAHA AZ olur (bloklar same hatalari duzeltir). Toplam yalnizca GATE kararidir, VAAT DEGILDIR.
