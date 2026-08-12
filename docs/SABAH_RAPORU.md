# SABAH RAPORU -- 2026-08-12 05:23

Butun sayilar `tam` MARKA KATLARINDA (LOMO). D7 SINAVINA BAKILMADI. Manset metrik MIKRO robot F1.

## 1. KAPI A -- tam-acik havuzun yonlu recall'u
| havuz | kume | konum | **yonlu** | F1 tavani | aday/parca | KAPI A |
|---|---|---|---|---|---|---|
| tam-acik, tavan 12 | d6 | 0.8997 | **0.7347** | 0.8470 | 266 | gecmedi |
| onceki havuz | d6 | 0.8713 | **0.7264** | 0.8415 | 251 | gecmedi |
| onceki havuz | tam | 0.9789 | **0.7352** | 0.8474 | 239 | gecmedi |

KAPI A esigi: yonlu recall >= 0.85.

Kapi gecmezse havuz genisletme kolu KAPANIR: tavan yetmiyorsa secici ne kadar iyilesirse iyilessin hedefe ulasilamaz. TAVAN, mukemmel bir secicinin alacagi F1'dir -- VAAT DEGIL, UST SINIR.

## 1b. SECENEK TAVANI (MAX_SEC) BAGLIYOR MU?
**orneklem: ? / 18 parca, yelpaze 256**

| tavan | yonlu recall | secenek maliyeti |
|---|---|---|
| 12 | 0.5913 | 1.00x |
| 24 | 0.8755 | 1.33x |
| 48 | 0.8755 | 1.35x |

**orneklem: ? / 30 parca, yelpaze 256**

| tavan | yonlu recall | secenek maliyeti |
|---|---|---|
| 12 | 0.8462 | 1.00x |
| 24 | 0.9077 | 1.20x |
| 40 | 0.9077 | 1.23x |

> Bugunku tavan **12**. Tavan bagliyorsa yon kaynagi eklemek (yelpaze cozunurlugu) recall'u ARTIRMAZ -- yeni yonler tavana takilip mevcutlarin yerini alir. `YB_MAX_SEC` ile ayarlanir.

## 2. EK OZNITELIK BLOKLARI (kapi +0.01)
Henuz makbuz yok.

## 3. KADEME2 KOL KIYASLARI
- **taban (u25)** (results/_p6_oz_tam3) secilen=P6 -> P6 0.3195
- **SIRA kapali** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.2938
- **SIRA acik** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.3041

## 3b. SECICI VERIMLILIGI -- 0.50 nereden gelebilir?
- havuz F1 TAVANI (`tam`, mukemmel secici): **0.8474**
- GERCEKLESEN (P6 kolu): **0.3195**
- **secici verimliligi = 37.7%**

0.50'ye iki yoldan gidilebilir:
1. **Havuzla:** verimlilik sabit kalirsa tavanin **1.3263** olmasi gerekir  -> 1.0'i asiyor, TEK BASINA IMKANSIZ
2. **Seciciyle:** tavan sabit kalirsa verimliligin **59.0%** olmasi gerekir (1.57x iyilesme)

> Havuz kolu tek basina hedefe goturmuyor; SECICI kolu zorunlu. Bu, EK bloklarina ve aday-kumesi modeline (D2) verilen onceligi belirler.

## 4. GECE FAZLARI
- [00:14:14] BITTI: A1b -- 141 dosya
- [04:12:23] BITTI: A1b -- 3051 dosya
- [04:12:31] BITTI: A2_havuz_tavani (8s)
- [04:58:24] BITTI: B1_zor_negatif (2752s)

## 4b. SAHA -- AUTO KATMANI (tier cokusu)
Dagitilan AUTO esigi = **0.6**
- `d7_p6`: AUTO payi **1.0000**, kesinlik **0.3471**  <- REVIEW KATMANI BOS
- `d7_taban`: **OLCULMEMIS_varsayilan_dolgu** (2001 isaretin hepsi ayni skor)

> Gorulmemis markada robot HER isarete otonom guveniyor. Esigi yukseltmek kurtarmiyor (0.95'te bile kesinlik ~0.47). Oneri: gorulmemis marka icin AUTO katmani KAPATILSIN.

## 5. D7 OKUMA #2 KARARI
- kapiyi gecen blok sayisi: **0**
- bu bloklarin toplam kazanci: **+0.0000** (kapi +0.10)

**KARAR: D7 OKUNMAZ.** Kumulatif kazanc kapinin altinda; okuma HARCANMAZ. Butcede kalan okuma sayisi degismez.

> Kazanclar TOPLANARAK tahmin edilir; gercek birlesik kazanc genellikle DAHA AZ olur (bloklar ayni hatalari duzeltir). Toplam yalnizca KAPI kararidir, VAAT DEGILDIR.
