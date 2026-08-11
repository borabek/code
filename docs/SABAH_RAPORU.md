# SABAH RAPORU -- 2026-08-12 01:53

Butun sayilar `tam` MARKA KATLARINDA (LOMO). D7 SINAVINA BAKILMADI. Manset metrik MIKRO robot F1.

## 1. KAPI A -- tam-acik havuzun yonlu recall'u
Henuz olculmedi (A2 fazi kosmadi).
- ONCEKI havuz (u25): yonlu recall 0.7352, F1 tavani 0.8474 -> kapi GECMEMISTI

## 2. EK OZNITELIK BLOKLARI (kapi +0.01)
Henuz makbuz yok.

## 3. KADEME2 KOL KIYASLARI
- **taban (u25)** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.2938
- **SIRA kapali** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.2938
- **SIRA acik** (results/_p6_oz_u25) secilen=P6 -> P6 0.3091, P6_KAFES 0.3041

## 4. GECE FAZLARI
- [00:14:14] BITTI: A1b -- 141 dosya

## 3b. SECICI VERIMLILIGI -- 0.50 nereden gelebilir?
- havuz F1 TAVANI (`tam`, mukemmel secici): **0.8474**
- GERCEKLESEN (P6 kolu): **0.3091**
- **secici verimliligi = 36.5%**

0.50'ye iki yoldan gidilebilir:
1. **Havuzla:** verimlilik sabit kalirsa tavanin **1.3707** olmasi gerekir  -> 1.0'i asiyor, TEK BASINA IMKANSIZ
2. **Seciciyle:** tavan sabit kalirsa verimliligin **59.0%** olmasi gerekir (1.62x iyilesme)

> Havuz kolu tek basina hedefe goturmuyor; SECICI kolu zorunlu. Bu, EK bloklarina ve aday-kumesi modeline (D2) verilen onceligi belirler.

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
