# PLAN 0.75 — YON CEPHESI  ⚠️ ONERMESI CURUTULDU

> **2026-08-12 AKSAM — BU PLANIN OMURGASI YANLIS CIKTI.**
> Plan "0.75 = yon problemi" onermesine dayaniyordu. O onerme SECENEK
> sayisindan (6322 = 260 konum × 24 yon) TURETILMISTI, olculmemisti.
> Dogrudan olcum tersini soyledi (`results/konum_auc_d6.json`):
>
> | marka | konum AUC | yon AUC |
> |---|---|---|
> | NIT | **0.7053** | **0.8899** |
>
> Model yogun parcada **yonu biliyor, hangi acikligin kablo girisi oldugunu
> bilmiyor**. Yon kollari (A1 isin, B1 bimodal) olculdu: A1 dustu, B1 gercek
> ama uctan uca ~+0.002. Gecerli guzergah asagidaki **CEPHE C** (havuz
> kucultme) ve **CEPHE D** (veri); CEPHE A ve B kapandi.
>
> Ders: secenek sayisindan konum/yon payini TAHMIN ETME, OLC.
> Gecerli teshis: `docs/OTOPSI_YOGUN_PARCA.md` + [[konum-vs-yon-ayrimi]].



Tarih: 2026-08-12 · Hedef: gorulmemis marka **robot F1 = 0.75**
Bugunku en iyi gercek olcum: **D7 0.2773** · d6 0.3012

> Bu plan bir vaat degil, bir GUZERGAHTIR. Her madde ONCE ilan edilmis bir
> kapiya baglidir. Kapiyi gecmeyen kol yazilmaz, urune girmez, mansete
> cikmaz. D7 butcesinde **2 okuma** kaldi.

---

## 0. Planin dayandigi tek bulgu

`docs/OTOPSI_YOGUN_PARCA.md` — aday duzeyinde olculdu:

| marka | GT | n_aday | auc_secici | ilk-k orani | gereken auc |
|---|---|---|---|---|---|
| NIT | 1222 | 6322 | 0.8854 | 0.049 | 0.9968 |
| MOR | 240 | 11555 | 0.9657 | 0.202 | 0.9997 |
| SUPU | 539 | 1600 | 0.9696 | 0.561 | 0.9981 |
| UPUN | 366 | 2098 | 0.9877 | 0.800 | 0.9986 |

Ilk-k'nin dolmasi icin gereken aday sayisi ≈ `k / (1 − AUC)`:

| marka | su anki | gereken | oran |
|---|---|---|---|
| NIT | 6322 | ~209 | **30×** |
| MOR | 11555 | ~262 | **44×** |

**Ve fazlaligin tamami YON coklugundan geliyor:**

```
  NIT:  6322 secenek  =  ~260 KONUM  ×  24 YON        (MAX_SEC = 24)
        konum/GT = 11:1   IYI
        secenek/GT = 263:1  YONETILEMEZ
```

Yon konum basina TEK olsaydi: NIT 260/24 = 11:1 → gereken AUC **0.9968 → 0.91**.
Konum duzeyinde AUC muhtemelen orada.

> **0.75 sorusu konum sorusu degil. Tamamen YON sorusuna indirgenir.**

Bugun olculen 22 kolun 20'si skoru iyilestirmeye calisiyordu. Yanlis cephe.

---

## 1. Yon probleminin IKIYE bolunmesi

NIT'te: yon **kahini 0.593**, gerceklesen **0.150**.

| alt problem | olcu | deger | anlam |
|---|---|---|---|
| **(A) BULUNURLUK** | dogru yon 24 secenek arasinda var mi | **0.593** | %41'inde HIC YOK |
| **(B) SECIM** | varken bulabiliyor muyuz | 0.150/0.593 = **0.25** | 4'te 3'unu kaciriyoruz |

Ikisi ayri cephedir ve ayri kollarla dovusulur. **(A) cozulurse (B) kendiliginden
kucuur**, cunku secenek sayisi duser.

---

## 2. CEPHE A — YON BULUNURLUGU (ana cephe)

### A1 ★ ISIN ATMA / SERBEST YOL — en guclu denenmemis kol

**Fizik.** Bir kablo girisi bir DELIKTIR. Ekseni boyunca uzun ve engelsiz bir
kanal vardir; diger her yonde birkac mm'de duvara carpilir. O halde:

```
  eksen = govde icine EN DERIN nufuz eden yon
```

**Neden bugune kadar denenmedi:** yon hep bir SINIFLANDIRMA problemi gibi ele
alindi (24 secenek uret, skorla). Bir OLCUM problemi oldugu hic denenmedi.

**Neden diger yon kollarindan farkli:**
- B-rep agzi CP'de bulunma 0.011 → B-rep'e bagimli DEGIL
- mesh normali secili adaylarda 0.334 → yerel normale bagimli DEGIL
- yalnizca mesh ucgenleri ve isin-ucgen kesisimi gerekir (trimesh/embree)

**Cikti:** konum basina **1–3** yon. Havuz 24× kuculur.

**Kapi:** NIT'te tek-isin yonunun GT ile 10 derece icinde olma orani
**≥ 0.40** (bugunku secim 0.150). Sonda: `sonda_isin_ekseni.py`

**Riskler ve on tedbirler:**
- Mesh su gecirmez olmayabilir → `trimesh` `is_watertight` kontrolu, degilse
  cift yonlu isin (ileri+geri) ile derinlik
- rtree sessiz hatasi (bkz. `trimesh-rtree-silent-failure`) → duman testi SART
- Klemens govdesi ic bosluklu → derinlik **ilk carpma** degil, **toplam serbest
  yol** olarak olculur

### A2 SILINDIR OTURTMA (yerel)

Adayin cevresindeki 3–8 mm mesh yamasina silindir/koni oturt; ekseni al.
Delik duvari silindiriktir, en kucuk-kare eksen dogrudan cevaptir.
**Kapi:** A1 ile ayni. A1 dusrse bu kosulur (bagimsiz mekanizma).

### A3 DERINLIK ALANI GRADYANI

Isaretli mesafe alaninin adaydaki gradyani; delik icinde gradyan eksene
paraleldir. `derinlik` EK blogu zaten kosuyor — ciktisi yon icin de okunur.

### A4 MAX_SEC ARTIRIMI (kontrollu)

24 → 48 bulunurlugu yukseltir ama orani bozar. **Yalniz A1 gecerse ve yalniz
A1'in dusuk guvenli oldugu konumlarda** yedek olarak acilir. Tek basina
ACILMAZ — olculdu, tavani acip gerceklesenı acmiyor.

---

## 3. CEPHE B — YON SECIMI

### B1 ISARETSIZ EKSEN + DISARI ISARETI  *(KOSUYOR)*

Klemens girisleri tek yonlu degildir; isaretli acida 180 derece TAM
basarisizliktir. Modal oylama parcanin yarisini ters isaretler. Cozum: ekseni
isaretsiz sec, isareti "govdeden disari"dan turet (GT sozlesmesi 1.000 disari).
Iki olculmus cokusu birlikte aciklar: K2.1 `robot −0.0100`, `dik_kipsel` 0.064.
**Kapi:** NIT'te bugunkuyu +0.05 asmak. Sonda: `sonda_bimodal_yon.py`

### B2 YON'E OZEL SECICI (iki kademeli karar)

Bugun tek model hem konumu hem yonu puanliyor. Ayirmak:
1. konum modeli → konumlari sirala (aday/GT 11:1)
2. yon modeli → SECILEN konumlarda yonu sec
Ikinci modelin egitim dagilimi cok daha temiz olur.
**Kapi:** +0.01 mikro.

### B3 KOMSU TUTARLILIGI

Bir siradaki delikler paraleldir. Secilen konumlarda yonleri ortak
optimizasyonla duzlestir (parca ici yumusatma). B1 gecerse anlamli.

---

## 4. CEPHE C — HAVUZ KUCULTME (A1'in dogal devami)

A1 gecerse `MAX_SEC` 24 → 1–3 iner. Beklenen etki, otopsi denklemiyle:

| marka | n_aday (bugun) | n_aday (A1 sonrasi) | gereken AUC |
|---|---|---|---|
| NIT | 6322 | ~260 | 0.9968 → **0.908** |
| MOR | 11555 | ~480 | 0.9997 → **0.981** |

NIT icin gereken AUC elimizdekine (0.885) **cok yaklasiyor**. MOR icin hala
acik var → C2 gerekir.

### C2 KONUM ON-SUZGECI (yuksek recall)

MOR'da 480 konum / 9 GT = 53:1. Ucuz, yuksek recall'lu bir on-suzgec
(orn. yuzey disa bakma + minimum delik capi + `kanonik` blogu) konumlari
yariya indirirse gereken AUC 0.981 → 0.963'e iner.
**Kapi:** recall kaybi ≤ 0.01 karsiliginda konum sayisinda ≥ 2× kucultme.

---

## 5. CEPHE D — VERI (paralel, bagimsiz)

### D1 ★ SENTETIK KLEMENS URETECI
Marka kavrami tanimaz. Yogun parca ailesini (asil duvar) istenildigi kadar
uretir: kutup sayisi, adim, delik capi, giris acisi, iki-sirali/tek-sirali.
GT insaattan gelir, etiketleme hatasi sifirdir.
**Kapi:** +0.03 mikro. En buyuk tek veri kaldiraci.

### D2 SIE +310 PARCA
Ucuncu uretici stogunda kullanilabilir tek marka (A-B/KLM/CWT/EFX = D7 sinav
markalari, DOKUNULMAZ). Olculmus veri kaldiraci: +110 WEI → +0.0391.
**Kapi:** +0.01.

### D3 CESITLILIK AGIRLIKLANDIRMA
Egitimde parca basina agirlik: yogun parcalar CP sayilariyla dogru orantili
agirlik aliyor ve seyrekleri ezyor. Parca-esitleyici agirlik olculmedi.
**Kapi:** +0.01.

---

## 6. CEPHE E — OLCUM BUTUNLUGU (kapiyi acan degil, YANLIS ACMAYI onleyen)

- **E1 YON SOZLESMESI DENETIMI** — GT yonu markalar arasi tutarli mi. D7 dahil
  TUM yon olcumlerini etkiler. A1/B1 sonuclarini yorumlamadan ONCE bitmeli.
- **E2 GLB dogrulamasi** — `urun_p6`/`urun_genis` sahaya girmiyor
  (`glb-olculen-zinciri-kullanmiyor`). Kazanc olculse bile sahaya inmez.
- **E3 Sismis siralama altinda kapanan kollar** — S4, kafes yayilimi, adet
  kisiti, zor negatif, dik-yon. A1 gecerse **yeniden acilir**.

---

## 7. SIRA VE KAPILAR

```
  A1 ISIN EKSENI  ──kapi 0.40──┬── GECERSE ──> C1 havuz kucultme
       │                       │                    │
       │ dusrse                │                    v
       v                       │              E3 kapanan kollari YENIDEN AC
  A2 SILINDIR ──kapi 0.40──────┘                    │
       │                                            v
       │ ikisi de dusrse                      C2 konum on-suzgeci
       v
  B1/B2 secim kollari (kismi kazanc, 0.75'e YETMEZ)
```

**Paralel yurur (A'dan bagimsiz):** D1 sentetik uretec, D2 SIE, E1 sozlesme
denetimi.

**FINAL KAPI:** `tam` katlarinda kumulatif kazanc **≥ +0.10** olmadan D7
okunmaz. Bugunku birikim: **+0.0151**.

---

## 8. 0.75 HAKKINDA DURUST HUKUM

Elimdeki hicbir olcum bugun 0.75'i desteklemiyor. Bugun 22 kol olculdu,
en iyisi +0.0151. Ogrenme egrisi olcumu (`ogrenme-egrisi-fiyat-etiketi`)
0.80 icin **3.8×**, 0.85 icin **17.3×** korpus istiyor.

0.75'in gerceklesmesi icin **hepsi birden** dogru cikmali:
1. A1 (veya A2) yon olcumu calisir → havuz 24× kuculur
2. C2 konum on-suzgeci MOR tipini de menzile sokar
3. D1 sentetik uretec yogun aileyi doyurur
4. E3'te yeniden acilan kollar (S4, kafes, adet) sismis siralama kalkinca
   ise yarar

Bu dortlunun tamaminin tutmasi olasi degildir. **Ama A1 tutarsa denklemin
sekli ilk kez 0.75'e izin verir** — bu sabah izin vermiyordu. Guzergah budur;
sonuc kapilarda belli olacak.
