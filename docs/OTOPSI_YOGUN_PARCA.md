# OTOPSI RAPORU — YOGUN PARCA DUVARI (NIT / CWT tipi)

Tarih: 2026-08-12 · Kume: d6 (gorulmemis marka kosulu) · D7'ye BAKILMADI

---

## 0. Olu ilan

Gorulmemis marka robot F1'i 0.50'nin uzerine cikaramamamizin sebebi tek bir
parca ailesidir. NIT tipi **yogun klemensler** (parca basina ~24 CP) d6'daki
GT'nin **%46'sini** tasir (1222 / 2660). Duvar oradadir.

Bu ailede **BES ayri mekanizma** tek tek denendi ve **besi de dustu**:

| # | mekanizma | olculen | sonuc |
|---|-----------|---------|-------|
| 1 | baglam modeli (S4, DeepSets) | −0.0194 / −0.0380 | KAPANDI |
| 2 | kafes yayilimi | konum 0.676 → uctan uca 0.077 | YON'de coktu |
| 3 | analitik yon (B-rep agzi) | agiz CP'de bulunma 0.011 | YOK |
| 4 | mesh normali | 0.334 (secili adaylarda) | OLU |
| 5 | adet kisiti | adet hatasi 18.0 | CURUDU |

---

## 1. Otopsinin birinci bulgusu: besi de ZINCIRIN SONUNDAYDI

Hicbiri zincirin **basina** bakmadi. Hepsi asagi akista, ayni secici skorunu
tuketerek calisiyordu:

```
  mesh -> SEGMENTASYON -> aday havuzu -> oznitelik -> SECICI -> secim -> yon
          ^^^^^^^^^^^^                               ^^^^^^
          hic bakilmadi                              bes mekanizma burada
```

**Bagimsiz bes deneme sanilan sey, tek bir denemenin bes tekrariydi.**

---

## 2. Ilk teshis ve NEDEN DUZELTILDI

`sonda_otopsi_segmentasyon.py` mesh tepelerinde olctu (412 parca):

| marka | GT'de olasilik | RASTGELE yuzeyde | oran | tepe AUC |
|-------|------|------|------|------|
| SUPU | 0.4815 | 0.0058 | 83× | 0.8395 |
| UPUN | 0.4448 | 0.0219 | 20× | 0.8636 |
| MOR | 0.2075 | 0.1289 | 1.6× | 0.5552 |
| **NIT** | **0.5244** | **0.4394** | **1.19×** | 0.6985 |

Ilk okumam: "NIT'te segmentasyon tum govdeyi boyuyor, kaynak bozuk".

**Bu okuma eksikti ve duzeltildi.** Sebebi, bugun bir kez daha yakalanan ayni
tuzak: **mesh TEPESINDE olculen sey ADAY duzeyinde gecerli degildir.** (Ayni
hatayi bugun mesh normalinde yapmistim: tepelerde 0.800, secili adaylarda
0.334.) Bu yuzden ayni sey aday duzeyinde ayrica olculdu.

---

## 3. Duzeltilmis teshis: darbogaz SECICI DEGIL, ADAY/GT ORANI

`sonda_aday_auc.py` — egitilmis secici, gorulmemis marka katlari,
**aday duzeyinde** (`results/aday_auc_d6.json`):

| marka | GT | n_aday | **auc_secici** | sira_ilk | sira_son | ilk-k orani | gereken auc |
|---|---|---|---|---|---|---|---|
| NIT | 1222 | **6322** | **0.8854** | 19 | 2222 | 0.049 | 0.9968 |
| SUPU | 539 | 1600 | **0.9696** | 1 | 27 | 0.561 | 0.9981 |
| UPUN | 366 | 2098 | **0.9877** | 1 | 75 | 0.800 | 0.9986 |
| MOR | 240 | **11555** | **0.9657** | 8 | 624 | 0.202 | 0.9997 |

**Secici kotu degil.** AUC 0.885–0.988. Bagliyan sey ayirt edicilik degil,
**aday sayisinin GT sayisina orani**:

- MOR'da ~9 CP icin **11.555 aday** var. AUC 0.9657 olsa bile beklenen
  "bir pozitifin ustundeki negatif sayisi" ≈ (1−0.9657)×11555 ≈ **396**;
  olculen ortanca son-dogru sirasi **624**. Ayni mertebe.
- NIT'te (1−0.8854)×6322 ≈ **725**; olculen **2222**. Ayni mertebe.
- UPUN'da (1−0.9877)×2098 ≈ **26**; olculen **75**. Ayni mertebe.

Yani **uc markada da uctan uca davranis, AUC ve aday sayisindan onceden
kestirilebiliyor.** Duvarin denklemi budur.

---

## 4. Mekanizma: her yanlis konuma 24 PIYANGO BILETI

`n_aday` **konum** sayisi degil **secenek** sayisidir. Her konum
`MAX_SEC = 24` yon secenegi uretir: 6322 ≈ **260 konum × 24 yon**.

Konum-GT orani 260/24 ≈ **11:1** — gayet yonetilebilir.
Secenek-GT orani 6322/24 ≈ **263:1** — yonetilemez.

Bugun konumlar **en yuksek skorlu secenekleriyle** siralaniyor. Bu, her
**yanlis** konuma 24 piyango bileti vermektir: yanlis bir konumun 24
denemeden birinde yuksek skor kapma olasiligi, dogru konumun tek gercek
sinyalini bastirir. Klasik coklu-karsilastirma sismesi.

Bu, bagimsiz bir bilmeceyi de cozer: **MAX_SEC 12→24 yonlu havuz recall'unu
0.5913→0.8755 yukseltti ama gerceklesen F1'i acmadi.** Cunku ayni degisiklik
tavani acarken yanlis konumlarin bilet sayisini da ikiye katladi.

---

## 5. Neden "ilkini buluyor, tekrarlarini bulamiyor"

NIT'te ilk dogru 19. sirada, sonuncusu 2222. sirada. Aciklama: bir tanesini
tepeye tasimak icin gurultunun bir kez lehe dusmesi yeter. Yirmi dordunu
birden tasimak icin yirmi dort kez ust uste lehe dusmesi gerekir.

Kafes yayiliminin konum uretip (0.676) uctan uca cokmesinin (0.077) sebebi
de aynidir: urettigi konumlarda yonu yine ayni sismis siralamadan sordu.

---

## 6. Neden bugune kadar gorulmedi

1. **Mikro toplama NIT'i gizledi** — GT'nin %46'si kendi cokusunu kendi
   agirligiyla seyreltiyor.
2. **Havuz tavani "iyi" gorunuyordu** (0.843). O tavan KONUM tavaniydi;
   secenek/GT orani hic tavan olarak okunmadi.
3. **Rastgele taban cizgisi hic konmadi** — GT'de 0.52 olcup "yuksek" demek,
   yanindaki 0.44'luk zemin olculmedigi surece anlamsizdir.
4. **AUC ile aday sayisi birlikte hic okunmadi** — ikisi ayri ayri
   "iyi/kotu" diye yorumlandi; belirleyici olan carpimlaridir.

---

## 7. Denenen ve DUSEN kaldiraclar (bu otopsiden sonra)

| kol | ilan edilen kapi | olculen | hukum |
|---|---|---|---|
| yerel karsitlik (p − yerel ortanca, R=2/5/10) | NIT AUC +0.05 | **−0.033** (en iyi) | DUSTU, blok yazilmadi |
| dik-yon kisiti (eksen siraya diktir) | bugunkuyu asmak | NIT +0.017, SUPU/UPUN'da DUSURUYOR | DUSTU |

Yerel karsitligin dusmesi bilgi verdi: alan yalnizca "yukari kaymis" degil,
**yerel olarak da duz**. Yani segmentasyonun uzamsal yapisinda kullanilmamis
bir rezerv yok.

---

## 8. Gecebilecek kaldirac: TOPLAMA KURALI

Otopsinin dogrudan sonucu. Piyango sismesi **modelden degil, KARAR
kuralindan** geliyor — yani yeniden egitim gerektirmiyor.

Konum skoru `max` yerine sisme yapmayan bir toplamayla kurulur:

```
  ortanca : medyan       -- piyangoyu tamamen sondurur
  ort     : ortalama
  q75     : 75. yuzdelik -- max ile ortanca arasi
  say     : skoru 0.5 ustunde olan secenek ORANI
  maxxort : max x ortalama
  ust2    : en yuksek IKI secenegin ortalamasi
```

Gerekce: **yanlis** konumda 24 skor rastgele dagilir (yuksek max, dusuk
ortanca); **dogru** konumda bircok secenek makul skor alir (yuksek ortanca).

**Neden bu kaldiracin sansi otekilerden yuksek:**
- Olculen patolojiye dogrudan nisan aliyor, tahmine degil.
- Model YENIDEN EGITILMIYOR; yalnizca karar kurali degisiyor.
- Marka kavrami tanimaz.
- Ayni anda hem NIT'i (263:1) hem MOR'u (1284:1) hedefliyor — d6 GT'sinin
  %55'i.

**Kapi (once ilan edildi):** mikro robot F1'de **+0.01**.
Sonda: `sonda_konum_toplama.py` → `results/konum_toplama_d6.json`

**Ikinci kaldirac (sirada):** isaretsiz eksen + "disari" isareti
(`sonda_bimodal_yon.py`). Gerekce: NIT'te yon kahini 0.593, gerceklesen
0.150. Klemens girisleri tek yonlu degildir; isaretli acida 180 derece tam
basarisizliktir, modal oylama parcanin yarisini ters isaretler. Bu, K2.1'in
`robot −0.0100` vermesini ve `dik_kipsel`in 0.064'e dusmesini birlikte
aciklar.

---

## 9. Bu otopsinin kollara etkisi

Asagidakiler "KAPANDI" degil, **"SISMIS SIRALAMA ALTINDA KAPANDI"** olarak
isaretlenir. Toplama kurali gecerse **yeniden denenmeleri gerekir**:

- baglam modeli (S4)
- kafes yayilimi
- adet kisiti
- zor negatif / focal kayip / sinif agirligi
- dik-yon kisiti

Bu, `docs/KAPANAN_KOLLAR_DENETIMI.md`'deki hukmun tekrari: **kapatma hukmu
sondanin kusuru olabilir.**
