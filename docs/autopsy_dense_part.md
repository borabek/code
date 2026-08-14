# OTOPSI RAPORU — YOGUN PARCA DUVARI (NIT / CWT tipi)

Tarih: 2026-08-12 · Kume: d6 (unseen brand kosulu) · D7'ye BAKILMADI

---

## 0. Olu ilan

Gorulmemis brand robot F1'i 0.50'nin uzerine cikaramamamizin sebebi tek a
part ailesidir. NIT tipi **dense klemensler** (part basina ~24 CP) d6'daki
GT'nin **%46'sini** tasir (1222 / 2660). Duvar oradadir.

Bu ailede **BES ayri mekanizma** tek tek denendi ve **besi de dustu**:

| # | mekanizma | measured_path | sonuc |
|---|-----------|---------|-------|
| 1 | baglam modeli (S4, DeepSets) | −0.0194 / −0.0380 | CLOSED |
| 2 | lattice yayilimi | konum 0.676 → uctan uca 0.077 | YON'de coktu |
| 3 | analitik direction (B-rep mouth) | mouth CP'de bulunma 0.011 | YOK |
| 4 | mesh normali | 0.334 (secili adaylarda) | OLU |
| 5 | count kisiti | count hatasi 18.0 | CURUDU |

---

## 1. Otopsinin birinci bulgusu: besi de ZINCIRIN SONUNDAYDI

Hicbiri zincirin **basina** bakmadi. Hepsi asagi akista, same selector skorunu
tuketerek calisiyordu:

```
  mesh -> SEGMENTASYON -> candidate pool -> oznitelik -> SECICI -> secim -> direction
          ^^^^^^^^^^^^                               ^^^^^^
          no bakilmadi                              bes mekanizma burada
```

**Bagimsiz bes deneme sanilan sey, tek a denemenin bes tekrariydi.**

---

## 2. Ilk teshis ve WHY DUZELTILDI

`probe_autopsy_segmentation.py` mesh tepelerinde olctu (412 part):

| brand | GT'de probability | RASTGELE yuzeyde | ratio | tepe AUC |
|-------|------|------|------|------|
| SUPU | 0.4815 | 0.0058 | 83× | 0.8395 |
| UPUN | 0.4448 | 0.0219 | 20× | 0.8636 |
| MOR | 0.2075 | 0.1289 | 1.6× | 0.5552 |
| **NIT** | **0.5244** | **0.4394** | **1.19×** | 0.6985 |

Ilk okumam: "NIT'te segmentasyon tum govdeyi boyuyor, source bozuk".

**Bu okuma eksikti ve duzeltildi.** Sebebi, bugun a kez more yakalanan same
tuzak: **mesh TEPESINDE measured_path sey ADAY duzeyinde valid degildir.** (Ayni
hatayi bugun mesh normalinde yapmistim: tepelerde 0.800, secili adaylarda
0.334.) Bu yuzden same sey candidate duzeyinde also measured.

---

## 3. Duzeltilmis teshis: darbogaz SECICI DEGIL, ADAY/GT ORANI

`probe_candidate_auc.py` — egitilmis selector, unseen brand katlari,
**candidate duzeyinde** (`results/candidate_auc_d6.json`):

| brand | GT | n_aday | **auc_secici** | sira_ilk | sira_son | ilk-k ratio | gereken auc |
|---|---|---|---|---|---|---|---|
| NIT | 1222 | **6322** | **0.8854** | 19 | 2222 | 0.049 | 0.9968 |
| SUPU | 539 | 1600 | **0.9696** | 1 | 27 | 0.561 | 0.9981 |
| UPUN | 366 | 2098 | **0.9877** | 1 | 75 | 0.800 | 0.9986 |
| MOR | 240 | **11555** | **0.9657** | 8 | 624 | 0.202 | 0.9997 |

**Secici kotu not.** AUC 0.885–0.988. Bagliyan sey ayirt edicilik not,
**candidate sayisinin GT sayisina ratio**:

- MOR'da ~9 CP for **11.555 candidate** present. AUC 0.9657 olsa bile beklenen
  "a pozitifin ustundeki negatif sayisi" ≈ (1−0.9657)×11555 ≈ **396**;
  measured_path median son-correct sirasi **624**. Ayni mertebe.
- NIT'te (1−0.8854)×6322 ≈ **725**; measured_path **2222**. Ayni mertebe.
- UPUN'da (1−0.9877)×2098 ≈ **26**; measured_path **75**. Ayni mertebe.

Yani **uc markada da uctan uca davranis, AUC ve candidate sayisindan onceden
kestirilebiliyor.** Duvarin denklemi budur.

---

## 4. Mekanizma: each wrong konuma 24 PIYANGO BILETI

`n_aday` **konum** sayisi not **option** sayisidir. Her konum
`MAX_SEC = 24` direction secenegi uretir: 6322 ≈ **260 konum × 24 direction**.

Konum-GT ratio 260/24 ≈ **11:1** — gayet yonetilebilir.
Secenek-GT ratio 6322/24 ≈ **263:1** — yonetilemez.

Bugun konumlar **en high skorlu secenekleriyle** siralaniyor. Bu, each
**wrong** konuma 24 piyango bileti vermektir: wrong a konumun 24
denemeden birinde high score kapma olasiligi, correct konumun tek gercek
sinyalini bastirir. Klasik coklu-comparison sismesi.

Bu, independent a bilmeceyi de cozer: **MAX_SEC 12→24 yonlu pool recall'unu
0.5913→0.8755 yukseltti but gerceklesen F1'i acmadi.** Cunku same degisiklik
tavani acarken wrong konumlarin bilet sayisini da ikiye katladi.

---

## 5. Neden "ilkini buluyor, tekrarlarini bulamiyor"

NIT'te ilk correct 19. sirada, sonuncusu 2222. sirada. Aciklama: a tanesini
tepeye tasimak for gurultunun a kez lehe dusmesi yeter. Yirmi dordunu
birden tasimak for yirmi dort kez ust uste lehe dusmesi gerekir.

Kafes yayiliminin konum uretip (0.676) uctan uca cokmesinin (0.077) sebebi
de aynidir: urettigi konumlarda direction yine same sismis siralamadan sordu.

---

## 6. Neden bugune up to gorulmedi

1. **Mikro toplama NIT'i gizledi** — GT'nin %46'si kendi cokusunu kendi
   agirligiyla seyreltiyor.
2. **Havuz tavani "iyi" gorunuyordu** (0.843). O ceiling KONUM tavaniydi;
   option/GT ratio no ceiling as okunmadi.
3. **Rastgele baseline cizgisi no konmadi** — GT'de 0.52 olcup "high" demek,
   yanindaki 0.44'luk zemin olculmedigi surece anlamsizdir.
4. **AUC with candidate sayisi birlikte no okunmadi** — ikisi ayri ayri
   "iyi/kotu" diye yorumlandi; belirleyici which carpimlaridir.

---

## 7. Denenen ve DUSEN kaldiraclar (this otopsiden after)

| arm | ilan edilen gate | measured_path | verdict |
|---|---|---|---|
| yerel karsitlik (p − yerel median, R=2/5/10) | NIT AUC +0.05 | **−0.033** (en iyi) | DUSTU, blok yazilmadi |
| dik-direction kisiti (axis siraya diktir) | bugunkuyu asmak | NIT +0.017, SUPU/UPUN'da DUSURUYOR | DUSTU |

Yerel karsitligin dusmesi bilgi verdi: alan yalnizca "yukari kaymis" not,
**yerel as da duz**. Yani segmentasyonun uzamsal yapisinda kullanilmamis
a rezerv none.

---

## 8. Gecebilecek kaldirac: TOPLAMA KURALI

Otopsinin dogrudan sonucu. Piyango sismesi **modelden not, DECISION
kuralindan** geliyor — i.e. yeniden training gerektirmiyor.

Konum skoru `max` yerine sisme yapmayan a toplamayla kurulur:

```
  median : medyan       -- piyangoyu tamamen sondurur
  ort     : mean
  q75     : 75. yuzdelik -- max with median arasi
  say     : skoru 0.5 ustunde which option ORANI
  maxxort : max x mean
  ust2    : en high IKI secenegin ortalamasi
```

Gerekce: **wrong** konumda 24 score rastgele dagilir (high max, low
median); **correct** konumda bircok option makul score alir (high median).

**Neden this kaldiracin sansi otekilerden high:**
- Olculen patolojiye dogrudan nisan aliyor, tahmine not.
- Model YENIDEN EGITILMIYOR; yalnizca karar kurali degisiyor.
- Marka kavrami tanimaz.
- Ayni anda hem NIT'i (263:1) hem MOR'u (1284:1) hedefliyor — d6 GT'sinin
  %55'i.

**Kapi (before ilan edildi):** mikro robot F1'de **+0.01**.
Sonda: `probe_position_toplama.py` → `results/position_toplama_d6.json`

**Ikinci kaldirac (sirada):** unsigned axis + "disari" isareti
(`probe_bimodal_direction.py`). Gerekce: NIT'te direction kahini 0.593, gerceklesen
0.150. Klemens girisleri tek yonlu degildir; signed acida 180 derece tam
basarisizliktir, modal oylama parcanin yarisini ters isaretler. Bu, K2.1'in
`robot −0.0100` vermesini ve `dik_kipsel`in 0.064'e dusmesini birlikte
aciklar.

---

## 9. Bu otopsinin kollara etkisi

Asagidakiler "CLOSED" not, **"SISMIS SIRALAMA ALTINDA CLOSED"** as
isaretlenir. Toplama kurali gecerse **yeniden denenmeleri gerekir**:

- baglam modeli (S4)
- lattice yayilimi
- count kisiti
- zor negatif / focal loss / sinif agirligi
- dik-direction kisiti

Bu, `docs/KAPANAN_KOLLAR_DENETIMI.md`'deki hukmun tekrari: **kapatma hukmu
sondanin kusuru olabilir.**
