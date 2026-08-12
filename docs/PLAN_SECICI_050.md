# PLAN: GORULMEMIS MARKA ROBOT F1 0.32 -> 0.50

**Bugunku gercek sayi:** D7 sinavi **0.3115**, `tam` marka katlari **0.3091**
(taban) / **0.3195** (B1, henuz tek degiskenli degil). Hedef **0.50**.

## Aritmetik -- neyi degistirmemiz gerekiyor

F1 ~ (secici verimliligi) x (havuz tavani).

| | bugun | tavan-24 sonrasi | 0.50 icin gereken |
|---|---|---|---|
| havuz F1 tavani | 0.8474 | **0.9432** (olculdu) | - |
| gerceklesen F1 | 0.3091 | ? (olculuyor) | 0.50 |
| secici verimliligi | %36.5 | ? | **%53.0** |

**Tavan artik darbogaz DEGIL.** 0.9432 tavanda 0.50'ye ulasmak icin secicinin
verimliligi %36.5'ten %53'e cikmali. Bu **1.45x**'lik bir iyilesme ve tek bir
kaldiracla gelmez.

**Yeni risk:** tavan-24 secenek sayisini parca basina 3217 -> 4857 (1.51x)
cikardi. Daha cok secenek = daha cok CELDIRICI. Verimlilik OLDUGU YERDE
KALIRSA tavan kazanci F1'e tam donmez; bu yuzden asagidaki S0 olcumu once
gelir.

---

## S0. TAVAN-24 UCTAN UCA (kosuyor) -- her seyin on kosulu

`kos_tavan_uctan_uca.sh`, d6 marka katlari, tek degisken korpus.

- **Cikti:** tavan 12 vs 24 uctan uca robot F1 farki.
- **Karar:** fark **>= +0.02** ise tavan-24 urune alinir ve `tam` korpusunda
  tekrarlanir. **< 0** ise celdirici etkisi kazanci yiyor demektir; o zaman
  once **S1 (budama)** gelir, tavan ondan sonra acilir.
- d6 GELISTIRME kumesidir; sayi YON gosterir, manset degildir.

## S1. SECENEK BUDAMA (yeni, tavan-24'un dogal esi)

Tavan-24 dogru yonu havuza sokuyor ama yaninda ~1600 ek celdirici getiriyor.
Iki kademeli budama: UCUZ bir on-siralayici secenekleri parca basina ~1200'e
indirsin, pahali secici yalniz onlari gorsun.

- **Kapi:** budama sonrasi YONLU recall kaybi **<= 0.01** iken secenek sayisi
  **>= %40** azalmali.
- **Neden simdi:** bu kol tavan-24 OLMADAN anlamsizdi; celdirici sorunu yeni
  dogdu.

**Budamanin F1'e hangi YOLDAN dokundugu onemli.** Karar kurali GORELI
(parca-maksimumunun %85'i), yani DUSUK skorlu secenekleri atmak secimi
degistirmez -- onlar zaten secilmiyordu. Budamanin gercek mekanizmasi
EGITIMDEDIR: negatifler `NEG_KAT=6` ile rastgele orneklenıyor ve secenek
havuzu buyudukce ornek daha KOLAY negatiflerden olusuyor, model zayifliyor.

Bu, B1'in (zor negatif madenciligi) mekanizmasiyla **AYNI**. Dolayisiyla
S1 ve S3'un kazanclari BUYUK OLCUDE ORTUSUR; ikisini toplamak yanlis olur.
S1'in ayirt edici degeri, S5'i (konum genisletme) mumkun kilmasidir.

## S2. EK OZNITELIK BLOKLARI (kuyrukta, kismen kosuyor)

Tek degiskenli cerceve, kapi **+0.01**. Sirasiyla: `kanonik` (parcanin kendi
ekseni), `kume` (adaylar arasi rekabet), `topoloji` (es-eksenli aile),
`ozkalib`, `kafes_adet`, `simetri`, `derinlik`.

- **Kapi:** +0.01 alti blok ATILIR.
- **Gercekci beklenti:** gecen blok basina +0.01..+0.03, ve bloklar AYNI
  hatalari duzelttigi icin toplamlari toplanmaz. Uc blok gecerse ~+0.03..+0.05.
- **`kume` ayrica bir SONDADIR:** kazandirirsa S4 (aday-kumesi modeli)
  gerekcelenir, kazandirmazsa o kolun hipotezi zayiflar.

## S3. ZOR NEGATIF (B1) DOGRULAMASI

B1 +0.0104 verdi ama korpus VE yontem birlikte degisti. `tam3` TABAN kosusu
kuyrukta; gelince:

    taban(u25) -> taban(tam3) = KORPUS etkisi
    taban(tam3) -> B1(tam3)   = ZOR NEGATIF etkisi

- **Kapi:** zor negatif etkisi tek basina **>= +0.01** ise urune alinir.

## S4. ADAY-KUMESI MODELI (D2) -- yapisal kaldirac

Bugunku secici NOKTASAL: her secenegi tek basina puanliyor. Karar kurali ise
GORELI. Aradaki bosluk yapisaldir ve oznitelik eklemekle kapanmaz.

- Adaylar uzerinde dikkat (set transformer): parca icindeki tum secenekler
  BIRLIKTE puanlanir.
- **Kapi:** LOMO'da HGB'ye **+0.05**. Alti kalirsa kol kapanir.
- **On kosul:** S2'deki `kume` blogu pozitif olmali (ucuz sonda).

## S5. KONUM KAYBI (%10.1) -- tavan-24 sonrasi kalan tek havuz acigi

Tavan-24 ile yon kaybi NIT'te SIFIRLANDI; kalan kayip saf konum. Mesh esigi,
seyreltme yaricapi ve aday ust siniri bu kayba gore YENIDEN taranmali
(bugunku ayarlar yon darbogazi varken secilmisti).

- **Kapi:** konum recall **>= 0.95** (bugun 0.8989).

**Bu kolun tavani ZATEN OLCULU** (seyreltme taramasi, D6):

| seyreltme | konum recall | aday/parca |
|---|---|---|
| uzamsal 2.5mm, ust sinir 350 (BUGUNKU) | 0.8713 | 251 |
| **uzamsal 2.0mm, SINIRSIZ** | **0.9768** | 458 |

Yani konum kaybinin buyuk kismi ayarla kapanabiliyor. **Ama bedeli S1'i
ZORUNLU kiliyor:** 458 aday x ~17 secenek = parca basina ~7800 secenek
(bugun 4857). Budama olmadan bu kol celdirici sayisini patlatir.

**S1 -> S5 sirasi bu yuzden onemli:** once budama, sonra konum genisletme.
Ters sirada yapilirsa S5 muhtemelen NEGATIF olcer ve haksiz yere kapanir --
bu kampanyada "kapali kol aslinda sondanin kusuruydu" hatasi bir kez yasandi.

## S6. SAHA -- olculen zinciri urune bagla

Bunlar F1'i degistirmez ama URUNE deger tasir; F1 kollarindan BAGIMSIZ.

1. `export_robot_glb.py` -> `kanonik_zincir.urun_cikti` (tier alani ortak yere
   tasinmali; bugun `robot_cp.extract` icinde).
2. AUTO katmani: gorulmemis markada REVIEW BOS ve kesinlik 0.3471. Kalibre bir
   skor cikana kadar AUTO **kapatilmali**.

---

## Sira ve kapilar

| adim | ne zaman | kapi | gecmezse |
|---|---|---|---|
| S0 tavan uctan uca | KOSUYOR | >= +0.02 | once S1 |
| S1 budama | S0'dan sonra | recall kaybi <= 0.01, secenek -%40 | tavan-24 ertelenir |
| S2 EK bloklari | KOSUYOR | +0.01/blok | blok atilir |
| S3 zor negatif | taban gelince | >= +0.01 | atilir |
| S4 aday-kumesi | `kume` pozitifse | +0.05 | kol kapanir |
| S5 konum kaybi | S0 sonrasi | konum >= 0.95 | mevcut ayar kalir |
| S6 saha | bagimsiz | - | - |

## DURUST BEKLENTI

Toplam gercekci kazanc: tavan-24 uctan uca (?) + bloklar (~+0.03..0.05) +
zor negatif (~+0.01) = **~0.36-0.40**. **0.50'ye S4 (aday-kumesi modeli)
olmadan ulasilmasi olasi degil**, S4 ile de garanti degil.

**Kural degismedi:** D7 OKUMA #2 ancak `tam` katlarinda kumulatif **+0.10**
birikirse yapilir. Bugun birikim +0.0104 (ve o bile tek degiskenli degil).

---

# S0 SONUCU (2026-08-12 06:20) -- KAPI GECMEDI, PLAN DEGISTI

`results/_gece/UCTAN_UCA.log` · d6 marka katlari, tek degisken korpus.

| | tavan 12 | tavan 24 | fark |
|---|---|---|---|
| **robot** | 0.2991 | **0.2898** | **-0.0092** |
| recall | 0.2027 | 0.1915 | -0.0112 |
| kesinlik | 0.5706 | 0.5961 | +0.0255 |

Kat kat: UPUN +0.0227 · SUPU -0.0119 · NIT +0.0016 · MOR **-0.0506**.

**Tavan kazanci F1'e DONMUYOR.** Celdirici etkisi gercek: kesinlik yukseliyor
ama recall daha cok dusuyor. Onceden ilan edilen kural geregi **once S1
(budama), tavan ondan sonra**.

**KAYIT:** bu deney d6-only LOMO'dur; her kat yalnizca ~400 parca uzerinde
egitiliyor (gercek kurulumda 3051). Zayif model ek celdiricilerden DAHA COK
zarar gorur, yani -0.0092 tavan-24 aleyhine YANLI olabilir. Kol
KAPATILMIYOR; `tam` korpusunda tekrarlanacak (S0b).

## ASIL BULGU: secici verimliligi IKI KUTUPLU

| marka | havuz tavani | gerceklesen | **verimlilik** |
|---|---|---|---|
| UPUN | 0.9807 | 0.6422 | **%65.5** |
| SUPU | 0.9591 | 0.4560 | **%47.5** |
| MOR | 0.9297 | 0.0533 | **%5.7** |
| NIT | 0.9147 | 0.0049 | **%0.5** |

Markalar ya CALISIYOR (%47-66) ya da COKUYOR (%0.5-6). Arada bir sey yok.
NIT'te havuz cevabi TASIYOR (tavan 0.9147) ama secici bulamiyor.

**Ayni imza baska kumelerde de var:** `tam` katlarinda TOGI 0.1689 (recall
0.1054, kesinlik 0.4240 -- yani "az ama dogru"), D7'de CWT 0.0466. Yani bu,
d6'nin kucuk egitim kumesine bagli bir artefakt DEGIL; sistemin sureklilesen
kor noktasi.

## PLANIN CERCEVESI DEGISTI

Hedef **"ortalama verimliligi 1.45x artirmak" DEGIL**, "coken markalardaki
COKUSU durdurmak". Aritmetik bunu destekliyor: NIT d6 GT'sinin %46'si ve
bugun 0.005'te. NIT tek basina calisan markalarin seviyesine (%47) ciksaydi,
mikro F1 0.29'dan ~0.60'a cikardi -- tek bir markadan.

**Yeni birinci soru:** COKEN MARKA ile CALISAN MARKA arasindaki fark NE?
Bu, sonraki adimlarin (S1/S4) neye gore tasarlanacagini belirler ve
olculmeden hicbiri dogru hedeflenemez.

**S7 (YENI, ONCELIKLI): COKUS TESHISI.** NIT/MOR ile UPUN/SUPU arasinda
sistematik farki olc: parca basina CP sayisi, aday yogunlugu, skor dagilimi,
kural secimi, GT'nin havuzdaki skor sirasi. Kapi yok -- bu bir TESHIS, kol
degil; ciktisi sonraki kollarin hedefini belirler.
