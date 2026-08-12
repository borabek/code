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

---

# S7 SONUCU -- COKUSUN SEBEBI: SKOR AYRIMI

Makbuz `results/cokus_teshisi_d6.json` · sonda `sonda_cokus.py` (OOF, d6).

| marka | CP/p | aday/p | secenek/p | havuzda | ilk10 | ilk50 | sira% | **poz-neg** | skor araligi | verimlilik |
|---|---|---|---|---|---|---|---|---|---|---|
| UPUN | 3.2 | 198 | 3074 | 0.949 | 0.939 | 0.974 | 0.001 | **0.847** | 0.981 | %65.5 |
| SUPU | 3.3 | 152 | 2248 | 0.934 | 0.804 | 0.908 | 0.016 | **0.665** | 0.901 | %47.5 |
| MOR | 3.4 | 453 | 9421 | 0.653 | 0.604 | 0.868 | 0.008 | **0.277** | 0.853 | %5.7 |
| NIT | 24.4 | 492 | 7470 | 0.898 | 0.380 | 0.760 | 0.005 | **0.050** | 0.651 | %0.5 |

## Cevap: POZITIF-NEGATIF SKOR AYRIMI

`poz-neg` (dogru seceneklerin medyan skoru eksi yanlislarinki) verimlilikle
**birebir ayni sirada** iniyor: 0.847 -> 0.665 -> 0.277 -> 0.050.

**NIT'te dogru secenek yanlistan yalnizca 0.05 daha yuksek puan aliyor.**
Model orada pratikte ayrim YAPMIYOR. Havuz cevabi tasiyor (0.898) ama skor
onu gostermiyor.

## Yogunluk SEBEP degil, CARPAN

`sira%` hepsinde cok kucuk (0.001-0.016): dogru secenek NIT'te bile ilk ~40
icinde. Ama NIT'te parca basina **24.4 CP** var -- bir tanesini tepede bulmak
yetmez, ~24'unu bulmak gerekir. Zayif ayrim + yuksek yogunluk carpimi cokusu
uretiyor. MOR yogun DEGIL (3.4) ama ayrimi da zayif (0.277) ve havuzu dusuk
(0.653) -- iki farkli yoldan ayni yere.

## Bunun kollara etkisi

- **S1 (budama) bu sorunu COZMEZ.** Dogru secenek zaten ilk ~40'ta; celdirici
  atmak ayrimi buyutmez. S1'in gerekcesi S5'i mumkun kilmakla sinirli kaldi.
- **S4 (aday-kumesi modeli) DOGRU HEDEFTE:** parca icinde secenekleri
  BIRBIRIYLE karsilastirmak, tam da zayif mutlak ayrimin oldugu yerde
  goreli bilgi uretir.
- **YENI S8: MARKA-ICI SKOR NORMALIZASYONU.** `skor araligi` NIT'te 0.651,
  UPUN'da 0.981 -- dagilimlar marka basina farkli olcekte. Goreli kural
  (%85 x parca-maks) bu olcege duyarli. Parca-ici z-skor / yuzdelik
  donusumu ucuz bir sondadir (`ozkalib` blogu tam bunu deniyor, kuyrukta).

## OLCUM TUTARSIZLIGI (acik, kapatilmadi)

Bu sondada MOR'un `havuzda` degeri **0.653**, oysa KAPI A olcumu ayni korpusta
MOR icin yonlu recall **0.8686** demisti. Iki olcum ayni kabul kutusunu
kullaniyor gorunuyor; fark aciklanmadi. Ihtimaller: (a) `yukle` ile
`sonda_havuz_tavani`in GT kayitlarini farkli okumasi, (b) eksen toleransinin
yuzdelik (`0.06 x diag`) vs sabit (40mm) uygulanmasi. **Cozulmeden MOR'un
havuz sayisi bu tablodan alintilanmamali.**

---

# S2 ILK SONUC: `ozkalib` COKTU (-0.0482)

Makbuz `results/ek_blok_ozkalib.json` (3051 parca, `tam` marka katlari).

| blok | bloksuz | blokla | fark | karar |
|---|---|---|---|---|
| ozkalib | 0.3124 | 0.2642 | **-0.0482** | **GECMEDI** |

`ozkalib` = parca-ici skor yuzdeligi + en yuksekten fark + medyandan fark.

**S8'IN UCUZ SONDASI BUYDU VE DUSTU.** "Marka-ici skor normalizasyonu"
hipotezi zayifladi: skoru parca icinde yeniden olceklendirmek modeli
IYILESTIRMIYOR, belirgin sekilde BOZUYOR. Muhtemel sebep: mutlak skor
duzeyinin kendisi bilgi tasiyor (kolay parca = yuksek skor) ve normalizasyon
bu bilgiyi siliyor.

**S8 kapatilmiyor ama onceligi DUSTU:** farkli bir normalizasyon (ornegin
yalniz olcek, konum degil) denenebilir; ancak once daha guclu kollar.

## KOL SAYIMI (bugune kadar, `tam` katlari)

| kol | sonuc |
|---|---|
| SIRA damgalama | 0.0000 (kapandi) |
| B6 topluluk | -0.0011 (kazandirmadi) |
| **ozkalib** | **-0.0482 (coktu)** |
| tavan-24 uctan uca | -0.0092 (d6, kapi gecmedi) |
| B1 zor negatif | +0.0104 (tek degiskenli DEGIL) |

**Bes koldan dordu negatif.** Bu, projenin taban oranina uygun: kollarin cogu
duser. Kalan bloklardan (kanonik, kume, topoloji, kafes_adet, simetri,
derinlik) gercekci beklenti 1-2 tanesinin +0.01..+0.03 vermesi.

## SON HEDEF TAHMINI (durust)

- **S4 olmadan:** 0.32 -> **~0.35-0.38**. Kalan bloklar bu tabloyu degistirmez.
- **S4 cokmeyi cozerse:** **0.45-0.50 mumkun.** Aritmetigi: D7'de CWT tek
  basina GT'nin %37'si ve F1'i 0.0466; calisan markalar ~0.45. Coken markalar
  calisan seviyeye cikarsa mikro F1 kollarin TOPLAMIYLA degil SICRAYARAK gelir.
- S4'un basari olasiligi durustce **%30-40**; kapisi +0.05.

**Bugun 0.50'yi vaat eden hicbir olcum YOK.** Elimizde olan, 0.50'ye giden tek
yolun hangisi oldugunu gosteren bir TESHIS var (S7: poz-neg skor ayrimi).

---

# S4 v1 SONUCU + `kume` BLOGU: BAGLAM HIPOTEZI ZAYIFLADI

## Uc bagimsiz kol, ayni yon

| kol | ne dener | sonuc |
|---|---|---|
| `ozkalib` | skoru parca icinde YENIDEN OLCEKLE | **-0.0482** |
| `kume` | adaylar arasi baglami ELLE ver (8 sutun) | **-0.0270** |
| S4 DeepSets | baglami OGRENEREK kullan | WEI -0.0100 · SIE -0.0336 · TOGI -0.0456 |

**Muhtemel sebep:** karar kurali ZATEN GORELI (parca-maksimumunun %85'i), yani
parca-ici normalizasyonu KURAL yapiyor. Baglami bir kez daha vermek ya tekrar
oluyor ya da gurultu/asiri uyum ekliyor.

**En can sikici ayrinti:** TOGI (coken marka, S4'un asil hedefi) DeepSets ile
DAHA DA KOTU (-0.0456). Yani kol, tasarlandigi yerde de calismadi.

## v2: TEK ve SON deneme (onceden ilan)

v1'de iki kol ESIT SINIF DENGESINDE yarismadi: HGB pozitif basina 6 negatifle
egitiliyor, DeepSets parcanin tamamini goruyordu (~300:1). BCE ortalama oldugu
icin negatifler kaybi boguyor. Bu **adalet kusuru**, metrige gore ayar degil.

Duzeltme: BCE yalnizca dengeli alt kumeden hesaplanir; ileri gecis ve listwise
TUM parca uzerinde kalir (baglam korunur).

**KURAL: v2 son denemedir.** Kapiyi (+0.05) gecmezse S4 KAPANIR. Gecene kadar
denemek, bu kampanyanin bastan beri kacindigi seydir.

## Eger S4 kapanirsa -- durust sonuc

0.50 icin gereken 1.45x'lik secici sicramasinin tek yapisal adayi S4 idi.
Kapanirsa elde kalan:

| kol | katki |
|---|---|
| kanonik (GECTI) | +0.0151 |
| zor negatif (dogrulanacak) | ~+0.01 |
| kalan bloklar (simetri/derinlik/kafes_adet) | belirsiz, tarihsel taban orani dusuk |

**Beklenen varis: 0.34-0.38.** 0.50'ye bu yoldan ULASILAMAZ ve bunu soylemek,
ulasilacakmis gibi davranmaktan iyidir. O noktada secenekler: (a) daha fazla
ve daha CESITLI veri, (b) farkli bir problem kurgusu (ornegin dogrudan
tepe-basi ag), (c) hedefi olculen gercege gore revize etmek.

---

# S3 SONUCU + DORDUNCU SESSIZ NO-OP: ZOR NEGATIF HIC DENENMEMIS

## Tek degiskenli taban kosusu ne buldu

| kosu | robot | recall | kesinlik | TP | FP | FN |
|---|---|---|---|---|---|---|
| u25 taban | 0.3091 | - | - | - | - | - |
| **tam3 TABAN (duz)** | **0.3195** | 0.2771 | 0.3772 | 3962 | 6543 | 10338 |
| **B1 (zor negatif)** | **0.3195** | 0.2771 | 0.3772 | 3962 | 6543 | 10338 |

**BIREBIR AYNI** -- TP/FP/FN'e kadar. Yani:

    KORPUS etkisi (u25 -> tam3) = +0.0104
    ZOR NEGATIF etkisi          =  0.0000

Daha once "B1 +0.0104" diye raporladigim kazanc **tamamen korpus
degisikligindendi**. Tek degiskenli taban kosusu olmasaydi bir NO-OP'u
kazanc diye urune yazacaktik.

## Kok neden (kod duzeyinde)

`kos_p6_kademe2.py` satir 430:

    s1_tr = [oof[i] for i in ic] if kf else None     # kf = (kol == "P6_KAFES")

`egit` icindeki sart ise:

    if ZORNEG and kol != "TABAN" and s1ler is not None:

**P6 kolunda `s1ler` HER ZAMAN None**, dolayisiyla ZORNEG hicbir zaman
devreye girmedi. Bayrak aciliyor, hicbir sey degismiyordu. Kontrollu testte
(80 parca, ayni tohum) ZORNEG=0 ve =1 **TP/FP/FN'e kadar birebir ayni** cikti.

**DUZELTME:** skorlar her kol icin zaten var ve `egit`/`skorla` onlari yalnizca
P6_KAFES dalinda oznitelik kurmak icin kullaniyor. Hepsine gecirmek guvenli.
Duzeltmeden sonra ayni testte ZORNEG=0 -> 0.4214, =1 -> 0.2500: **bayrak artik
etkili.**

## Durum

**Zor negatif CURUTULMEDI, HIC DENENMEDI.** Simdi gercekten olculebilir;
tam korpusta tek degiskenli kosu ile sinanmali. (Minik testteki dusus
baglayici degil: 80 parca, 30 iterasyon.)

## Bugunun dersi -- ayni sinif hata DORT kez

| # | sessiz no-op | nasil yakalandi |
|---|---|---|
| 1 | `str.replace` eslesmedi, sessizce hicbir sey yapmadi | grep ile dogrulama |
| 2 | bash fonksiyonu CAGRILDI ama TANIMLANMADI (koruma delindi) | bos RAM 0.9 GB'a dustu |
| 3 | yeni test `if __name__` blogundan SONRA tanimlandi, hic kosmadi | cikti hala 6 test gosteriyordu |
| 4 | `P6_ZORNEG` bayragi okundu ama kosul hep False | tek degiskenli taban kosusu |

**Ortak ders:** bir degisikligin ETKI ETTIGINI dogrulamadan "yapildi" deme.
Dordunun da maliyeti farkliydi ama dorduncusu en pahalisiydi: bir NO-OP'u
+0.0104 kazanc diye raporlamistim.
