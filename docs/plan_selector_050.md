# PLAN: GORULMEMIS MARKA ROBOT F1 0.32 -> 0.50

**Bugunku gercek number:** D7 sinavi **0.3115**, `tam` brand katlari **0.3091**
(baseline) / **0.3195** (B1, henuz tek degiskenli not). Hedef **0.50**.

## Aritmetik -- neyi degistirmemiz gerekiyor

F1 ~ (selector verimliligi) x (pool tavani).

| | bugun | ceiling-24 sonrasi | 0.50 for gereken |
|---|---|---|---|
| pool F1 tavani | 0.8474 | **0.9432** (measured) | - |
| gerceklesen F1 | 0.3091 | ? (olculuyor) | 0.50 |
| selector verimliligi | %36.5 | ? | **%53.0** |

**Tavan residual darbogaz DEGIL.** 0.9432 tavanda 0.50'ye ulasmak for secicinin
verimliligi %36.5'ten %53'e cikmali. Bu **1.45x**'lik a iyilesme ve tek a
kaldiracla gelmez.

**Yeni risk:** ceiling-24 option sayisini part basina 3217 -> 4857 (1.51x)
cikardi. Daha very option = more very CELDIRICI. Verimlilik OLDUGU YERDE
KALIRSA ceiling kazanci F1'e tam donmez; this yuzden asagidaki S0 olcumu before
gelir.

---

## S0. CEILING-24 UCTAN UCA (kosuyor) -- each seyin on kosulu

`run_ceiling_end_to_end.sh`, d6 brand katlari, tek variable corpus.

- **Cikti:** ceiling 12 vs 24 uctan uca robot F1 farki.
- **Karar:** diff **>= +0.02** ise ceiling-24 urune alinir ve `tam` korpusunda
  tekrarlanir. **< 0** ise celdirici etkisi kazanci yiyor demektir; o zaman
  before **S1 (budama)** gelir, ceiling ondan after acilir.
- d6 GELISTIRME kumesidir; number YON gosterir, headline degildir.

## S1. SECENEK BUDAMA (yeni, ceiling-24'un dogal esi)

Tavan-24 correct direction havuza sokuyor but yaninda ~1600 ek celdirici getiriyor.
Iki kademeli budama: UCUZ a on-siralayici secenekleri part basina ~1200'e
indirsin, pahali selector only onlari gorsun.

- **Kapi:** budama sonrasi YONLU recall kaybi **<= 0.01** iken option sayisi
  **>= %40** azalmali.
- **Neden simdi:** this arm ceiling-24 OLMADAN anlamsizdi; celdirici sorunu yeni
  dogdu.

**Budamanin F1'e hangi YOLDAN dokundugu onemli.** Karar kurali GORELI
(part-maksimumunun %85'i), i.e. DUSUK skorlu secenekleri atmak secimi
degistirmez -- onlar already secilmiyordu. Budamanin gercek mekanizmasi
EGITIMDEDIR: negatifler `NEG_KAT=6` with rastgele orneklenıyor ve option
pool buyudukce ornek more KOLAY negatiflerden olusuyor, model zayifliyor.

Bu, B1'in (zor negatif madenciligi) mekanizmasiyla **AYNI**. Dolayisiyla
S1 ve S3'un kazanclari BUYUK OLCUDE ORTUSUR; ikisini toplamak wrong olur.
S1'in ayirt edici degeri, S5'i (konum genisletme) mumkun kilmasidir.

## S2. EK OZNITELIK BLOKLARI (kuyrukta, kismen kosuyor)

Tek degiskenli cerceve, gate **+0.01**. Sirasiyla: `kanonik` (parcanin kendi
axis), `cluster` (candidates arasi rekabet), `topoloji` (es-eksenli aile),
`ozkalib`, `kafes_adet`, `symmetry`, `depth`.

- **Kapi:** +0.01 alti blok ATILIR.
- **Gercekci beklenti:** passing blok basina +0.01..+0.03, ve bloklar AYNI
  hatalari duzelttigi for toplamlari toplanmaz. Uc blok gecerse ~+0.03..+0.05.
- **`cluster` also a SONDADIR:** kazandirirsa S4 (candidate-set modeli)
  gerekcelenir, kazandirmazsa o kolun hipotezi zayiflar.

## S3. ZOR NEGATIF (B1) DOGRULAMASI

B1 +0.0104 verdi but corpus VE yontem birlikte degisti. `tam3` BASELINE kosusu
kuyrukta; gelince:

    baseline(u25) -> baseline(tam3) = KORPUS etkisi
    baseline(tam3) -> B1(tam3)   = ZOR NEGATIF etkisi

- **Kapi:** zor negatif etkisi tek basina **>= +0.01** ise urune alinir.

## S4. ADAY-KUMESI MODELI (D2) -- yapisal kaldirac

Bugunku selector NOKTASAL: each secenegi tek basina puanliyor. Karar kurali ise
GORELI. Aradaki bosluk yapisaldir ve oznitelik eklemekle kapanmaz.

- Adaylar uzerinde dikkat (set transformer): part icindeki tum options
  BIRLIKTE puanlanir.
- **Kapi:** LOMO'da HGB'ye **+0.05**. Alti kalirsa arm kapanir.
- **On kosul:** S2'deki `cluster` blogu pozitif olmali (ucuz probe).

## S5. KONUM KAYBI (%10.1) -- ceiling-24 sonrasi remaining tek pool acigi

Tavan-24 with direction kaybi NIT'te SIFIRLANDI; remaining loss saf konum. Mesh esigi,
seyreltme yaricapi ve candidate ust siniri this kayba per YENIDEN taranmali
(bugunku settings direction darbogazi varken secilmisti).

- **Kapi:** konum recall **>= 0.95** (bugun 0.8989).

**Bu kolun tavani ZATEN OLCULU** (seyreltme taramasi, D6):

| seyreltme | konum recall | candidate/part |
|---|---|---|
| uzamsal 2.5mm, ust sinir 350 (BUGUNKU) | 0.8713 | 251 |
| **uzamsal 2.0mm, SINIRSIZ** | **0.9768** | 458 |

Yani konum kaybinin large kismi ayarla kapanabiliyor. **Ama bedeli S1'i
ZORUNLU kiliyor:** 458 candidate x ~17 option = part basina ~7800 option
(bugun 4857). Budama olmadan this arm celdirici sayisini patlatir.

**S1 -> S5 sirasi this yuzden onemli:** before budama, after konum genisletme.
Ters sirada yapilirsa S5 muhtemelen NEGATIF measures ve haksiz yere kapanir --
this kampanyada "kapali arm aslinda sondanin kusuruydu" hatasi a kez yasandi.

## S6. SAHA -- measured_path zinciri urune bagla

Bunlar F1'i degistirmez but URUNE value tasir; F1 kollarindan BAGIMSIZ.

1. `export_robot_glb.py` -> `canonical_chain.product_output` (tier alani ortak yere
   tasinmali; bugun `robot_cp.extract` icinde).
2. AUTO katmani: unseen markada REVIEW BOS ve precision 0.3471. Kalibre a
   score cikana up to AUTO **kapatilmali**.

---

## Sira ve kapilar

| adim | ne zaman | gate | gecmezse |
|---|---|---|---|
| S0 ceiling uctan uca | KOSUYOR | >= +0.02 | before S1 |
| S1 budama | S0'dan after | recall kaybi <= 0.01, option -%40 | ceiling-24 ertelenir |
| S2 EK bloklari | KOSUYOR | +0.01/blok | blok atilir |
| S3 zor negatif | baseline gelince | >= +0.01 | atilir |
| S4 candidate-set | `cluster` pozitifse | +0.05 | arm kapanir |
| S5 konum kaybi | S0 sonrasi | konum >= 0.95 | mevcut setting kalir |
| S6 field | independent | - | - |

## DURUST BEKLENTI

Toplam gercekci kazanc: ceiling-24 uctan uca (?) + bloklar (~+0.03..0.05) +
zor negatif (~+0.01) = **~0.36-0.40**. **0.50'ye S4 (candidate-set modeli)
olmadan ulasilmasi olasi not**, S4 with de garanti not.

**Kural degismedi:** D7 OKUMA #2 however `tam` katlarinda kumulatif **+0.10**
birikirse yapilir. Bugun birikim +0.0104 (ve o bile tek degiskenli not).

---

# S0 SONUCU (2026-08-12 06:20) -- GATE GECMEDI, PLAN DEGISTI

`results/_gece/UCTAN_UCA.log` · d6 brand katlari, tek variable corpus.

| | ceiling 12 | ceiling 24 | diff |
|---|---|---|---|
| **robot** | 0.2991 | **0.2898** | **-0.0092** |
| recall | 0.2027 | 0.1915 | -0.0112 |
| precision | 0.5706 | 0.5961 | +0.0255 |

Kat fold: UPUN +0.0227 · SUPU -0.0119 · NIT +0.0016 · MOR **-0.0506**.

**Tavan kazanci F1'e DONMUYOR.** Celdirici etkisi gercek: precision yukseliyor
but recall more very dusuyor. Onceden ilan edilen rule geregi **before S1
(budama), ceiling ondan after**.

**KAYIT:** this deney d6-only LOMO'dur; each fold yalnizca ~400 part uzerinde
egitiliyor (gercek kurulumda 3051). Zayif model ek celdiricilerden DAHA COK
zarar gorur, i.e. -0.0092 ceiling-24 aleyhine YANLI olabilir. Kol
KAPATILMIYOR; `tam` korpusunda tekrarlanacak (S0b).

## ASIL FINDING: selector verimliligi IKI KUTUPLU

| brand | pool tavani | gerceklesen | **verimlilik** |
|---|---|---|---|
| UPUN | 0.9807 | 0.6422 | **%65.5** |
| SUPU | 0.9591 | 0.4560 | **%47.5** |
| MOR | 0.9297 | 0.0533 | **%5.7** |
| NIT | 0.9147 | 0.0049 | **%0.5** |

Markalar ya CALISIYOR (%47-66) ya da COKUYOR (%0.5-6). Arada a sey none.
NIT'te pool cevabi TASIYOR (ceiling 0.9147) but selector bulamiyor.

**Ayni imza baska kumelerde de present:** `tam` katlarinda TOGI 0.1689 (recall
0.1054, precision 0.4240 -- i.e. "az but correct"), D7'de CWT 0.0466. Yani this,
d6'nin small training kumesine bagli a artefakt DEGIL; sistemin sureklilesen
kor noktasi.

## PLANIN CERCEVESI DEGISTI

Hedef **"mean verimliligi 1.45x artirmak" DEGIL**, "coken markalardaki
COKUSU durdurmak". Aritmetik bunu destekliyor: NIT d6 GT'sinin %46'si ve
bugun 0.005'te. NIT tek basina running markalarin seviyesine (%47) ciksaydi,
mikro F1 0.29'dan ~0.60'a cikardi -- tek a markadan.

**Yeni birinci soru:** COKEN MARKA with CALISAN MARKA arasindaki diff NE?
Bu, sonraki adimlarin (S1/S4) neye per tasarlanacagini belirler ve
olculmeden hicbiri correct hedeflenemez.

**S7 (YENI, ONCELIKLI): COKUS TESHISI.** NIT/MOR with UPUN/SUPU arasinda
sistematik farki olc: part basina CP sayisi, candidate yogunlugu, score dagilimi,
rule secimi, GT'nin havuzdaki score sirasi. Kapi none -- this a DIAGNOSIS, arm
not; ciktisi sonraki kollarin hedefini belirler.

---

# S7 SONUCU -- COKUSUN SEBEBI: SKOR AYRIMI

Makbuz `results/cokus_teshisi_d6.json` · probe `probe_cokus.py` (OOF, d6).

| brand | CP/p | candidate/p | option/p | havuzda | ilk10 | ilk50 | sira% | **poz-neg** | score araligi | verimlilik |
|---|---|---|---|---|---|---|---|---|---|---|
| UPUN | 3.2 | 198 | 3074 | 0.949 | 0.939 | 0.974 | 0.001 | **0.847** | 0.981 | %65.5 |
| SUPU | 3.3 | 152 | 2248 | 0.934 | 0.804 | 0.908 | 0.016 | **0.665** | 0.901 | %47.5 |
| MOR | 3.4 | 453 | 9421 | 0.653 | 0.604 | 0.868 | 0.008 | **0.277** | 0.853 | %5.7 |
| NIT | 24.4 | 492 | 7470 | 0.898 | 0.380 | 0.760 | 0.005 | **0.050** | 0.651 | %0.5 |

## Cevap: POZITIF-NEGATIF SKOR AYRIMI

`poz-neg` (correct seceneklerin medyan skoru eksi yanlislarinki) verimlilikle
**birebir same sirada** iniyor: 0.847 -> 0.665 -> 0.277 -> 0.050.

**NIT'te correct option yanlistan yalnizca 0.05 more high puan aliyor.**
Model orada pratikte ayrim YAPMIYOR. Havuz cevabi tasiyor (0.898) but score
onu gostermiyor.

## Yogunluk REASON not, CARPAN

`sira%` hepsinde very small (0.001-0.016): correct option NIT'te bile ilk ~40
icinde. Ama NIT'te part basina **24.4 CP** present -- a tanesini tepede bulmak
yetmez, ~24'unu bulmak gerekir. Zayif ayrim + high yogunluk carpimi cokusu
uretiyor. MOR dense DEGIL (3.4) but ayrimi da zayif (0.277) ve pool low
(0.653) -- iki different yoldan same yere.

## Bunun kollara etkisi

- **S1 (budama) this sorunu COZMEZ.** Dogru option already ilk ~40'ta; celdirici
  atmak ayrimi buyutmez. S1'in gerekcesi S5'i mumkun kilmakla sinirli kaldi.
- **S4 (candidate-set modeli) DOGRU HEDEFTE:** part icinde secenekleri
  BIRBIRIYLE karsilastirmak, tam da zayif mutlak ayrimin oldugu yerde
  goreli bilgi uretir.
- **YENI S8: MARKA-ICI SKOR NORMALIZASYONU.** `score araligi` NIT'te 0.651,
  UPUN'da 0.981 -- dagilimlar brand basina different olcekte. Goreli rule
  (%85 x part-maks) this olcege duyarli. Parca-ici z-score / yuzdelik
  donusumu ucuz a sondadir (`ozkalib` blogu tam bunu deniyor, kuyrukta).

## OLCUM TUTARSIZLIGI (acik, kapatilmadi)

Bu sondada MOR'un `havuzda` degeri **0.653**, oysa GATE A olcumu same korpusta
MOR for yonlu recall **0.8686** demisti. Iki measurement same kabul kutusunu
kullaniyor gorunuyor; diff aciklanmadi. Ihtimaller: (a) `yukle` with
`probe_pool_ceiling`in GT kayitlarini different okumasi, (b) axis toleransinin
yuzdelik (`0.06 x diag`) vs fixed (40mm) uygulanmasi. **Cozulmeden MOR'un
pool sayisi this tablodan alintilanmamali.**

---

# S2 ILK RESULT: `ozkalib` COKTU (-0.0482)

Makbuz `results/extra_blok_ozkalib.json` (3051 part, `tam` brand katlari).

| blok | bloksuz | blokla | diff | karar |
|---|---|---|---|---|
| ozkalib | 0.3124 | 0.2642 | **-0.0482** | **GECMEDI** |

`ozkalib` = part-ici score yuzdeligi + en yuksekten diff + medyandan diff.

**S8'IN UCUZ SONDASI BUYDU VE DUSTU.** "Marka-ici score normalizasyonu"
hipotezi zayifladi: skoru part icinde yeniden olceklendirmek modeli
IYILESTIRMIYOR, belirgin sekilde BOZUYOR. Muhtemel reason: mutlak score
duzeyinin kendisi bilgi tasiyor (kolay part = high score) ve normalizasyon
this bilgiyi siliyor.

**S8 kapatilmiyor but onceligi DUSTU:** different a normalizasyon (for example
only olcek, konum not) denenebilir; however before more guclu kollar.

## KOL SAYIMI (bugune up to, `tam` katlari)

| arm | sonuc |
|---|---|
| SIRA damgalama | 0.0000 (closed) |
| B6 ensemble | -0.0011 (kazandirmadi) |
| **ozkalib** | **-0.0482 (coktu)** |
| ceiling-24 uctan uca | -0.0092 (d6, gate gecmedi) |
| B1 zor negatif | +0.0104 (tek degiskenli DEGIL) |

**Bes koldan dordu negatif.** Bu, projenin baseline oranina eligible: kollarin cogu
duser. Kalan bloklardan (kanonik, cluster, topoloji, kafes_adet, symmetry,
depth) gercekci beklenti 1-2 tanesinin +0.01..+0.03 vermesi.

## SON TARGET TAHMINI (durust)

- **S4 olmadan:** 0.32 -> **~0.35-0.38**. Kalan bloklar this tabloyu degistirmez.
- **S4 cokmeyi cozerse:** **0.45-0.50 mumkun.** Aritmetigi: D7'de CWT tek
  basina GT'nin %37'si ve F1'i 0.0466; running markalar ~0.45. Coken markalar
  running seviyeye cikarsa mikro F1 kollarin TOPLAMIYLA not SICRAYARAK gelir.
- S4'un basari olasiligi durustce **%30-40**; kapisi +0.05.

**Bugun 0.50'yi vaat eden hicbir measurement YOK.** Elimizde which, 0.50'ye giden tek
yolun hangisi oldugunu gosteren a DIAGNOSIS present (S7: poz-neg score ayrimi).

---

# S4 v1 SONUCU + `cluster` BLOGU: BAGLAM HIPOTEZI ZAYIFLADI

## Uc independent arm, same direction

| arm | ne dener | sonuc |
|---|---|---|
| `ozkalib` | skoru part icinde YENIDEN OLCEKLE | **-0.0482** |
| `cluster` | candidates arasi baglami ELLE ver (8 sutun) | **-0.0270** |
| S4 DeepSets | baglami OGRENEREK kullan | WEI -0.0100 · SIE -0.0336 · TOGI -0.0456 |

**Muhtemel reason:** karar kurali ZATEN GORELI (part-maksimumunun %85'i), i.e.
part-ici normalizasyonu RULE yapiyor. Baglami a kez more vermek ya tekrar
oluyor ya da noise/asiri uyum ekliyor.

**En can sikici ayrinti:** TOGI (coken brand, S4'un asil hedefi) DeepSets with
DAHA DA KOTU (-0.0456). Yani arm, tasarlandigi yerde de calismadi.

## v2: TEK ve SON deneme (onceden ilan)

v1'de iki arm ESIT SINIF DENGESINDE yarismadi: HGB pozitif basina 6 negatifle
egitiliyor, DeepSets parcanin tamamini goruyordu (~300:1). BCE mean oldugu
for negatifler kaybi boguyor. Bu **adalet kusuru**, metrige per setting not.

Duzeltme: BCE yalnizca dengeli alt kumeden hesaplanir; ileri gecis ve listwise
TUM part uzerinde kalir (baglam korunur).

**RULE: v2 son denemedir.** Kapiyi (+0.05) gecmezse S4 KAPANIR. Gecene up to
denemek, this kampanyanin bastan beri kacindigi seydir.

## Eger S4 kapanirsa -- durust sonuc

0.50 for gereken 1.45x'lik selector sicramasinin tek yapisal adayi S4 idi.
Kapanirsa elde remaining:

| arm | katki |
|---|---|
| kanonik (GECTI) | +0.0151 |
| zor negatif (dogrulanacak) | ~+0.01 |
| remaining bloklar (symmetry/depth/kafes_adet) | ambiguous, tarihsel baseline ratio low |

**Beklenen varis: 0.34-0.38.** 0.50'ye this yoldan ULASILAMAZ ve bunu soylemek,
ulasilacakmis gibi davranmaktan iyidir. O noktada options: (a) more excess
ve more CESITLI data, (b) different a problem kurgusu (for example dogrudan
tepe-basi ag), (c) hedefi measured_path gercege per revize etmek.

---

# S3 SONUCU + DORDUNCU SESSIZ NO-OP: ZOR NEGATIF HIC DENENMEMIS

## Tek degiskenli baseline kosusu ne buldu

| kosu | robot | recall | precision | TP | FP | FN |
|---|---|---|---|---|---|---|
| u25 baseline | 0.3091 | - | - | - | - | - |
| **tam3 BASELINE (duz)** | **0.3195** | 0.2771 | 0.3772 | 3962 | 6543 | 10338 |
| **B1 (zor negatif)** | **0.3195** | 0.2771 | 0.3772 | 3962 | 6543 | 10338 |

**BIREBIR AYNI** -- TP/FP/FN'e up to. Yani:

    KORPUS etkisi (u25 -> tam3) = +0.0104
    ZOR NEGATIF etkisi          =  0.0000

Daha before "B1 +0.0104" diye raporladigim kazanc **tamamen corpus
degisikligindendi**. Tek degiskenli baseline kosusu olmasaydi a NO-OP'u
kazanc diye urune yazacaktik.

## Kok why (kod duzeyinde)

`run_p6_kademe2.py` satir 430:

    s1_tr = [oof[i] for i in ic] if kf else None     # kf = (arm == "P6_KAFES")

`egit` icindeki sart ise:

    if ZORNEG and arm != "BASELINE" and s1ler is not None:

**P6 kolunda `s1ler` HER ZAMAN None**, therefore ZORNEG hicbir zaman
devreye girmedi. Bayrak aciliyor, hicbir sey degismiyordu. Kontrollu testte
(80 part, same seed) ZORNEG=0 ve =1 **TP/FP/FN'e up to birebir same** cikti.

**DUZELTME:** skorlar each arm for already present ve `egit`/`skorla` onlari yalnizca
P6_KAFES dalinda oznitelik kurmak for kullaniyor. Hepsine gecirmek guvenli.
Duzeltmeden after same testte ZORNEG=0 -> 0.4214, =1 -> 0.2500: **bayrak residual
etkili.**

## Durum

**Zor negatif CURUTULMEDI, HIC DENENMEDI.** Simdi gercekten olculebilir;
tam korpusta tek degiskenli kosu with sinanmali. (Minik testteki dusus
baglayici not: 80 part, 30 iterasyon.)

## Bugunun dersi -- same sinif error DORT kez

| # | sessiz no-op | nasil yakalandi |
|---|---|---|
| 1 | `str.replace` eslesmedi, sessizce hicbir sey yapmadi | grep with dogrulama |
| 2 | bash fonksiyonu CAGRILDI but TANIMLANMADI (koruma delindi) | bos RAM 0.9 GB'a dustu |
| 3 | yeni test `if __name__` blogundan SONRA tanimlandi, no kosmadi | cikti hala 6 test gosteriyordu |
| 4 | `P6_ZORNEG` bayragi okundu but kosul hep False | tek degiskenli baseline kosusu |

**Ortak ders:** a degisikligin ETKI ETTIGINI dogrulamadan "yapildi" deme.
Dordunun da maliyeti farkliydi but dorduncusu en pahalisiydi: a NO-OP'u
+0.0104 kazanc diye raporlamistim.
