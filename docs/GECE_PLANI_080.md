# GECE PLANI — gorulmemis brand robot F1 0.80 hedefi (2026-08-11 gecesi)

## 0. Durust baslangic noktasi

| measurement | value |
|---|---|
| D7 robot (dagitilan) | 0.2980 |
| D7 robot (P6, okuma #1) | **0.3115** |
| D6 pool YONLU recall | 0.7264 (ceiling F1 0.8414) |
| CWT yonlu recall | 0.4415 (D7 GT'sinin %37'si) |
| Secicinin tavandan aldigi pay | ~%37 |
| Ayrilamazlik tavani (iki independent measurement) | ~0.86 |

**0.80'in aritmetigi:**
```
F1 0.80 ≈ recall 0.80 x precision 0.80
→ pool YONLU recall >= 0.88   (bugun 0.7264)
→ selector tavanin >= %91'ini alsin (bugun %37)
```

**Durust risk notu:** 0.80, ayrilamazlik tavanina (~0.86) very yakin. Havuz
tarafi large olcude olculmus kaldiraclarla ulasilabilir; asil belirsizlik
secicidedir ve orasi el yapimi oznitelikle not MIMARI degisiklikle acilir.
Gercekci bant this planla 0.55-0.70; 0.80 however F fazi (tepe-basi ag) beklenenin
ustunde cikarsa gelir. Bunu bastan yazmak, after sisik number vermekten iyidir.

---

## FAZ A — HAVUZU SONUNA KADAR AC (olculmus kaldiraclar, sadece yurutme)

Uc kaldirac da MEASURED, birlestirilmedi:

| kaldirac | olculen | durum |
|---|---|---|
| mesh olasilik esigi 0.50 -> 0.05 | CWT konum 0.5532 -> **0.7914** | hazir, corpus cikarimi gerek |
| direction yelpazesi 256 direction | NIT yonlu +**0.0676** | hatta baglandi (`YB_FAN`) |
| seyreltmeyi kaldir (2.0mm sinirsiz) | D6 konum 0.8713 -> **0.9768** | parametre |

**A1.** Korpusu UC kaldirac OPEN cikar: `P6_KAYNAK=012 YB_FAN=256
MESH_ESIK=0.05 P6_MESH_R=2.0 P6_MESH_MAX=600 P6_MESH_KAT=8`
**A2.** Yeni havuzun YONLU recall'unu D6'da ve brand bazinda olc.
**GATE A:** D6 yonlu recall >= 0.85 degilse pool arm kapanir, F fazina gecilir.

*Bedel:* secenek/part ~2500 -> ~6000. Rejim kapisi + kaskad bunu yonetmek
zorunda; yonetemezse precision coker ve A2 kapisi bunu gosterir.

---

## FAZ B — SECICIYI SIK (el yapimi, ucuz, olculebilir)

**B1. Zor negatif madenciligi.** OOF skorlarindan YUKSEK SKORLU FP'leri secip
training kumesini onlarla yeniden dengele. Su an negatifler RASTGELE ornekleniyor
ve model kolay negatiflerle doluyor.

**B2. Uc sinifli ayrim.** Ikili "CP mi not mi" yerine
`tel girisi / vida deligi / alet yuvasi`. FP kaynagini dogrudan modelle.
Etiket: GT'ye uzak but YUKSEK skorlu agizlar + yaricap/depth kurallariyla
zayif etiketleme.

**B3. Kafes adedinden beklenen number.** Adim x uzanimdan parcanin CP sayisini
prediction et; secimi tam o sayida kes. Esikten adet karari KOTUYDU but this
YAPISAL a prediction, ogrenilmis not.

**B4. Parca-ici oz-kalibrasyon (transduktif).** Parcanin kendi en guvenli
tespitlerinden yerel a siniflandirici kur, kalan adaylari ona per yeniden
sirala. Marka kaymasini part duzeyinde emer.

**B5. Simetri kisiti.** Klemenslerin ayna simetrisini bul (SVD + eslesme testi);
tahminlerin simetrik ciftler halinde gelmesini odullendir. Markadan independent,
guclu yapisal kisit.

**B6. Uc seed ensemble + sira ortalamasi.** Ucuz varyans azaltma.

**GATE B:** each biri `tam` brand katlarinda TEK DEGISKEN olculur; +0.01 altinda
kalan DAGITILMAZ.

---

## FAZ C — YENI FIZIKSEL BILGI (oznitelik uzayini genislet)

**C1. Derinlik profili imzasi.** Eksen boyunca 0.5/1/2/4/8/16mm'de serbest
yaricap olc -> 6 sutunluk profil. Tel girisi: huni after SABIT delik. Vida
deligi: konik daralma. Alet yuvasi: sig ve genis. Mevcut 9 mouth olcusu bunu
however kabaca yakaliyor.

**C2. B-rep topoloji grafigi.** Agza komsu yuzlerin TIPI (silindir/duzlem/koni),
sayisi, ic halka present mi, komsu silindirlerin es-eksenligi. Su an only
silindir yaricapi/axis kullaniliyor.

**C3. Kanonik hizalama.** Parcayi ray eksenine per kanonik cerceveye getir;
direction oznitelikleri this cercevede ifade edilsin. Bir noise boyutunu siler.

**GATE C:** each blok ayri olculur; +0.01 altinda kalan bloк ATILIR (oznitelik
sisirmek modeli bozar -- P6_GEO deneyi bunu gosterdi).

---

## FAZ D — MIMARI (asil 0.80 kaldiraci)

**D1. TEPE-BASI AG (Sutun 1).** Her tepe for:
`p(CP)` + 3B ofset (tepeden gercek CP'ye) + YON ALANI (birim vektor).
Veri: 3051 part / ~17k CP. Segmentasyon agi 71 parcayla egitilmisti; 43 fold
veriyle DOGRUDAN HEDEFE egitmek en large tek sans.
**GATE D1:** agin tek basina D6 pool recall'u (konum+direction) >= 0.90 degilse
omurga/etiket degistir.

**D2. ADAY-KUMESI TRANSFORMER (Sutun 2).** Adaylar token, part basina small
a transformer. Attention sira/adim/lattice yapisini KENDI kesfeder; NIT
teshisi "candidate-basina oznitelikle 12x'te tikaniyor" diyordu.
**GATE D2:** LOMO'da HGB'ye +0.05 otherwise mimariyi buyutme, veriye don.

**D3. KURESAL COZUMLEME (Sutun 3).** Adim/axis tahmini + izgara/adet
kisitiyla kuresel atama (ILP ya da yapilandirilmis acgozlu). Bugunku NMS bunun
ilkel hali.

---

## FAZ E — VERI

**E1. SIE +310 part** (`_ds1`, Graphic3d mesh'li). A-B/KLM/CWT/EFX **SINAVA
SAKLANIR**, egitime ASLA girmez.
**E2. Sentetik parametrik klemens ureteci.** Klemens geometrisi son derece
parametrik (sira x adim x yaricap x body). CP'leri kurulustan bilinen on
binlerce part -> tepe-basi agin on egitimi.
**E3. Oz-training.** Etiketsiz training-markasi STEP'lerinde high-precision sahte
etiket. RISKLI; before small pilot, kapili.

---

## YURUTME SIRASI VE KAPILAR

```
A1 corpus cikarimi (3 kaldirac acik)   ~2 sa   [kosuyor birakilir]
A2 pool tavani olcumu                 ~10 dk  GATE A: yonlu recall >= 0.85
B1 zor negatif                         ~40 dk  GATE B: +0.01
B3 lattice adedi                         ~20 dk  GATE B
B4 oz-kalibrasyon                      ~30 dk  GATE B
C1 depth profili                    ~1 sa   GATE C
B5 simetri                             ~40 dk  GATE B
C2 B-rep topoloji                      ~1 sa   GATE C
D1 tepe-basi ag pilotu                 ~2 sa   GATE D1: recall >= 0.90
```

**D7 BUTCESI: 2 OKUMA KALDI.** Gece boyunca D7'ye BAKILMAZ. Ancak `tam` brand
katlarinda kumulatif kazanc **+0.10'u gecerse** D7 okuma #2 yapilir; aksi halde
sabah rapor edilir ve okuma harcanmaz.

**DEGISMEZ KURALLAR**
1. Her arm TEK DEGISKEN olculur, `tam` MARKA KATLARINDA, MAKRO olcutle.
2. Dagitilan model paketi each kosuda KORUNUR/GERI YUKLENIR.
3. Kapatma karari verirken uc soru: sondanin cozunurlugu yeterli mi ·
   probe etkinin BEKLENDIGI yerde mi kosuldu · baseline o kumede already guclu muydu.
4. Sisik number YOK: gelistirme kumesinden okunan kazanc baseline gucuyle birlikte
   yazilir.
