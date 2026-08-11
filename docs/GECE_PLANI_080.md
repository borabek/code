# GECE PLANI — gorulmemis marka robot F1 0.80 hedefi (2026-08-11 gecesi)

## 0. Durust baslangic noktasi

| olcum | deger |
|---|---|
| D7 robot (dagitilan) | 0.2980 |
| D7 robot (P6, okuma #1) | **0.3115** |
| D6 havuz YONLU recall | 0.7264 (tavan F1 0.8414) |
| CWT yonlu recall | 0.4415 (D7 GT'sinin %37'si) |
| Secicinin tavandan aldigi pay | ~%37 |
| Ayrilamazlik tavani (iki bagimsiz olcum) | ~0.86 |

**0.80'in aritmetigi:**
```
F1 0.80 ≈ recall 0.80 x kesinlik 0.80
→ havuz YONLU recall >= 0.88   (bugun 0.7264)
→ secici tavanin >= %91'ini alsin (bugun %37)
```

**Durust risk notu:** 0.80, ayrilamazlik tavanina (~0.86) cok yakin. Havuz
tarafi buyuk olcude olculmus kaldiraclarla ulasilabilir; asil belirsizlik
secicidedir ve orasi el yapimi oznitelikle degil MIMARI degisiklikle acilir.
Gercekci bant bu planla 0.55-0.70; 0.80 ancak F fazi (tepe-basi ag) beklenenin
ustunde cikarsa gelir. Bunu bastan yazmak, sonra sisik sayi vermekten iyidir.

---

## FAZ A — HAVUZU SONUNA KADAR AC (olculmus kaldiraclar, sadece yurutme)

Uc kaldirac da OLCULDU, birlestirilmedi:

| kaldirac | olculen | durum |
|---|---|---|
| mesh olasilik esigi 0.50 -> 0.05 | CWT konum 0.5532 -> **0.7914** | hazir, korpus cikarimi gerek |
| yon yelpazesi 256 yon | NIT yonlu +**0.0676** | hatta baglandi (`YB_FAN`) |
| seyreltmeyi kaldir (2.0mm sinirsiz) | D6 konum 0.8713 -> **0.9768** | parametre |

**A1.** Korpusu UC kaldirac ACIK cikar: `P6_KAYNAK=012 YB_FAN=256
MESH_ESIK=0.05 P6_MESH_R=2.0 P6_MESH_MAX=600 P6_MESH_KAT=8`
**A2.** Yeni havuzun YONLU recall'unu D6'da ve marka bazinda olc.
**KAPI A:** D6 yonlu recall >= 0.85 degilse havuz kolu kapanir, F fazina gecilir.

*Bedel:* secenek/parca ~2500 -> ~6000. Rejim kapisi + kaskad bunu yonetmek
zorunda; yonetemezse kesinlik coker ve A2 kapisi bunu gosterir.

---

## FAZ B — SECICIYI SIK (el yapimi, ucuz, olculebilir)

**B1. Zor negatif madenciligi.** OOF skorlarindan YUKSEK SKORLU FP'leri secip
egitim kumesini onlarla yeniden dengele. Su an negatifler RASTGELE ornekleniyor
ve model kolay negatiflerle doluyor.

**B2. Uc sinifli ayrim.** Ikili "CP mi degil mi" yerine
`tel girisi / vida deligi / alet yuvasi`. FP kaynagini dogrudan modelle.
Etiket: GT'ye uzak ama YUKSEK skorlu agizlar + yaricap/derinlik kurallariyla
zayif etiketleme.

**B3. Kafes adedinden beklenen sayi.** Adim x uzanimdan parcanin CP sayisini
tahmin et; secimi tam o sayida kes. Esikten adet karari KOTUYDU ama bu
YAPISAL bir tahmin, ogrenilmis degil.

**B4. Parca-ici oz-kalibrasyon (transduktif).** Parcanin kendi en guvenli
tespitlerinden yerel bir siniflandirici kur, kalan adaylari ona gore yeniden
sirala. Marka kaymasini parca duzeyinde emer.

**B5. Simetri kisiti.** Klemenslerin ayna simetrisini bul (SVD + eslesme testi);
tahminlerin simetrik ciftler halinde gelmesini odullendir. Markadan bagimsiz,
guclu yapisal kisit.

**B6. Uc tohum ensemble + sira ortalamasi.** Ucuz varyans azaltma.

**KAPI B:** her biri `tam` marka katlarinda TEK DEGISKEN olculur; +0.01 altinda
kalan DAGITILMAZ.

---

## FAZ C — YENI FIZIKSEL BILGI (oznitelik uzayini genislet)

**C1. Derinlik profili imzasi.** Eksen boyunca 0.5/1/2/4/8/16mm'de serbest
yaricap olc -> 6 sutunluk profil. Tel girisi: huni sonra SABIT delik. Vida
deligi: konik daralma. Alet yuvasi: sig ve genis. Mevcut 9 agiz olcusu bunu
ancak kabaca yakaliyor.

**C2. B-rep topoloji grafigi.** Agza komsu yuzlerin TIPI (silindir/duzlem/koni),
sayisi, ic halka var mi, komsu silindirlerin es-eksenligi. Su an yalniz
silindir yaricapi/ekseni kullaniliyor.

**C3. Kanonik hizalama.** Parcayi ray eksenine gore kanonik cerceveye getir;
yon oznitelikleri bu cercevede ifade edilsin. Bir gurultu boyutunu siler.

**KAPI C:** her blok ayri olculur; +0.01 altinda kalan bloк ATILIR (oznitelik
sisirmek modeli bozar -- P6_GEO deneyi bunu gosterdi).

---

## FAZ D — MIMARI (asil 0.80 kaldiraci)

**D1. TEPE-BASI AG (Sutun 1).** Her tepe icin:
`p(CP)` + 3B ofset (tepeden gercek CP'ye) + YON ALANI (birim vektor).
Veri: 3051 parca / ~17k CP. Segmentasyon agi 71 parcayla egitilmisti; 43 kat
veriyle DOGRUDAN HEDEFE egitmek en buyuk tek sans.
**KAPI D1:** agin tek basina D6 havuz recall'u (konum+yon) >= 0.90 degilse
omurga/etiket degistir.

**D2. ADAY-KUMESI TRANSFORMER (Sutun 2).** Adaylar token, parca basina kucuk
bir transformer. Attention sira/adim/kafes yapisini KENDI kesfeder; NIT
teshisi "aday-basina oznitelikle 12x'te tikaniyor" diyordu.
**KAPI D2:** LOMO'da HGB'ye +0.05 yoksa mimariyi buyutme, veriye don.

**D3. KURESAL COZUMLEME (Sutun 3).** Adim/eksen tahmini + izgara/adet
kisitiyla kuresel atama (ILP ya da yapilandirilmis acgozlu). Bugunku NMS bunun
ilkel hali.

---

## FAZ E — VERI

**E1. SIE +310 parca** (`_ds1`, Graphic3d mesh'li). A-B/KLM/CWT/EFX **SINAVA
SAKLANIR**, egitime ASLA girmez.
**E2. Sentetik parametrik klemens ureteci.** Klemens geometrisi son derece
parametrik (sira x adim x yaricap x govde). CP'leri kurulustan bilinen on
binlerce parca -> tepe-basi agin on egitimi.
**E3. Oz-egitim.** Etiketsiz egitim-markasi STEP'lerinde yuksek-kesinlik sahte
etiket. RISKLI; once kucuk pilot, kapili.

---

## YURUTME SIRASI VE KAPILAR

```
A1 korpus cikarimi (3 kaldirac acik)   ~2 sa   [kosuyor birakilir]
A2 havuz tavani olcumu                 ~10 dk  KAPI A: yonlu recall >= 0.85
B1 zor negatif                         ~40 dk  KAPI B: +0.01
B3 kafes adedi                         ~20 dk  KAPI B
B4 oz-kalibrasyon                      ~30 dk  KAPI B
C1 derinlik profili                    ~1 sa   KAPI C
B5 simetri                             ~40 dk  KAPI B
C2 B-rep topoloji                      ~1 sa   KAPI C
D1 tepe-basi ag pilotu                 ~2 sa   KAPI D1: recall >= 0.90
```

**D7 BUTCESI: 2 OKUMA KALDI.** Gece boyunca D7'ye BAKILMAZ. Ancak `tam` marka
katlarinda kumulatif kazanc **+0.10'u gecerse** D7 okuma #2 yapilir; aksi halde
sabah rapor edilir ve okuma harcanmaz.

**DEGISMEZ KURALLAR**
1. Her kol TEK DEGISKEN olculur, `tam` MARKA KATLARINDA, MAKRO olcutle.
2. Dagitilan model paketi her kosuda KORUNUR/GERI YUKLENIR.
3. Kapatma karari verirken uc soru: sondanin cozunurlugu yeterli mi ·
   sonda etkinin BEKLENDIGI yerde mi kosuldu · taban o kumede zaten guclu muydu.
4. Sisik sayi YOK: gelistirme kumesinden okunan kazanc taban gucuyle birlikte
   yazilir.
