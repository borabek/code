# Plan: tam-otomatik 0.85 + sahada 0.90 kaliteli sign — 2026-08-11

Iki plan tek belgede, kirpilmadan. Birincisi tam-otomatik robot F1'i 0.85'e
tasima kapasite plani; ikincisi robotun GERCEK DUNYADA kullandigi GLB
isaretlerinin 0.90 kalitede olmasini bugunden saglayan urun plani. Ikisi
birbirini besler: kapasite plani kapsamayi buyutur, confidence kapisi sahayi korur.

---

## BOLUM A — 0.85'in aritmetigi (bugunku olcumlerden)

```
F1 0.85 ≈ recall 0.85 × precision 0.85 dengesi
→ pool tavani ≥ 0.92 olmali        (bugun: 0.7264)
→ selector, tavaninin ≥ %92'sini almali (bugun: %56)
```

Yani 0.85 iki ayri devrim istiyor; ikisi de "biraz more oznitelik" with gelmez.
Bugunku mimarinin yapisal siniri su: el yapimi candidate pool + el yapimi 92 sutun
+ gradient boosting. Uc sutunlu plan:

### Sutun 1 — Havuzu oldur: dense tepe-basi prediction
Bugun olctuk: konum bilgisi mesh'te ZATEN VAR (tepe konum recall 0.977). O
zaman candidate uretme/eleme hatti yerine, ag HER TEPE for uc sey ogrensin:
- `p(CP)` isi haritasi
- 3B ofset vektoru (tepeden gercek CP'ye — 2mm lateral hatayi ag kendisi duzeltir)
- YON ALANI (birim vektor regresyonu — direction bankasinin ogrenilmis hali)

Bu, 2B'de CenterNet/keypoint tespitinin mesh karsiligi. 3051 part / ~17k CP
training verisi present — a keypoint agi for yeterli olcek. Segmentasyon agi 71
parcayla egitilmisti; 43 KATI veriyle DOGRUDAN HEDEFE egitmek en large tek
sans. Omurga da serbest: DiffusionNet sart not, Point Transformer sinifi a
model denenebilir.

### Sutun 2 — Siralayiciyi oldur: candidate-set transformer'i
NIT teshisi gosterdi: candidate-basina oznitelikle siralama 12x'te tikaniyor, because
karar PARCA-DUZEYI yapiya bagli (24 kontakli sira). HGB adaylari tek tek
goruyor. Dogrusu: adaylari TOKEN yap, part basina small a transformer kos —
attention, sira/adim/lattice yapisini kendisi kesfeder. Kafes ozniteliklerini
elle yazmak yerine modelin icine gomer. Yogun part problemi tam as budur.

### Sutun 3 — Kuresel cozumleme
Tahminleri tek tek esiklemek yerine part-duzeyi TUTARLI KUME sec: adim/axis
tahmini + "kontaklar izgarada, number katalogla tutarli" kisitiyla kuresel atama
(small a ILP ya da yapilandirilmis acgozlu). Bugunku NMS bunun ilkel hali.

### Veri arm (paralel)
- SENTETIK PARAMETRIK KLEMENS URETECI: klemens geometrisi son derece parametrik
  (sira × adim × yaricap × body). CP'leri kurulustan known on binlerce
  sentetik part + domain randomization → tepe-basi agin on egitimi.
- WARNING: `_ds1`'deki SIE/A-B/KLM/CWT stoku cazip but A-B/KLM/CWT/EFX D7 SINAV
  markalari — egitime katmak sinavi yakar. Ya only SIE kullanilir ya da o
  stok YENI BIR FINAL SINAVINA saklanir.

### Kapilar (sisik number none)
| gate | criterion | kesme |
|---|---|---|
| K1 | tepe-basi ag pool recall'u (konum+direction, D6) | >=0.90 degilse omurga/etiket degistir |
| K2 | set-transformer LOMO'da HGB'yi geciyor mu | +0.05 altiysa mimariyi buyutme, veriye don |
| K3 | uctan uca D7 (okuma #2/#3) | tek atis, onceden ilan |

DURUST RISK NOTU: eldeki iki independent ceiling olcumu (ayrilamazlik ~0.86, tam
pool ~0.87-0.92) 0.85'in TAVANIN DIBINDE oldugunu soyluyor — i.e. this plan
"mukemmele yakin" a sistem istiyor. 0.70-0.75 bandi this planla gercekci; 0.85
however Sutun 1 beklenenden iyi cikarsa gelir. Bunu bastan soylemek, after %90
deyip patlamaktan iyidir.

SIRA: D7 okumasi bitsin (mevcut kazanci muhurleyelim) → Sutun 1'in K1 pilotu
(2-3 gun esdegeri is) → K1 gecerse Sutun 2.

---

## BOLUM B — Sahada 0.90: confidence kapili GLB + brand carki

Saha akisi IKI REJIMIN KARISIMI — bildigimiz markadan yeni model + no
bilmedigimiz markadan yeni model. "0.90" sozunun hangi yarisi verilebilir:

**VERILEMEYECEK SOZ:** karisik akista tam otomatik tek number 0.90.
Bilinen-brand/yeni-model bugun ~0.60 olculu, unseen brand 0.2980 →
kampanyayla yukseliyor but tavani ~0.84. Bu ikisinin karisimindan tam otomatik
0.90 cikmaz; passing ayki sahte 90 muhtemelen tam this karisimi tek sisik sayiyla
ortmekten geldi.

**VERILEBILECEK SOZ — robotun kullandigi GLB'deki isaretin kalitesi 0.90+:**
uc katmanla:

1. **GUVEN KAPILI ISARETLEME (hemen):** GLB sozlesmesine confidence alani eklenir;
   isaretler iki sinif olur — `ONAYLI` (score esigi precision >=0.90 verecek
   sekilde D6/LOMO'da kalibre edilir; robot YALNIZ bunlari kullanir) ve
   `ONERI` (operatore/incelemeye duser). Durust maliyet metrigi KAPSAMADIR:
   "isaretlerin %X'i otomatik, o %X'in kesinligi >=0.90". Iki number AYRI
   raporlanir, asla tek 0.90'a karistirilmaz.

2. **MARKA CARKI:** unknown a markanin ilk parcalari geldiginde low
   kapsama with baslar, operator onaylari ETIKET olur, o brand egitime girer →
   "known"e doner → sonraki modelleri high F1 bandina gecer. Bu akista each
   brand a kez "yeni"dir; cark, unknown havuzunu surekli kucultur.

3. **KAPASITE PLANI (Bolum A, Sutun 1-3):** tam otomatik bandin kendisini
   yukseltir — dense tepe-basi ag + set-transformer + kuresel cozumleme. Bu,
   hem ONAYLI kapsamasini hem known-brand F1'ini buyutur.

MUHUR CUMLESI: *"Robotun kullandigi isaretler >=0.90 kesinliktedir; this
kalitede otomatik kapsama su an %X'tir ve carkla buyur."* Bu gercek a
0.90'dir ve patlamaz.

ILK ADIM: D7 okumasi bitince confidence-kapili GLB katmanini kurup X'i OLCMEK.
