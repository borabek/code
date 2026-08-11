# OTOPSI RAPORU — GLB gorsellerindeki "yanlis CP" sikayeti

**Tarih:** 2026-08-04 · **Numune:** 3 parca, hicbiri (kendisi ve geometri ikizi) egitimde
gorulmemis · **Yontem:** `otopsi_cp.py`, `robot_viz.py --compare`, `glb_audit.py`
**Makbuzlar:** `results/otopsi_cp.json`, `results/_audit_last.json`

---

## 1. KLINIK TABLO (sikayet)

GLB'lerde isaretciler gozle yanlis goruldu: govde ucundan tasan oklar, seridin uzun ekseni
boyunca yonler, yogun parcada yariklarla hizasiz noktalar.

**Celiski:** ayni parcalardan `2669020000` metrikte **F1 1.000** aldi. Sayi "kusursuz",
goz "bozuk" diyor. Otopsinin sorusu: hangisi yalan soyluyor.

---

## 2. OLCUM BULGULARI

### 2.1 Metrigin tolerans kutusu, delikler arasi mesafeden BUYUK

`sina_kume.esle` bir tahmini su kutuda "dogru" sayar:

| eksen | tolerans |
|---|---|
| yanal | `max(3mm, %6 x kosegen)` → medyan **5.01mm**, %90 **7.27mm** |
| eksenel | **±40mm** |
| aci (tespit) | **serbest (180°)** |

Ayni korpusta GT'lerin en-yakin-komsu mesafesi **medyan 5.15mm**. Yani tolerans ≈ adim.
Olculdu (194 parca, 1314 GT):

| rejim | tolerans kutusunda BASKA bir GT bulunan GT |
|---|---|
| dusuk-CP | 83/400 = **%20.8** |
| cok-CP | 302/914 = **%33.0** |
| **toplam** | **385/1314 = %29.3** |

**Tespitlerin yaklasik ucte birinde, "dogru" sayilan eslesme fiziksel olarak KOMSU DELIGE
ait olabilir.** Metrik bunu ayirt edemez.

### 2.2 Agiz-vs-kontak (H1): eksenel kaymanin BIR KISMINI acikliyor

Tez CP'yi acikligin AGZINDA (`v_o`), uretici KONTAKTA tanimlar. Bizim noktamizi ureticinin
**kendi agziyla** (`cp_geometry.seat_to_mouth`) karsilastirinca:

| parca | seat→agiz medyani | eksenel kayma (seat'e) | eksenel kayma (AGZA) |
|---|---|---|---|
| 2669020000 | 10.6mm | 12.0–14.9mm | **1.7–4.2mm** (3 CP) / **15.7mm** (2 CP) |
| 2464750000 | 7.5mm | 9.4–12.6mm | **1.9–5.1mm** (4/4) |
| 2506380000 | **0.0mm** | 0.6–35.2mm | **degismiyor** |

**H1 KISMEN DOGRU.** Ince klemenslerde eksenel 12-15mm gercekten tanim farki. Ama
`2506380000`'de ureticinin CP'si ZATEN yuzeyde (offset 0.0) — orada her eksenel hata
GERCEK hatadir, mazereti yoktur.

### 2.3 Gercek kusurlar — yalniz YOGUN parcada

`2506380000` (50 GT, 25 tahmin):

| bulgu | sayi |
|---|---|
| nokta GOVDENIN ICINDE (`is_inside`) | **9/25** |
| yon ~180° TERS (aci 178–180°) | **3/25** |
| agiz ic capi < 0.7mm (tel gecmez) | **7/25** |
| agiz ic capi > 11mm (kanal degil, acik alan) | **2/25** |
| eksenel kayma > 10mm (mazeretsiz) | **11/25** |

### 2.4 Kusur OLMAYAN sey — sikayetin buyuk kismi

`2669020000`'de 5 CP'nin **4'unde aci uretici `InsertDirection`'iyla TAM 0.0°**.
Seridin uzun ekseni boyunca gorunen oklar **ureticinin kendi tanimi**. Goz onlari yanlis
sandi; olcum aklidi.

`2464750000`'de yerlestirilen 4 CP'nin **dordu de temiz**: agza gore eksenel 1.9–5.1mm,
aci 0.0°, onu ACIK (`ileri = inf`), arkasi MALZEME (`geri` 1.9–5.1mm). O parcanin derdi
yerlestirilenler degil, **yerlestirilmeyen 3 tanesi**.

Ve `glb_audit.py` 6/6 dosyada 9 maddelik cizim sozlesmesini GECTI; igne uclarinin tamami
govde disinda. **Cizim katmaninda kusur bulunamadi (H4 REDDEDILDI).**

---

## 3. AYIRICI TANI

| hipotez | hukum | dayanak |
|---|---|---|
| H1 agiz-vs-kontak tanim farki | **KISMEN DOGRU** | ince parcada kayma 15→2-4mm cokuyor; yogun parcada offset 0.0, cokmuyor |
| H2 yon serbest uzayi gostermiyor | **DOGRU, YEREL** | 3/25 ters (180°); 1 CP 65° sapmis ve onunde 2.9mm sonra malzeme var |
| H3 nokta agizda degil | **DOGRU, YOGUN PARCADA** | 9/25 govde icinde, 7/25 agiz capi <0.7mm |
| H4 cizim hatasi | **REDDEDILDI** | sozlesme 6/6 gecti, ucu disarida, olcumler GLB ile tutarli |

## 4. KESIN TANI

> **Iki ayri hastalik var ve biri digerini gizliyordu.**
>
> **(A) OLCUM KORLUGU (birincil):** tespit metrigi eksende ±40mm serbest, acida tamamen
> serbest, yanalda adim mesafesi kadar genis. GT'lerin **%29.3'unde** tolerans kutusunda
> baska bir GT var. Bu yuzden fiziksel olarak yanlis bir cikti "F1 1.000" alabiliyor.
> Sayi yalan soylemiyor — **yanlis soruyu yanitliyor.**
>
> **(B) YOGUN PARCADA GERCEK YERLESIM BOZUKLUGU (ikincil):** dusuk-CP parcalarda cikti
> temiz; yogun parcada noktalarin %36'si govde icinde, %12'si ters yonlu, %28'inin agzindan
> tel gecmez. Bu, mansetteki cok-CP F1 0.6583 rakaminin GORSEL karsiligidir — yeni bir
> sorun degil, ayni sorunun goze gorunen yuzu.

---

## 5. TEDAVI PLANI

Sira zorunlu: **(A) olculemeyen sey duzeltilemez.**

### FAZ A — OLCUM CERRAHISI (once bu; hicbiri modeli degistirmez)

**A1. Eksenel toleransi olcuye dayandir.** ±40mm keyfi. Olculen seat→agiz medyani
7.5–10.6mm. Yeni olcut: eksenel ≤ **15mm**. Eski ve yeni sayi YAN YANA raporlanir; manset
degisir ve DUSER — bu bir kayip degil, sisintinin geri alinmasidir.

**A2. BELIRSIZ ESLESME BAYRAGI.** Her TP icin "tolerans kutusunda baska GT var miydi"
kaydedilir. Iki sayi raporlanir: `F1` ve `F1_kesin` (yalniz tek-adayli eslesmeler).
Aradaki fark, metrigin belirsizlik payidir (su an %29.3'luk bir ust sinir).

**A3. Aciyi tespit metrigine de yaz.** Su an tespit aciya KOR (180° serbest). Aci
raporlanir (esik olarak degil, gorunurluk olarak) ki 180° ters bir CP tabloda gorulsun.

**A4. FIZIKSEL GECERLILIK denetcisi.** Her uretilen CP icin: `is_inside`, `ileri` serbest
mesafe, `agiz_ic_cap`. Uc bayrak. Metrikten BAGIMSIZ raporlanir — "kac CP fiziksel olarak
imkansiz" sayisi hicbir F1'in arkasina saklanamaz.

### FAZ B — URUN TARAFI (A4'un bayraklarini KARARA cevir)

**B1. Govde ici RED.** `is_inside=True` olan aday elenir ya da agza tasinir
(`seat_to_mouth` zaten var). Yogun parcada 9/25'i etkiler.

**B2. Kapali agiz REDDI.** `agiz_ic_cap < 0.8mm` → tel gecmez, CP olamaz. 7/25.

**B3. Onu kapali yon REDDI/CEVIRME.** `ileri` serbest mesafe < ~5mm ise yon ters
cevrilip yeniden olculur; iki taraf da kapaliysa aday elenir. 3/25 ters + 1 sapmis.

> B1-B3'un ucu de **fiziksel gecerlilik**, tez tanimina dokunmaz (ag, remesh, `v_o` ayni).
> Ve F0-4 hata bankasindaki **110 dusuk-CP FP** ile **203 cok-CP FP** icin dogrudan aday.
> Ancak A1-A2 yapilmadan olculemezler — cunku mevcut metrik bu duzeltmeleri GOREMEZ.

### FAZ C — SORUSTURMA (Faz 3 hipotezleri, henuz kanit yok)

**C1.** `pose_duzelt` noktayi govde ICINE itiyor olabilir mi? Duzeltme oncesi/sonrasi
`is_inside` sayilir. (Poz kafasi +0.0428 robot kazandirmisti; govde ici yan etkisi
hic olculmedi.)

**C2.** Yogun parcada 25/50 kacirma ile 9/25 govde-ici AYNI kok nedenden mi
(cozunurluk: 6000 tepe / 50 aciklik = aciklik basina ~120 tepe, cc-a "%82.7'si <30 tepe")?
Bu, F1-8 maddesinin (366 FN anatomisi) ta kendisi.

### FAZ D — GORSELLESTIRME (kusur bulunmadi, ama koru)

**D1.** A4 bayraklarini GLB'ye yaz: govde ici / kapali agiz / ters yon olan isaretciler
FARKLI renkte cizilsin. Boylece bir dahaki sefere goz, metrigin goremedigini DOGRUDAN
gorur. Cizim sozlesmesine 10. madde olarak eklenir.

---

## 6. SUNUM ICIN DURUST CUMLE

> "Dusuk-CP klemenslerde robot temiz calisiyor (yerlestirdigi noktalarin tamami dogru
> agizda, yonu ureticiyle 0.0° uyumlu). Yogun coklu-klemenslerde yariya yakinini buluyor
> ve bulduklarinin bir kismi fiziksel olarak gecersiz. Ayrica olcum metrigimiz bu farki
> gormeye yeterince duyarli degildi; onu duzeltiyoruz."

Bu, "0.7584" demekten daha dusuk bir sayi ama **savunulabilir** bir cumledir.

---

## 7. SONRAKI ADIM

Faz A tek basina 1 gunluk is ve **22 maddelik ana planin F0-4/F0-5 maddelerinin onune
gecmeli**: hata bankasi ve hedef aritmetigi, duzeltilmemis bir metrigin uzerine kurulursa
tum plan sisik sayilara kilitlenir.
