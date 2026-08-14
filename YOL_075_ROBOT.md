# CP TESPİTİ — ölçülmüş yol listesi

> **2026-07-31 tam revizyon.** Bu dosyanın önceki tüm sayıları geçersiz. Sebep: ölçüm protokolü
> üç kez düzeltildi (kaba→keskin geometri anahtarı, tek küme→üçlü bölme, yapılandırma-başına
> eşik ayarı→sabit ürün eşiği) ve her düzeltme bazı "kazançları" yok etti.
> Aşağıdaki "ölen kollar" tablosu neyin neden düştüğünü taşıyor.

## MANŞET — İKİ SAYI (2026-08-01)

Eski manşet geometri-ayrıktı ama **üretici-karışıktı**. Görülmemiş bir üreticide ne olduğu hiç
ölçülmemişti; ölçünce çöktüğü görüldü ve gece boyunca o düzeltildi.

| | gece başı | **şimdi** |
|---|---|---|
| **tanıdık** (üretici-karışık, DEV+VAL 200 parça) | 0.7437 | **0.7410** |
| **görülmemiş üretici — WEI dışarıda** (108) | **0.3150** | **0.4832** |
| görülmemiş üretici — PXC dışarıda (91) | 0.7037 | **0.7203** |

robot-hazır: tanıdık 0.4486 · WEI dışarıda 0.2242 · PXC dışarıda 0.4466

**Ürün:** 22 sütunlu gate (13 baseline + 5 B-rep fiziksel + 4 içbükey topoloji) + **göreli eşik**
(parça-içi 0.5 × en yüksek, mutlak baseline 0.25). 109/109 test yeşil.
**Geri alma:** `wire_gate.pkl.pre_topo` / `.pre_fiz` + `cp_config` bayrakları — her biri tek satır.

Makbuz: `results/t15_topo_uctan_uca.json`, `results/t8_uretici_disi_uctan_uca.json`,
`results/manset_goreli.json` · Ayrıntılı gece raporu: `GECE_2026_08_01.md`

*tespit* = lateral ≤ max(3 mm, %6×köşegen), axis serbest · *robot-hazır* = lateral ≤2 mm **ve** axis ≤10°

**Ölçüm altyapısı:** `results/split3.json` (DEV karar / VAL sınav / **LOCKED harcanmadı**),
keskin geometri anahtarı, eşli bootstrap. **Yeni axis:** her arm artık **üretici-dışı** da ölçülüyor.

---

## TAVAN — nereye kadar gidilebilir *(ölçüldü, tahmin değil)*

`m_konum_tavani.py`: konum **ve** axis ORACLE ile kusursuz yapıldığında robot-hazır F1
**tespit F1'e eşitleniyor**. Yani:

> **Robot-hazır'ın tavanı = tespit F1.** Konum rötuşu ve axis iyileştirmesi tek başına hiçbir
> şey açmaz; kazanç yalnızca **daha çok açıklık bulmaktan** gelir.

Bu tek cümle listenin sırasını belirliyor: "konum" kalemleri listeden **düştü**.

---

## 🔴 DENETİM SONRASI DURUM (2026-08-01) — önce bunu oku

Dış denetim 8 bulgu getirdi, **hepsi doğrulandı ve düzeltildi.** En ağır sonuç: manşetlerin
dayandığı ölçüm hattı kırıktı, dolayısıyla **önceki sayılar geçersiz.**

| bulgu | durum |
|---|---|
| DEV/VAL adları yanlış bağlıydı (gerçek bölme `split3.json`) | ✅ `measure_set.py` tek kaynak |
| 200 satır = 197 parça (3 tekrar) + 3 LOCKED doğrudan kullanılmış | ✅ 194 parça / 171 grup; **98 temiz LOCKED korunuyor** |
| parça bootstrap ikizleri bağımsız sayıyordu | ✅ **grup** bootstrap |
| eğitim ≠ runtime candidate üretimi (`votes` 12 vs 4) | ✅ `robot_cp.derive_candidates` tek kaynak + 3 test; corpus yeniden üretildi |
| GA karara katılmıyordu | ✅ `karar_olcutu` (5) KANIT şartı |
| çöküş yönlendiricisi testini geçmeden dağıtılmıştı | ✅ **GERİ ALINDI** (D→R geçmedi) |
| MD5 bayattı, hiçbir şey kontrol etmiyordu | ✅ tazelendi + `tests/test_artifact.py` |
| kökten `pytest` hiç çalışmıyordu | ✅ `pytest.ini` |

**Yeni manşet (194 parça, grup-bootstrap):** DEV **0.7985** / VAL **0.7223** → *uçurum 0.076*,
havuzlanmış 0.7461, WEI-dışı 0.5689, PXC-dışı 0.6877.

**`0.9372` recall DEĞİL, kâhin F1.** Gerçek candidate recall'ı ağırlıklı **0.8795** / ham 0.8073;
çok-CP **0.7691** → çok-CP F1 tavanı **0.8695**, regime-ağırlıklı ceiling 0.9354.

**Sıradaki (denetimin P1'i):** zengin gate blokları — konum-9 + çok-yarıçap-24, güncel 22
özelliğe yeniden üretilmeli (taper ölçülmüş ölü). Kanıt: `results/rich_gate_receipt.json`
(part-out +0.0351, family-out +0.0489). Ağ çıkarımı gerektiriyor → ayrı tam koşu.

## AÇIK LİSTE — öncelik sırası, her madde ölçülmüş gerekçeyle

### 0a. TESPİT 0.90 / ROBOT 0.75 — ceiling merdiveni *(2026-08-01)*

| basamak | TESPİT | ROBOT |
|---|---|---|
| şu an | 0.7439 | 0.4500 |
| + kâhin **gate** | **0.9372** | 0.5091 |
| + kâhin yön | 0.9372 | 0.6124 |
| + kâhin konum | 0.9372 | **0.9372** |
| **candidate tavanı** | **0.9372** | 0.9372 |

**Adaylar zaten GT'nin %93.7'sini içeriyor** → tespit 0.90 bir **gate** problemi, türetme değil.
Gate boşluğu (+0.1933): **%38 eşik / %62 SIRALAMA**. Sıralama yarısı yeni bilgi ister
(belgeli duvar). Eşik yarısı 4 formülasyonla denendi:

| formülasyon | fark |
|---|---|
| parça-başına regresyon | −0.0148 |
| yalnız-azaltma | +0.0089 |
| **dur/devam kararı** (candidate başına, 3277 satır) | +0.0142 |
| **+ F1 ağırlıklı** | **+0.0150** |

Doğrulama: 5/5 seed pozitif (ort +0.0141, sd 0.0013) ama bootstrap **[−0.0010, +0.0330]** —
sıfırı kıl payı içeriyor. Bar +0.02 → **dağıtılmadı**, istiflenebilir candidate.
Makbuz: `results/t_tavan.json`, `t2`…`t7`

### 0. ~~ROBOT-HAZIR 0.65~~ — **ULAŞILAMAZ, ölçüldü 2026-08-01**

`robot ≈ 0.673 × geçiş_oranı` (tespit 0.7439, geçiş %66.9 → robot 0.4500).
**0.65 için geçiş %96.6 olmalı.** Yapısal duvar buna izin vermiyor:

**107/107 dik nokta (eşleşmelerin %12.3'ü), KESİN analitik B-rep ekseni verildiğinde bile
45° üstünde kalıyor.** Üreticinin takma yönü o ağızlarda silindir eksenine gerçekten dik —
yarık/push-in sınıfı. Bunlar hep kaybedilirse geçiş tavanı %87.7 → **robot tavanı 0.5905**
(ve bu *mükemmel* lateral varsayımıyla).

| ölçülen arm | sonuç |
|---|---|
| parça-içi axis uzlaşısı | kâhin tavanı +1.5 puan; gerçekte **net −110 nokta**. ÖLÜ |
| B-rep kapsaması → %100 | **+3.2 puan** ceiling. Küçük |
| BREP_MAX_OFF (kodda asılı "P2 taraması") | **yapıldı: fark yok** (0.4500/0.4495/0.4492). TODO kapatıldı |
| noktayı analitik eksene izdüşürme | candidate +2.1 → **uçtan uca −0.045**. ÖLÜ (`CP_AXIS_PROJECT`, kapalı) |

**"Yarık/push-in yönünü modelle" DENENDİ (2026-08-01) — üç ölçüm, üçü de kapatıcı:**

| deney | sonuç |
|---|---|
| S1: yönü hangi aşama bozuyor? | **hiçbiri — hepsi bozuk.** Son yön >45° saptığında tez normali / channel_axis / yuvarlama / normal-kovaryans / B-rep'in **hepsi** %0. Kâhin (en iyi aşama) **+0.0 puan** |
| S2: yarık dedektörü hedefte ne verir? | 107 dik noktanın 74'ünde yarık var, 26'sının yönü ≤10° (medyan 26° vs 90°) |
| S3: kural **herkese** uygulanınca | dedektör eşleşmelerin **%80.4'ünde** ateşliyor → düzelen 41 / bozulan 228 = **net −187** |
| **S4: seçiciyi ÖĞREN** (8 özellik, fold-dışı) | **mekanizma düzeldi: bozulan 228 → 0**, ama net yalnız **+11** (geçiş +%1.26 → robot ~0.4585). Bar %2 → dağıtılmadı |

**Bu dalın kâhin tavanı: +41 nokta = robot 0.4816.** Mükemmel bir seçici bile 0.48'e kadar taşır.

Yani eksik olan bir algoritma değil **bilgi**: telin hangi yarıktan girdiğini söyleyen işaret yok.
Tez §5.3.6'nın kendi normali (`v_o − v_s`, bbox'a hizalı) da bu sınıfta %0.

**Gerçekçi hedef: ~0.46–0.48.** 0.65 için iki şey birden şart: (a) yarık/push-in ağız sınıfının
yönü doğru modellenmeli, (b) tespit F1 yükselmeli (0.673 çarpanı oradan geliyor).
Makbuz: `results/r0_robot_ayristir.json`, `r2`, `r3`, `r4`, `r5`, `r6`



### 0b. ~~TOPOLOJİ YARIÇAPI~~ — **KAPANDI 2026-08-01, R=6.0 kalır**

`topo_feats.R_VARSAYILAN = 6.0 mm` keyfi seçilmişti; tarandı ve **uçtan uca çürüdü**.

| R (mm) | candidate düzeyi F1 | **tanıdık** | PXC-dışı | WEI-dışı |
|---|---|---|---|---|
| **6.0 (dağıtılan)** | 0.7404 | **0.7410** | 0.7203 | 0.4832 |
| 8.0 | 0.7451 | 0.7422 | 0.6961 | 0.5013 |
| 12.0 | **0.7504** ← candidate kazananı | **0.7301** | 0.7121 | 0.4871 |

**Aday düzeyinin kazananı (R=12) uçtan uca kaybetti** (tanıdık −0.0109). R=8 ön-yazılı barın
lafzını geçiyordu ama eşleştirilmiş bootstrap bitirdi: kazancı **gürültü**
(WEI +0.018, GA [−0.007, +0.044]), bedeli **gerçek** (PXC −0.024, GA [−0.044, −0.006]).

**Dünkü karar doğruymuş:** candidate düzeyine bakıp korpusu yeniden üretseydim bir **gerileme**
dağıtmış olacaktım. Bu, candidate düzeyinin bu araştırmadaki **dördüncü** yanıltması.

**Yan kazanım:** yarıçap artık ayarlanabilir (`wire_gate.TOPO_R` / `cp_config.gate_topo_r`) ve
topoloji sütunlarını ağ çıkarımı olmadan yeniden üretme yolu var (`u1_topo_r_yeniden.py`,
R=6'da birebir doğrulandı: 4 sütunda da maks fark 0). ~80 dk yerine ~3 dk.
Makbuz: `results/u2_topo_r_uctan_uca.json`, `results/t19_topo_yaricap.json`

### 1. B-rep FİZİKSEL özellikler → gate *(en somut, kanıtı en taze)*
FP otopsisi yanlışların **%86.3'ünü** üç fiziksel imzaya bağlıyor (boydan-boya delik %31,
yarıçap <1 mm %21, axis-dik %35) ve gate'in 13 özelliğinin **hiçbiri bunları taşımıyor**.

Ölçüldü (`q4_brep_fiziksel.py`, 3277 candidate, 200 ayrık parça, bağ düzeltmeli Mann-Whitney + null):

| özellik | AUC | null p95 | TP med | FP med |
|---|---|---|---|---|
| **brep_r** (analitik yarıçap) | **0.707** | 0.519 | 1.80 mm | **0.15 mm** |
| **boş_derinlik** | 0.290 → ters 0.710 | 0.520 | 0.00 | 1.32 |
| eş-eksenli silindir | 0.603 | 0.518 | 1.0 | 1.0 |
| r_oranı / geçen-delik | 0.482 / 0.579 | — | — | — |

Bu ölçüm **yarıçap hatası düzeltilmeden yapılamazdı** (aşağıya bak).

**ARTIMLI DEĞER ÖLÇÜLDÜ** (`q5_artimli_deger.py`, aynı candidates, grup-çapraz OOF):

| gate | OOF AUC | candidate F1 | precision | recall |
|---|---|---|---|---|
| 13 mevcut | 0.8576 | 0.6997 | 0.684 | 0.716 |
| **13 + 5 fiziksel** | **0.8996** | **0.7554** | 0.732 | 0.780 |

`brep_r` ile mevcut `size` **korelasyonu −0.009** → kopya değil, **gerçekten yeni bilgi**.

**Durum:** özellikler `wire_gate`'e bağlandı (`WG_FIZ_FEATS`, varsayılan **kapalı**,
`step_path` yokken nötr); `gate_regrow` artık `step_path` geçiriyor (vermezse 5 sütun sessizce
sıfır olurdu — E maddesinde tam olarak bu yaşandı). Tam corpus verisi 18 sütunla üretiliyor.
**Sıradaki adım:** `q6_fiz_uctan_uca.py` — iki gate **aynı candidate havuzunda**, uçtan uca.
**Kill:** DEV'de uçtan uca tespit F1 artmazsa dağıtılmaz; artarsa VAL'de sınanır.
**Uyarı:** candidate-düzeyi F1 uçtan uca F1 değildir — E maddesi gate düzeyinde +0.0068 kazanıp
uçtan uca −0.0125 kaybetmişti.

### 2. Bağımsız 6. üye *(tek genelleşen sinyali güçlendirir)*
`votes` dürüst bölmede transfer eden **tek** özellik (AUC düşüşü 0.018; diğerleri 0.12–0.18).
5. üye eğitildi (`recall_hard_keig96_s3`, val Conn-IoU **0.6676** = mevcut en iyi üyeden yüksek):
VAL **+0.0208**, DEV **−0.0038**; ikisinde de **recall ↑ / precision ↓**.
Oy-ölçeği açıklaması **denendi ve tutmadı** (votes×4/5 → VAL +0.0164, ölçeksizden kötü).
**Karar:** havuzlanmış ~+0.008 barajın altında → 5. üye **alınmadı**. Ama gate 4-üyeli havuzda
eğitilmiş; 5-üyeli havuzda yeniden eğitilmeden bu karar kesin değil.
**Kill:** gate yeniden eğitimiyle VAL'de +0.01 gelmezse kapanır.

### 3+4. EK BLOK — ÖLÇÜLDÜ, **DAĞITILMADI** *(kapalı; kod duruyor)*

İçeri kanal derinliği + 3 renk özelliği (22 sütun). Tam corpus verisi üretildi, uçtan uca
DEV→VAL ölçüldü, **havuzlanmış (200 parça) karar:**

| arm | tespit | robot |
|---|---|---|
| 18 sütun (dağıtılan) | 0.7437 | 0.4594 |
| 19 (+ic_derinlik) | 0.7506 | 0.4595 |
| 22 (+renk) | 0.7492 | 0.4701 |

| katkı | tespit | robot |
|---|---|---|
| depth (19−18) | +0.0069 gürültü | +0.0003 yok |
| renk (22−19) | −0.0015 gürültü | +0.0103 gürültü |
| **toplam (22−18)** | **+0.0054 gürültü** | +0.0107 belirgin |

**Neden dağıtılmadı:** önceden yazılı kill ölçütü **tespit F1** idi ve havuzlanmışta gürültü.
Tek belirgin sonuç **sonradan bakılan** robot metriğinde ve **6 testten biri** — yokluk
hipotezi altında 6 testten en az birinin belirgin çıkma olasılığı ~%26. Bunu kazanç saymak,
bu gece L2'yi öldüren hatanın aynısı olurdu. Ayrıca bedeli var: parça başına bir OCC STEP
okuması (0.24 s) + candidate başına içeri ışın.

**Yine de kazanılanlar (kodda duruyor):**
- `ic_derinlik` doğru çalışıyor (nötr %16.3, medyan 3.32 mm) — `bos_derinlik`'in yanlışlıkla
  **dışarı** baktığı bu sayede anlaşıldı.
- Sessiz-nötrleşme sayacı: `kanalda_metal` adayların **%99'unda** hiç ateşlemiyormuş —
  sayaç olmasa ölü sütun dağıtılacaktı.
- Renk kapsaması iki koldan iyileştirildi (**bu veriden SONRA**, yani ölçüme yansımadı):
  katı-seviyesi renk kurtarma (doğrulanan parça 23→34/40) ve **renkten bağımsız kontrast**
  ölçütü (kapsama 14→19). `METAL_RGB` tek bir gümüş tonuna sabitti; parçaların %22'sinde renk
  var ama o ton yok. "Gri = metal" demek yanlış olurdu (gri plastik gövde yaygın), o yüzden
  ölçüt "baskın renkten farklı yüz" oldu.

**Yeniden açılırsa:** veriyi iyileştirilmiş renk kapsamasıyla yeniden üret, `kanalda_metal`'i
at, kill ölçütünü yine **tespit F1**'e yaz.

### 5. Veri
Korpus ×1.92 büyümesi WORK CP-F1'e +0.0107 vermişti; eğri düzleşiyor. WSCAD indirme kolu ölü
(325 deneme → 7 dosya). Yeni veri ararken ölçüt "kaç parça" değil **"kaç yeni geometri grubu"**.

---

## BU GECE ÖLEN KOLLAR *(hepsi ölçüldü, biri üründen geri alındı)*

| arm | sonuç | öğrettiği |
|---|---|---|
| **L2** (3 ezberci özelliği at) | DEV +0.0066 → **VAL −0.0111**; sabit eşikle DEV'de de 13-özellik **+0.0279 BELİRGİN** | eski kazanç **yapılandırma-başı eşik ayarından** geliyormuş → GERİ ALINDI |
| **J** (ağırlıklı konum ortalaması) | havuzlanmış **+0.0026**, GA [−0.0069, +0.0116] | gürültü. **Ablasyon iki şeyi birden değiştiriyordu**: konum ortalaması *ve* oy sayma anlamı (`vote_avg` yinelemeli / `_vote2` benzersiz). İkisi birlikte gürültü |
| **gate v6** (güncel hatla üretilmiş veri) | DEV −0.0146, VAL −0.0039, ikisi de gürültü | "dağıtılan gate bayat" hipotezi **kapandı** |
| **5. üye** | VAL +0.0208 / DEV −0.0038 | havuzlanmış baraj altında (bkz. madde 2) |
| **oy ölçeği** (votes ×4/5) | VAL +0.0164 (ölçeksizden kötü) | 5. üyenin precision kaybı ölçek kayması **değil** |
| **B-rep kapı taraması** | 8 ayar, yayılım **0.003** | `max_off`/`r_max` hiçbir şeyi açıklamıyor |

---

## YOL BOYUNCA BULUNAN GERÇEK HATA

**`brep_axes` silindir yarıçapı 3.5 fold küçüktü.** Yarıçap ve "axis üzeri nokta" örneklenen
yüzey noktalarının **centroid'inden** hesaplanıyordu; oysa ölçüldü ki bir parçadaki silindirik
yüzlerin **%0'ı tam tur** — hepsi çeyrek/yarım yay. Yay üzerindeki noktaların ortalaması eksenin
üstünde değildir.

Bozduğu yerler: robota giden **`size_mm`**, `axis_at`'in yarıçap filtresi, mesafe testi.
Düzeltme: eksene dik düzlemde çember oturtma (`brep_axes._fit_circle`).
**Doğrulama: STEP metniyle 327/327, medyan error 0.0000 mm.**

Düzeltmenin F1 etkisi: DEV robot 0.4974→0.4841, VAL tespit 0.6529→0.6417 — küçük ve n=100'de
gürültüden ayırt edilemiyor; kapı taraması da kurtarmadı (düz). **Düzeltme kalıyor:** F1
`size_mm` doğruluğunu hiç ölçmüyor, robot ise onu kullanıyor.

---

## BİLİNEN UYUMSUZLUK — gate eğitimi vs ürün *(ölçüldü, gürültü, ama kayıtlı)*

Gate **eğitim verisi** adayları `f1_sweep.union_all` ile birleştiriyor: yakındaki **her** üye
oyu sayılıyor, aynı model iki kez sayılabiliyor (korpusta `votes` max **12**, üye sayısı 4 iken).
**Ürün** ise `robot_cp._vote2` kullanıyor: **benzersiz model** sayımı (max = üye sayısı).

`votes` gate'in **tek genelleşen özelliği** (dürüst bölmede AUC düşüşü 0.018; diğerleri 0.12–0.18),
dolayısıyla bu bir eğitim/çalışma uyumsuzluğudur. Bu geceki parite düzeltmesi `_vote2`'yi benzersiz
sayıma çevirdiğinde doğdu; eğitim tarafı `union_all`'da kaldı.

**Ölçülen bedeli gürültü düzeyinde (~0.003)**, o yüzden gürültü için ürün kodu değiştirilmedi.
**Kural:** gate verisi bir daha üretilirken **ürünün kendi birleştiricisi** kullanılmalı.

---

### 6. İÇBÜKEY KENAR TOPOLOJİSİ → gate *(sıradaki candidate, henüz ölçülmedi)*

Delik-tanıma alanının birinci sinyali: **bir açıklık, içbükey kenarlarla çevrili yüz kümesidir**
(dışa çıkıntı ise dışbükey kenarlarla). Bizim 18 özelliğimizin **hiçbiri topolojik değil** —
hepsi ya olasılık istatistiği ya nokta-geometrisi.

**Neden teze dokunmuyor (doğrulandı):** gate tezde **yok**. Tez Contact sınıfını tanımı gereği
*"Kontaktierung bzw. Werkzeugeinschub"* diye birleştiriyor, yani tel/alet ayrımını hiç sormuyor;
wire-gate bizim eklediğimiz katman. Ağ, remesh, sınıflar değişmiyor.

**Neden umutlu:** bugün +0.0704'ü tam da böyle aldık — gate'e sahip olmadığı **fiziksel bilgiyi**
vererek. Bu, aynı kapının açılmamış ikinci kanadı.

**Neden ölebilir (peşinen):** GEO kolumuz silindir-filtresiyle F1 0.332'de ölmüştü. Farkı: orada
**bağımsız bir dedektör** kuruyorduk ve içbükeylik kullanmıyorduk; burada çalışan gate'e sütun ekliyoruz.

**Ölçüm sırası (kısayol yok):** (1) candidate düzeyinde AUC + null testi ~1 saat — ayırt edicilik
yoksa orada biter · (2) geçerse tam corpus verisi ~1.5 saat · (3) DEV → VAL uçtan uca.
**Kill:** tespit F1 +0.01. **Beklenti:** +0.02–0.04 (fiziksel bloğun yarısı).

---

## ALTYAPI BORCU — gate verisi üretimi tek çekirdekte

`gate_regrow.py` 1913 parçayı **sırayla** işliyor: parça başına ~2.7 s (4 model çıkarımı +
8 candidate türetmesi + B-rep/EK özellikleri), toplam ~85 dk, ve bu sürede **CPU %33** — yani
makinenin çoğu boşta. Parçalar birbirinden bağımsız; `multiprocessing` ile 3–4 fold hızlanır.

Bu bir doğruluk sorunu değil, ama **her özellik denemesi 85 dk bekleme** demek ve bu gece
üretim iki kez yeniden başlatıldı. Bir sonraki tam üretimden önce yapılmalı.
**Dikkat:** zehirli-parça koruması (INFLIGHT dosyası) süreç başına ayrılmalı, yoksa paralel
işçiler birbirinin kaydını ezer.

---

## KURALLAR *(bu gece üçü de çiğnendiği için yazılı)*

1. **Kill ölçütü ÜRÜN metriğine yazılır** (F1), AUC'ye değil — ve koşmadan önce.
2. **İki gate ancak AYNI candidate havuzunda** karşılaştırılır.
3. Bir kümede seçilen arm, **başka kümede sınanmadan** manşete girmez.
4. Bir ölçüm düzeltilince **ona göre ayarlanmış tüm eşikler bayatlar** — hepsi yeniden bakılır.
5. LOCKED tek atıştır; üzerinde hiçbir ayar yapılmaz.
6. Bir yardımcı fonksiyonun ne yaptığını varsayma, **docstring'ini oku** — eski F maddesi
   (`family_key` = parça numarasının kendisi) tam olarak bu yüzden doğmuştu.
