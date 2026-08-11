# OPERASYON ROBOT 0.80

> Tez çizgisi korunur: DiffusionNet semantik segmentasyon, tezin remesh'i (~6000 tepe),
> tezin v_o türetmesi. Hiçbir faz bunları değiştirmez.
> Her sayı bir makbuza bağlıdır; makbuzsuz sayı bu belgeye giremez.

---

## §0 — KANITLANMIŞ KISIT (planın temeli)

**robot F1 ≤ tespit F1 — yapısal, ölçüm değil.**
Robot eşleşme ölçütü her parçada tespitten sıkı: yanal 2mm sabit vs `max(3mm, %6·köşegen)`
(ölçüldü: medyan 5.00mm, min 3.00mm — hiçbir parçada 2mm'nin altına inmiyor), açı 10° vs serbest.
Robot'un doğru kümesi tespitin doğru kümesinin **alt kümesidir**.

Gözlenen geçiş oranı: **703/873 = 0.8053**.

Buradan robot 0.80'in fiyatı:

| poz/açı geçiş oranı | gereken tespit | değerlendirme |
|---|---|---|
| 0.803 (bugün) | **0.9963** | imkânsız |
| 0.900 | **0.8889** | Bayes tavanı 0.86'nın ÜSTÜNDE |
| 0.950 | 0.8421 | tavanın %98'i |
| 1.000 (kusursuz) | 0.8000 | tavanın %93'ü |

**Sonuç: robot 0.80 = tespit ≥0.84 VE poz/açı ≥%95.** İkisi birden.

---

## §1 — NEREDE DURUYORUZ (2026-08-03, düzeltilmiş)

| bölme | tespit | robot | işaretli |
|---|---|---|---|
| **HAVUZLANMIŞ (194 parça / 174 grup)** | **0.7584** [0.7145, 0.7992] | 0.5893 | 0.5818 |
| PXC dışarıda (90) | 0.6680 | 0.5591 | 0.5528 |
| WEI dışarıda (103) | 0.5968 | 0.4078 | 0.3880 |
| DEV (54) | 0.8060 | 0.6970 | — |
| VAL (100) | 0.7535 | 0.5939 | — |

**BUGÜN DÜZELTİLDİ** (`sina_kume.f1w` rejim ağırlığını yeniden normalleştirmiyordu):
ATANMAMIŞ 0.6635 → **0.7413** · seri 16 0.8333 → **0.9310** · seri 17 0.6531 → **0.7297**.
Manşet ve üretici-dışı bölmeler etkilenmedi (ikisinde de her iki rejim var).

### Ölçülen tavanlar

| tavan | değer | kaynak |
|---|---|---|
| aday havuzu recall'ı | 0.8579 | 188/1323 GT'ye hiç aday yok |
| kâhin gate | 0.9370 | `x1_bosluk.json` |
| **geometrik ayrılabilirlik (Bayes)** | **~0.86** | `rb_bayes.json` + doğrulama |
| açı mükemmel olsa robot | 0.6717 | `r3_yanal_tavan.json` |
| yanal mükemmel olsa robot | 0.6469 | aynı |
| ikisi mükemmel | 0.7584 (= tespit) | yapısal |

---

## §2 — KAPANMIŞ KOLLAR (tekrarlanmayacak)

Her biri önceden yazılmış kill ile, ürünün kendi karar yolundan, grup bootstrap'la ölçüldü.

**Gate tarafı (dokuz mekanizma, tek cephe):** öznitelik seçimi (sızıntılı ve sızıntısız),
dönüşüm, model sınıfı, grup/parça ağırlığı, eşik değeri, kutup örgüsü, B-rep boşluk grafı,
aday-tipi yönlendirme, tip-içi eşik. **Yasa: eklenen her ayırt edici güç kolay alt-popülasyona
harcanıyor.**

**Karar kuralı:** top-K sayım tabanlı seçim −0.0145 (kâhin K bilinse bile yalnız 0.7864).

**Yön/konum:** konum seçicisi +0.0000 (poz kafası zaten optimum) · yön sözlüğü kendi tavanının
%90'ını kapsıyor ama tavan 0.6717 · sürekli regresör −0.0256.

**Segmentasyon:** türetme eşiklerini sonuna kadar gevşetmek +0.0189 (ağ o açıklıkları görmüyor) ·
`keig128_s1` (projenin en yüksek val Conn-IoU'su, 0.7079) uçtan uca **−0.0220** — val Conn-IoU
uçtan uca aktarılmıyor.

**Ölçüt gevşetme:** 5mm/30° gibi absürt toleransta bile robot 0.657 — bağlayıcı kısıt tespit.

---

## §3 — TAVAN YÜKSELTME: ÜÇ CEPHE

### Cephe A — SEGMENTASYON (hiç dokunulmamış, en yüksek getiri)

Dağıtılan 4 ağ **yalnız 189 mesh gördü**. Diskte kullanılmamış 257 üretici-etiketli parça var.
"Daha çok veri" iki kez denenmiş ve ikisi de **karıştırılmış deneydi**: H3 aynı anda üç değişken
oynatmış (insan etiketlerinin %77'sini düşürmüş), G bambaşka bir mimariydi.

**A1 (KOŞUYOR):** dağıtılan tarif + AYNI insan dizinleri + üretici etiketleri → **415 mesh** (2.2×).
LOCKED sızıntısı ölçülüp temizlendi (12+19 grup atıldı; filtresiz eğitim sınavın 31 grubunu yakardı).
*Kill: val Conn-IoU dağıtılanların aralığını (0.6378–0.6629) geçmezse kapanır.*

**A2:** 5. üye (`keig96_s3`, val 0.6676) — config kendi notunda "karar kesin değil" diyor.
*Kill: tespit +0.01 ve GA>0.*

**A3:** A1 geçerse 3 tohum daha → yeni topluluk → gate yeniden uydurulur → uçtan uca sınav.

### Cephe B — VERİ (ölçülmüş fiyat etiketi)

`F1 = 0.5295 + 0.0330·ln(grup)`, doymuyor. Şu an 953 grup.

| hedef | gereken grup | kat |
|---|---|---|
| 0.78 | ~1.900 | 2.0× |
| **0.80** | ~3.630 | **3.8×** |
| 0.84 (robot 0.80 için) | ~7.900 | 8.3× |

Ve **bileşim hacimden ayrı değer taşıyor**: aynı grup sayısında karışık üretici, tek üreticiyi
her boyutta yendi (+0.033 / +0.051 / +0.049, ortalama **+0.0443**).

**Talep:** *terminal block + STEP + kesin ConnectionPoints* olan üçüncü üretici; en az 300,
tercihen 900+ yeni geometri grubu. **İndirmeden ÖNCE 60/20/20 bölme; son %20 mühürlü.**
900 family-dışı parçanın beklenen katkısı: tespit **+0.018…+0.021** (hacim) + çeşitlilik primi.

### Cephe C — ÖLÇÜT DOĞRULUĞU (otopsi yapıldı)

Sabit 2mm ölçütünün kodda hiçbir fiziksel gerekçesi yok ve **boyuta kör**:

| ağız genişliği | çift | 2mm geçen | 2mm, yarıçapın kaç katı |
|---|---|---|---|
| 3–5mm | 104 | %91 | 0.9× |
| 5–8mm | 218 | %89 | 0.7× |
| **8mm+** | **519 (%59)** | %88 | **0.3×** (gereğin 3 katı sıkı) |

Fiziksel olarak doğru ölçüt boyutla ölçeklenir. Tahminlerin **%96'sı ağzın içine** düşüyor.

| ölçüt | robot |
|---|---|
| sabit 2mm/10° | 0.5893 |
| **boyut-göreli 0.50× ağız yarıçapı** | **0.6081** (+0.019) |
| boyut-göreli 1.00× | 0.6399 |

**Karar kullanıcıya ait**: ölçüt değişikliği bir MODEL kazancı değildir ve öyle sunulmaz.
Robotun mekanik spesifikasyonu geldiğinde doğru değer oradan okunur.

---

## §4 — FAZLAR VE KAPILAR

| faz | iş | süre | kapı |
|---|---|---|---|
| **F0** | ölçüm bütünlüğü (bkz. §6) | bitti | 154 test yeşil |
| **F1** | A1 eğitimi + değerlendirme | 4 sa | val Conn-IoU > 0.6629 |
| **F2** | A1 geçerse: 3 tohum + topluluk + gate yeniden uydurma + uçtan uca | 1 gün | tespit +0.01, GA>0, üretici-dışı düşmemeli |
| **F3** | A2 (5. üye) | 2 sa | aynı kapı |
| **F4** | Cephe B talebi yazılır ve veri toplanır | dış | 300+ grup, %20 mühürlü |
| **F5** | veri geldiğinde yeniden eğitim + ölçüm | — | tespit ≥0.80 |
| **SON** | LOCKED sınavı — **yalnız bir kez** | — | `sinav_egitim_maskesi()` ile eğitilmiş gate ZORUNLU |

**LOCKED kuralı:** dağıtılan gate LOCKED'in 77/95 parçasını eğitimde görmüş. Sınav yalnız
`olcum_kumesi.sinav_egitim_maskesi()` ile eğitilmiş gate ile koşulur; korpus %7.5 küçüleceği için
sınav sayısının ~0.0026 düşük çıkması **beklenir ve gerileme değildir**.

---

## §5 — ÇARŞAMBA TESLİMİ

Sunum bir sayı iddia etmez, bir **sınır** kanıtlar. Elde olan:

- tespit **0.7584** (GA'lı, 9 bölmede dağılım), robot **0.5893**, fiziksel-işaretli **0.5818**
- segmentasyon mIoU **0.681** — tezin 0.514'ünün üstünde
- **ölçülmüş üç tavan** ve robot 0.80'in aritmetiği
- **fiyat etiketi**: 0.80 için 3.8× korpus, çeşitlilik +0.0443
- 95 parçalık sınav **harcanmadı**

---

## §6 — BUGÜN KAPATILAN ÖLÇÜM AÇIKLARI

| açık | etki | durum |
|---|---|---|
| `f1w` tek rejimli bölmede ağırlığı normalleştirmiyordu | 3 bölme %10.5 düşük raporlanmış | düzeltildi + bekçi testi |
| eski test bu hatayı "doğru davranış" diye çiviliyordu | hatayı koruyordu | test düzeltildi |
| `_der_tam.pkl`'i üreten betik ağaçta yoktu | tüm sayılar yeniden üretilemezdi | `turet.py` yazıldı, 6/6 birebir doğrulandı |
| üretici etiketleri LOCKED ile kesişiyor | filtresiz eğitim sınavın 31 grubunu yakardı | temiz kopyalar üretildi, ihlal 0 |
| manşet önbelleği uzunlukla hizalıyordu | üretici-dışı robot sayıları bozulabilirdi | kimlik kontrolü eklendi |
| `aci_secici` makbuzu "geçmedi" diyor ama dağıtılmış | çelişkili iz | `MAKBUZ_CELISKILERI` altında kayda geçti |
| yön seçicisinin etiketi işaretsizdi | sahte +0.0319 (fiziksel metrikte 0) | geri alındı, kök neden belgelendi |

**Bilinen açık (henüz kapanmadı):** üretici-dışı bölmelerde poz/açı/üye kafaları sabit
artefakt olarak uygulanıyor ve o üreticinin parçalarıyla eğitilmiş — yani `WEI dışarıda`
robot sayısı "o üreticiyi hiç görmemiş sistem" sayısı **değildir**. Tespit sayıları
etkilenmez (ölçülmüş bedel +0.0001).
