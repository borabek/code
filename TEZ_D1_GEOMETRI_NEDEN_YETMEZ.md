# D1 — Neden saf geometri yetmez, neden segmentasyon gerekli

> Teze girecek bölüm taslağı. Bu, negatif bir sonuç değil — **tezin kendi yöntem seçiminin
> deneysel gerekçesi**. Scheffler'in boru hattı bir segmentasyon ağı kullanıyor; bu bölüm o
> tercihin neden zorunlu olduğunu ölçümle gösteriyor.

## Soru

Bir tel giriş noktası (CP), CAD geometrisinden **doğrudan** bulunabilir mi? Terminal blokları
tam tanımlı imalat özelliklerinden oluşur: silindirik delikler, prizmatik yarıklar, pahlar.
Eğer tel girişi geometrik olarak ayırt edilebilirse, öğrenmeye gerek kalmaz — ve böyle bir
dedektör **eğitim verisi gerektirmez**, dolayısıyla görülmemiş üreticiye anında genelleşir.

Bu, pratikte önemli bir soru: ML kolu **üretici-dışı** bölünmede F1 0.35'e düşüyor.

## Yöntem

Dört bağımsız sonda, hepsi **önceden yazılmış kill kriteriyle** (ürün metriği CP-F1'e katkı
≥ +0.02) ve aynı eşleşme konvansiyonuyla (eksene duyarlı: dik mesafe + ±40 mm eksenel pencere).

## Sonuçlar

### S1 — Saf geometrik dedektör

| dedektör | P | R | F1 |
|---|---|---|---|
| mesh konkavlığı (`geo_cp`) | 0.313 | 0.555 | **0.400** |
| B-rep silindir tespiti | 0.193 | 0.580 | 0.289 |
| B-rep + yarık adayları, 12 noktalı süzgeç taramasının en iyisi | 0.260 | 0.460 | **0.332** |

Aynı rejimde ML boru hattı **0.82**.

**Recall çözülebilir bir sorun.** B-rep'ten paralel düzlem çiftleriyle yarık/push-in adayları
eklenince aday-recall **0.564 → 0.860** çıktı. Kaçırılan GT'lerin **%93.3'ü hiçbir silindirik
yüzeye 3 mm'den yakın değildi** — yani tel girişlerinin çoğu delik değil, yarık.

**Precision çözülemedi.** Üç fiziksel süzgeç (boydan geçen delikler, baskın eksene dik olanlar,
r < 1 mm) yanlış pozitiflerin %79'unu eledi ama recall'un 0.198'ini götürdü. Terminal blokları
tam tel çapı aralığında vida, montaj, test-probu ve perçin delikleriyle doludur:
**69 gerçek girişe karşı 289 aynı görünen delik.**

### S2 — Mevcut özellikler hayatta kalan yanlış pozitiflerde

Gate'in operasyon noktasında hayatta kalan 263 yanlış pozitif incelendi. Mevcut 41 geometrik
özelliğin **en iyisi AUC 0.36 / 0.61** — yani neredeyse kör.

### S3 — Açıklığın içine bakmak

Işın-mesh kesişimiyle kanal profili çıkarıldı: serbest derinlik, kesit değişimi, daralma,
eksen sapması ve **kanal dibindeki segmentasyon sınıfı**. Hipotez fiziksel olarak makuldü:
tel girişi bir kontakta biter, alet ağzı bir kol/yay yuvasında.

Hipotez **yönü doğrulandı** — `dip_CT` (kanal dibinde Contact olasılığı) gerçek girişlerde
medyan **0.43**, yanlış pozitiflerde **0.22**. Ama ürün metriğine katkısı **+0.003 F1**
(gate AUC'sinde +0.024, operasyon noktasına ulaşmıyor).

### S4 — B-rep yüzey bilgisi

Tessellation'da atılan bilgi geri alındı: yüzey tipi, tam yarıçap, alan, 6 mm içindeki
düzlem/silindir/koni sayıları. Bazı özelliklerin gerçek sinyali var — gerçek girişler
geometrik olarak **daha zengin komşuluklarda** oturuyor (6 mm içindeki düzlem sayısı
AUC 0.296; TP medyan 21 vs FP 13). Ama mevcut setle **fazlalık**: CP-F1 katkısı **+0.005**.

### S5 — Bilgi duvarı testi (belirleyici)

Sınıflandırıcı üç rejimde eğitildi: aile-dışı (dürüst), parça-dışı, ve **rastgele kat** —
sonuncusunda model **skorlandığı parçaları eğitimde görüyor**, yani sızıntılı bir üst sınır.

| koşul | CP-F1 |
|---|---|
| aile-dışı (dürüst) | 0.7956 |
| parça-dışı | 0.7953 |
| **rastgele kat (sızıntılı üst sınır)** | **0.7989** |
| aday tavanı (mükemmel gate) | 0.9499 |

**Genelleştirme boşluğu +0.0033. Bilgi boşluğu +0.1510.**

Modele skorlandığı parçaları göstermek **+0.003** kazandırıyor. Yani kalan hata veri, kapasite
veya genelleştirme sorunu değil — **cevap modele gösterilen şeyde yok.**

## Sonuç

Tel girişi, çevresindeki aynı ölçülerdeki diğer deliklerden **geometrik olarak ayırt edilemez.**
Geometri açıklığı *bulabilir* (recall 0.86'ya çıkarılabildi) ama hangisinin tel girişi olduğunu
*söyleyemez*.

Ayrımı yapabilen tek bileşen **semantik segmentasyon ağıdır**: yüzeyin ne olduğunu şekline değil,
öğrenilmiş bağlama göre sınıflandırır. Bu, Scheffler'in boru hattında segmentasyon adımının neden
vazgeçilmez olduğunun deneysel kanıtıdır.

**Sınırın niceliksel ifadesi:** geometriyle ulaşılabilir gate tavanı ≈ 0.80, ve gate zaten
0.7956'da. "Mükemmel gate 0.9499" rakamı bir *oracle* sayısıdır ve geometrik girdiyle
erişilebilir değildir.

---

### Makbuzlar

`results/geo070_g1_otopsi.json` · `results/tel_e_kill.json` · `results/tel_g_kill.json` ·
`geo_brep_cp.py` · `geo_g2_yarik.py` · `tel_b_kanal_profili.py` · `tel_g_brep.py`
