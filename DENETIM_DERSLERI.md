# DENETİM ANALİZİ — 2026-07-31

Bu belge kod denetiminin sonucunu **ve** bu oturumda yapılan hataların desen analizini taşır.
Amaç suçlama değil: aynı desenler tekrar edeceği için, her birine **tekrarı engelleyen bir
mekanizma** bağlandı.

---

## A. Statik denetim sonucu

| kontrol | sonuç |
|---|---|
| 378 dosya derleme | **0 sözdizimi hatası** |
| tanımsız isim (`undefined name`) | **0** — çökme seviyesinde hata yok |
| üretim dosyalarında kayıp değişken (f-string) | **0** (4 uyarı var, hepsi gereksiz `f` öneki) |
| kullanılmayan import | 113 (kozmetik) |
| kullanılmayan yerel değişken | 38 — üretimdeki 4'ü incelendi, hiçbiri düşen hesap değil |
| `cp_config` ↔ disk tutarlılığı | **TUTARLI** (4 checkpoint mevcut, gate 18 sütun = üretim 18 sütun) |

**Gerçek bulgu tek bir sınıftaydı:** üretim kodunda **27 sessiz istisna yutma**, bunların
8'i `wire_gate` içinde — orada bir hata özelliği sessizce nötre düşürür.

---

## B. Hata desenleri — her biri bu oturumda EN AZ İKİ KEZ tekrarladı

### 1. Sessiz nötrleşme *(3 kez)*
Bir bileşen hata alır, sessizce "zararsız" bir değer döner, çıktı makul görünür, kimse fark etmez.

- `rtree` kurulu değildi → `trimesh.contains()` her çağrıda patlıyordu, `except: continue` yutuyordu
  → iki fonksiyon **sessiz no-op**'a dönüşmüştü (önceki oturum).
- `step_path` 3 çağrının 1'ine geçmiyordu → 4 B-rep sütunu **sessizce sıfır**.
- `WG_FIZ_FEATS` yalnız ortam değişkeninden okunsaydı → üretimde **sessizce kapalı** kalacaktı
  (robot hattı o değişkeni set etmiyor).

**Mekanizma:** `wire_gate.FALLBACK` sayacı — her nötr-dönüş yolu sayılıyor, `fallback_ozet()`
ile okunuyor, `test_sessiz_notrlesme_sayiliyor` ile kilitli. Bayraklar artık `cp_config`'ten
okunuyor, ortam değişkeni yalnızca deney için ezer.

### 2. Yanlış istatistik → yanlış hüküm *(3 kez)*
Ölçüt yanlış seçilince sağlam bir yöntem "ölü" ilan edildi.

- Renk hizalamasında **iç-nokta oranı** kullandım; 19.680 aday çiftin çoğu inşaat gereği yanlış
  olduğu için oran hiçbir zaman yükselemezdi. Doğru ölçüt **artık**tı — düzeltince 5/15 parça
  zaten <0.5 mm hizalanmıştı.
- Kısa-yol AUC formülü ikili özelliklerde şişiyordu (0.953 sanılan gerçek 0.563).
- Sıra eşlemesi "metal yüzler daha derinde olmalı" gibi **dolaylı bir vekille** sınanıp
  reddedilmişti; **doğrudan** sınama vardı (iki taraf da yarıçap veriyor) ve 13'te 12 tuttu.

**Kural:** dolaylı vekil yerine **doğrudan** sınama varsa o kullanılır; her AUC'ye null testi.

### 3. Tek deneyde iki değişken *(3 kez)*
- `fiz18 vs rt2` hem yeni özellikleri hem güncel hattı değiştiriyordu → **üçüncü kol** (`fiz13`)
  eklendi; özellik +0.063, veri gürültü çıktı. Eklemeseydim özelliğin gerçek gücünü göremezdim.
- `J` ablasyonu hem konum ortalamasını hem oy sayma anlamını değiştiriyordu.
- İki gate farklı aday havuzlarında kıyaslanmıştı.

**Kural:** iki gate **aynı aday havuzunda** karşılaştırılır; bir ablasyon **tek** değişken değiştirir.

### 4. Seçim kümesinde ölçüm *(2 kez, ikisi de üründen geri alındı)*
- `L2` DEV'de +0.0066 kazandı, ayrık VAL'de **−0.0111** kaybetti → geri alındı.
- `J` havuzlanmış +0.0026, GA sıfırı içeriyor → kazanç sayılmadı.

**Mekanizma:** DEV (karar) / VAL (sınav) / **LOCKED (harcanmadı)** üçlü bölmesi + eşli bootstrap.

### 5. Yanlışlanamayan doğrulama *(2 kez)*
- "Süreçleri öldürdüm" dedim, doğrulamadım → bugün **4 zombi süreç** aynı dosyaya yazıyordu
  (`pkill` Windows'ta python.exe'ye işlemiyor).
- Determinizm testi tek süreç içinde koşuyordu, süreçler arası farkı göremezdi.

**Kural:** her "yaptım" iddiası, iddiayı **yanlışlayabilecek** bir kontrolle biter.

### 6. Yukarı akış düzeltilince aşağı akış bayatlar *(1 kez ama pahalı)*
Yarıçap 3.5 kat küçük hesaplanıyordu; düzeltilince ona göre ayarlanmış `r_range` (12 mm gerçekte
~42 mm demekti) ve `max_off` anlamını yitirdi. Tarandı: 8 ayar, yayılım 0.003 — kapılar suçlu
değilmiş, ama **bakılması** şarttı.

---

## C. En pahalı tek hata sınıfı: ölçüm aleti

Bu oturumda **ürün hatası 2**, **ölçüm hatası 6** çıktı. Ve raporlanan sayı ölçüm düzeltmeleriyle
0.779 → 0.673'e *düştü*, ürün ise 0.673 → 0.744'e *çıktı*.

Yani: **alet bozukken ürünü iyileştirdiğini sanmak, ürünü bozmaktan daha olasıydı.** Gecenin
ilk yarısında "kazanç" diye raporladığım iki kolun ikisi de sahteydi.

---

## D. Denetimden çıkan somut düzeltmeler *(bu oturumda uygulandı)*

1. `wire_gate.FALLBACK` sayacı + `fallback_ozet()` + test.
2. Bayraklar `cp_config`'ten okunuyor (`_fiz_default`, `_ek_default`) + tutarlılık testi
   (`config bayrağı == dağıtılan gate sütun sayısı`).
3. `apply` / `apply_spatial` model genişliğine göre sütun kırpar → eski modeller çalışmaya devam eder.
4. `PRODUCT_MODEL.md` üst bloğu üründen sapmıştı (`min_v 30` yazıyordu, gerçek 4; F1 merdiveni
   sızıntılı makbuzdan) → baştan yazıldı.
5. `brep_axes` yarıçapı çember oturtmayla düzeltildi (metinle **327/327**, medyan hata 0.0000 mm).
6. Süreç öldürme artık `Stop-Process` + **kalan sayısı doğrulaması**.

**Test sayısı 104 → 108.** Yeni testlerin hepsi bir hatayı değil, bir **hata sınıfını** kilitliyor.
