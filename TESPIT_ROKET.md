# TESPİT F1 — ROKET LİSTESİ

> Taban: **tespit 0.7584** (kesinlik 0.745 / recall 0.774) · robot ≈ 0.805 × tespit
> Tez çizgisi: ağ mimarisi, remesh, `v_o` türetmesi ve 5-sınıf kaybı **hiçbir maddede değişmez**.
> Her madde ÖNCEDEN yazılmış kill ile ölçülür; sonucu gördükten sonra bar değişmez.

---

## Kaybın ayrıştırması (ölçüldü)

| sızıntı | büyüklük | ne demek |
|---|---|---|
| **A** — adaysız GT | 188 (%14.2) | ağ orada CE/CT olasılığı üretmiyor |
| **B** — gate'in reddettiği doğru aday | 248 | aday var, gate atıyor. Ayırt edici özellik: **düşük oy** (1.33 vs 1.98) |
| **C** — yanlış pozitif | 313 | gerçek açıklık ama tel girişi değil |

**Tavanlar:** aday havuzu recall'ı 0.8579 · kâhin gate 0.9370 · el-yapımı uzayın Bayes tavanı ~0.86

---

## SIZINTI B — en büyük fırsat, en ucuz kollar

Kayıtta yazılı ve bugün doğrulandı: **gate'in 13 özelliğinden yalnız `votes` dürüst
geometri bölmesinde transfer ediyor** (AUC düşüşü 0.018; diğerleri 0.12–0.18). Yani
elimizdeki tek sağlam sinyal oy — ve reddedilen doğru adayların ayırt edici özelliği tam
olarak düşük oy. B kolları bu sinyali **keskinleştirir**.

### B-1 · 8 üyeli topluluk *(FB-3'te, koşuyor)*
4 üyeyle oy ∈ {1,2,3,4}; 8 üyeyle {1..8}. Genelleşen tek sinyalin çözünürlüğü **ikiye
katlanır**. Ayrıca birleşim daha çok aday bulur (sızıntı A'ya da dokunur).
**Kill:** tespit +0.01 ve GA>0.

### B-2 · Eksen-farkındalıklı oy havuzlaması *(FB-3'te)*
`_vote2` düz Öklid 5mm kullanıyor; ürünün puanlayıcısı `big_arbiter.greedy` ise
eksen-farkındalıklı. Ürün kendi içinde tutarsız. Ölçüldü: birleşme %67.7 → **%70.7**.
Yanal 3mm sınırı komşu kutupları (adım 3.5–6mm) korur, derinliği serbest bırakır.
**Kill:** aynı.

### B-3 · GÜVEN-AĞIRLIKLI OY — *en ucuz untried kol*
Şu an oy = kaç üye buldu (tamsayı). 0.92 güvenle bulan üye ile 0.31 ile bulan **aynı**
sayılıyor. Güven-ağırlıklı oy sürekli ve daha ince bir sinyal verir — ve genelleşen tek
özelliğin **rafine hâli** olduğu için transfer riski düşük.
**Maliyet:** ~15 dk (türetme zaten var, yalnız yeni sütun).
**Kill:** tespit +0.01 ve GA>0; üretici-dışı düşmemeli.

### B-4 · Parça başına RECALL TABANI
Top-K başarısız oldu çünkü *tam* K seçmeye zorluyordu (K yanlışsa eşikten kötü). **Taban**
farklı: "hiçbir parçada N'den az aday kabul etme". Yanlış K'nın cezası yok, yalnız
aşırı-sessiz parçalar kurtulur.
**Maliyet:** ~15 dk. **Kill:** aynı.

---

## SIZINTI C — tek açık kol

### C-1 · Tel/alet olasılığı gate'e öznitelik *(FB-3'te, koşuyor)*
Teşhis: tezin `Contact` sınıfı *"Kontaktierung bzw. Werkzeugeinschub"* — tel ve alet **tek
sınıf**; ağ ayrımı **silmek üzere** eğitildi. Yardımcı kafa o ayrımı geri veriyor. Gate'in
bugüne kadar hiç sahip olmadığı bilgi ve doğrudan 313 FP'yi hedefliyor.
**Kill:** aynı.

> C için başka açık kol yok: dokuz bağımsız gate mekanizması (öznitelik seçimi, dönüşüm,
> model sınıfı, ağırlık, eşik, örgü, boşluk grafı, tip yönlendirme, tip-içi eşik) kapandı.

---

## SIZINTI A — önce teşhis, sonra karar

### A-1 · 188 adaysız CP ULAŞILABİLİR mi? *(teşhis, ucuz)*
CP ekseni boyunca dışarıdan ışın at: malzemeye çarpmadan CP'ye 2mm yaklaşabiliyorsa **açık
kanal** var (ağ hatası, kazanılabilir); yaklaşamıyorsa CP malzemenin arkasında (**yapısal**,
hiçbir yüzey yöntemi bulamaz — üretici iç kontağı listelemiş).
**Neden önemli:** kapalı oranı yüksekse aday havuzu tavanı 0.8579 **sahte düşük** ve
recall'ımız göründüğünden iyi. Bu, hedefin gerçekçiliğini yeniden tanımlar.
**Maliyet:** ~20 dk (kendi Möller-Trumbore ışınımız; `trimesh.ray` rtree olmadan patlar).

### A-2 · Üye başına eşik kalibrasyonu
Tüm üyeler aynı `min_v`/`vertex_conf` kullanıyor. Sistematik olarak daha düşük güvenli bir
üye daha az aday üretiyor ve yalnız onun gördüğü açıklıklar kayboluyor. Her üyenin eşiğini
aday sayısını eşitleyecek şekilde kalibre etmek, tek üyenin zayıf gördüğü açıklıkları
kurtarır. **A-1 "açık kanal" oranı yüksek çıkarsa** anlamlı.
**Maliyet:** ~30 dk. **Kill:** aynı.

---

## ÇAPRAZ KESEN

### X-1 · Üretici-değişmez gate *(daha büyük yatırım)*
Ölçülen patoloji: havuzlanmışı iyileştiren her kol görülmemiş üreticiyi çökertiyor — gate
**ezberliyor**. Doğru çare düzenlileştirme değil, **alan-değişmezliği**: gradyan-tersleme
ile üreticiyi tahmin edemeyen bir temsil öğrenmek. RF gradyan taşımadığı için sinirsel bir
gate gerekir.
**Maliyet:** ~2 saat. **Ön koşul:** B ve C kolları bittikten sonra, çünkü onlar taban değiştirir.

### X-2 · Veri *(ölçülmüş fiyat etiketi, dış bağımlılık)*
`F1 = 0.5295 + 0.0330·ln(grup)`, doymuyor. 0.78 için 2.0×, 0.80 için 3.8× korpus.
Ve bileşim önemli: aynı hacimde karışık üretici **+0.0443**.
**Talep:** terminal block + STEP + kesin ConnectionPoints; en az 300, tercihen 900+ yeni
geometri grubu; **indirmeden ÖNCE 60/20/20, son %20 mühürlü.**

---

## SIRA

| # | kol | maliyet | durum |
|---|---|---|---|
| 1 | B-1 + B-2 + C-1 (FB-3, üçü tek koşuda) | eğitim + 40 dk | **koşuyor** |
| 2 | B-3 güven-ağırlıklı oy | 15 dk | sırada |
| 3 | A-1 ulaşılabilirlik teşhisi | 20 dk | sırada |
| 4 | B-4 recall tabanı | 15 dk | sırada |
| 5 | A-2 üye eşik kalibrasyonu | 30 dk | A-1'e bağlı |
| 6 | X-1 üretici-değişmez gate | 2 sa | 1–4 bitince |

**Makine disiplini:** aynı anda TEK ağır iş. Bugün beş süreç aynı anda koştu ve her şeyi
yavaşlattı; sebebi Git Bash `ps`'inin komut satırını göstermemesiydi (`grep` hep 0 döndü,
`pkill -f` hiçbir şey öldürmedi). Durum kontrolü **PowerShell `Get-CimInstance Win32_Process`**
ile yapılır.
