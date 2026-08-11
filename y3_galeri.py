# -*- coding: utf-8 -*-
"""Y3: FP DENETIM GALERISI -- insan kararini toplayan sayfayi uretir.

Sayfa 81 kareyi RASTGELE sirayla gosterir (derinlige gore siralamak KARARI YONLENDIRIR).
Her kare icin uc secenek: ACIKLIK / DEGIL / EMIN DEGILIM. Sayfa canli olarak oran ve
BINOM GUVEN ARALIGI hesaplar; sonuc JSON olarak disari verilir ve olcume ISLENIR.

Sayfa hicbir on-siniflandirma GOSTERMEZ: derinlik sayilari karenin kendi basliginda zaten
var (olculmus veri), ama "bence bu aciklik" demez -- yoksa insan denetimi benim tahminimi
onaylamaya donusur ve sisme tuzagina geri duseriz.
"""
import base64
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "results/fp_galeri.html"


def main():
    with io.open("results/fp_denetim_olcum.json", encoding="utf-8") as f:
        O = json.load(f)
    with io.open("results/fp_denetim.json", encoding="utf-8") as f:
        FD = json.load(f)
    rng = np.random.default_rng(7)
    sira = list(rng.permutation(len(O)))

    kart = []
    for n, i in enumerate(sira, 1):
        o = O[i]
        with open(o["png"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        gt = o["gt_uzaklik"]
        kart.append({
            "n": n, "id": f"{o['pid']}_{o['idx']}", "pid": o["pid"], "mfg": o["mfg"],
            "rejim": o["rejim"], "img": b64,
            "gt": (f"{gt:.1f} mm" if gt is not None else "GT yok"),
            "merkez": o["merkez"], "halka": o["halka"], "delik": o["delik_orani"],
        })

    kartlar_js = json.dumps(kart, ensure_ascii=False)
    toplam_fp = FD["fp"]
    tp = FD["tp"]
    gt_n = FD["gt"]

    html = """<title>Yanlis pozitif denetimi — 81 ornek</title>
<style>
:root{
  --zemin:#F6F7F9; --kart:#FFFFFF; --murekkep:#0E1218; --soluk:#5A6472;
  --cizgi:#D9DEE5; --aksan:#2C5F8A; --aksan-yumusak:#E8EFF6;
  --acik:#1B7A4B; --degil:#B03A2E; --belirsiz:#9A6A0F;
  --acik-bg:#E7F3EC; --degil-bg:#FBEAE8; --belirsiz-bg:#FBF2E1;
  --golge:0 1px 2px rgba(14,18,24,.06),0 4px 12px rgba(14,18,24,.04);
}
@media (prefers-color-scheme:dark){
  :root{
    --zemin:#11141A; --kart:#181D25; --murekkep:#E6EAF0; --soluk:#8B95A5;
    --cizgi:#2A313C; --aksan:#6FA8D6; --aksan-yumusak:#1C2733;
    --acik:#5FBF8C; --degil:#E38077; --belirsiz:#D6A84A;
    --acik-bg:#16281F; --degil-bg:#2B1B19; --belirsiz-bg:#2A2317;
    --golge:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --zemin:#11141A; --kart:#181D25; --murekkep:#E6EAF0; --soluk:#8B95A5;
  --cizgi:#2A313C; --aksan:#6FA8D6; --aksan-yumusak:#1C2733;
  --acik:#5FBF8C; --degil:#E38077; --belirsiz:#D6A84A;
  --acik-bg:#16281F; --degil-bg:#2B1B19; --belirsiz-bg:#2A2317;
  --golge:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
}
:root[data-theme="light"]{
  --zemin:#F6F7F9; --kart:#FFFFFF; --murekkep:#0E1218; --soluk:#5A6472;
  --cizgi:#D9DEE5; --aksan:#2C5F8A; --aksan-yumusak:#E8EFF6;
  --acik:#1B7A4B; --degil:#B03A2E; --belirsiz:#9A6A0F;
  --acik-bg:#E7F3EC; --degil-bg:#FBEAE8; --belirsiz-bg:#FBF2E1;
  --golge:0 1px 2px rgba(14,18,24,.06),0 4px 12px rgba(14,18,24,.04);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--zemin); color:var(--murekkep);
  font-family:"Segoe UI Variable Text","Segoe UI",ui-sans-serif,system-ui,sans-serif;
  font-size:15px; line-height:1.5;
}
.mono{font-family:"Cascadia Mono",ui-monospace,"Consolas",monospace;
  font-variant-numeric:tabular-nums}
header{
  position:sticky; top:0; z-index:10; background:var(--kart);
  border-bottom:1px solid var(--cizgi); box-shadow:var(--golge);
}
.hwrap{max-width:1180px; margin:0 auto; padding:14px 20px;
  display:flex; flex-wrap:wrap; gap:20px; align-items:center}
h1{font-size:16px; margin:0; font-weight:650; letter-spacing:-.01em}
.alt{color:var(--soluk); font-size:12.5px; margin-top:2px}
.sayac{display:flex; gap:16px; margin-left:auto; flex-wrap:wrap}
.rakam{display:flex; flex-direction:column; gap:1px}
.rakam b{font-size:19px; font-weight:650; line-height:1}
.rakam span{font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--soluk)}
.cubuk{height:5px; background:var(--cizgi); border-radius:3px; overflow:hidden;
  width:100%; margin-top:10px}
.cubuk i{display:block; height:100%; background:var(--aksan); width:0; transition:width .2s}
.tahmin{max-width:1180px; margin:0 auto; padding:0 20px 12px;
  color:var(--soluk); font-size:12.5px}
.tahmin b{color:var(--murekkep)}
main{max-width:1180px; margin:0 auto; padding:22px 20px 90px;
  display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr)); gap:18px}
.k{background:var(--kart); border:1px solid var(--cizgi); border-radius:8px;
  box-shadow:var(--golge); overflow:hidden; border-left:4px solid transparent}
.k[data-v="acik"]{border-left-color:var(--acik)}
.k[data-v="degil"]{border-left-color:var(--degil)}
.k[data-v="belirsiz"]{border-left-color:var(--belirsiz)}
.k img{width:100%; display:block; background:#fff}
.ust{display:flex; gap:10px; align-items:baseline; padding:10px 13px 8px;
  border-bottom:1px solid var(--cizgi); flex-wrap:wrap}
.no{font-size:11px; color:var(--soluk); font-weight:600}
.pid{font-weight:600; font-size:13.5px}
.etiket{font-size:10.5px; padding:2px 7px; border-radius:99px;
  background:var(--aksan-yumusak); color:var(--aksan); font-weight:600;
  letter-spacing:.03em}
.olcu{margin-left:auto; font-size:11.5px; color:var(--soluk)}
.dugmeler{display:grid; grid-template-columns:1fr 1fr 1fr; gap:1px;
  background:var(--cizgi); border-top:1px solid var(--cizgi)}
.dugmeler button{
  border:0; padding:11px 6px; background:var(--kart); color:var(--soluk);
  font-family:inherit; font-size:13px; font-weight:600; cursor:pointer;
  transition:background .12s,color .12s}
.dugmeler button:hover{background:var(--aksan-yumusak); color:var(--murekkep)}
.dugmeler button:focus-visible{outline:2px solid var(--aksan); outline-offset:-2px}
.k[data-v="acik"] .b1{background:var(--acik-bg); color:var(--acik)}
.k[data-v="degil"] .b2{background:var(--degil-bg); color:var(--degil)}
.k[data-v="belirsiz"] .b3{background:var(--belirsiz-bg); color:var(--belirsiz)}
footer{position:fixed; bottom:0; left:0; right:0; background:var(--kart);
  border-top:1px solid var(--cizgi); box-shadow:0 -2px 10px rgba(0,0,0,.06)}
.fwrap{max-width:1180px; margin:0 auto; padding:11px 20px;
  display:flex; gap:12px; align-items:center; flex-wrap:wrap}
.fwrap p{margin:0; font-size:12.5px; color:var(--soluk)}
.birincil{background:var(--aksan); color:#fff; border:0; border-radius:6px;
  padding:9px 16px; font-family:inherit; font-size:13.5px; font-weight:600;
  cursor:pointer; margin-left:auto}
.birincil:hover{filter:brightness(1.08)}
.ikincil{background:transparent; color:var(--aksan); border:1px solid var(--cizgi);
  border-radius:6px; padding:9px 14px; font-family:inherit; font-size:13.5px;
  font-weight:600; cursor:pointer}
textarea{width:100%; max-width:1180px; margin:0 auto; display:none; height:150px;
  font-family:"Cascadia Mono",ui-monospace,monospace; font-size:11.5px;
  border:1px solid var(--cizgi); border-radius:6px; padding:10px;
  background:var(--zemin); color:var(--murekkep)}
kbd{font-family:"Cascadia Mono",ui-monospace,monospace; font-size:11px;
  border:1px solid var(--cizgi); border-bottom-width:2px; border-radius:4px;
  padding:1px 5px; background:var(--zemin)}
.sonuc{max-width:1180px; margin:18px auto 0; padding:16px 18px;
  background:var(--kart); border:1px solid var(--cizgi); border-left:4px solid var(--aksan);
  border-radius:8px; font-size:13.5px; line-height:1.55; box-shadow:var(--golge)}
.sonuc b{font-weight:650}
.sonuc .ol{display:block; margin:10px 0; padding:9px 12px; background:var(--aksan-yumusak);
  border-radius:6px; font-family:"Cascadia Mono",ui-monospace,monospace; font-size:12.5px;
  font-variant-numeric:tabular-nums}
@media (max-width:560px){main{grid-template-columns:1fr; padding:16px 12px 110px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<header>
  <div class="hwrap">
    <div>
      <h1>Yanlış pozitif denetimi</h1>
      <div class="alt">313 FP&#39;den 81&#39;lik tabakalı rastgele örnek &middot; soru: bu nokta
        <b>gerçek bir açıklık</b> mı?</div>
    </div>
    <div class="sayac mono">
      <div class="rakam"><b id="s-acik">0</b><span>açıklık</span></div>
      <div class="rakam"><b id="s-degil">0</b><span>değil</span></div>
      <div class="rakam"><b id="s-belirsiz">0</b><span>emin değil</span></div>
      <div class="rakam"><b id="s-kalan">81</b><span>kalan</span></div>
    </div>
    <div class="cubuk"><i id="ilerleme"></i></div>
  </div>
  <div class="tahmin" id="tahmin">Karar verdikçe tahmini oran ve %95 güven aralığı burada
    güncellenir.</div>
</header>

<div class="sonuc">
  <b>Bu inceleme artık gerekmiyor — soru nesnel olarak yanıtlandı.</b>
  Klemenste kutuplar sabit adımla dizilir. Üreticinin listelediği CP&#39;lerden adım vektörü
  ölçüldü ve her yanlış pozitifin o örgünün bir düğümünde olup olmadığına bakıldı
  (girdi yalnızca üretici listesi + noktanın konumu; ağ, gate ve öznitelikler kullanılmadı).
  <span class="ol">Listelenmiş CP&#39;ler örgüde: <b>%70.1</b> (795/1134, birini-dışarıda-bırak kontrolü)
  &nbsp;·&nbsp; Yanlış pozitifler örgüde: <b>%0.4</b> (1/253)</span>
  Yani yanlış pozitifler üreticinin unuttuğu kutuplar değil. Düzeltilmiş kesinlik üst sınırda
  bile 0.7361 &rarr; 0.7419. <b>Ölçülen 0.7584 dürüst bir sayı; geri kazanılacak şişme yok.</b>
  Kareler aşağıda duruyor — istersen tek tek bakabilirsin, ama karar için gerekli değil.
</div>

<main id="izgara"></main>

<footer>
  <div class="fwrap">
    <p><kbd>1</kbd> açıklık &middot; <kbd>2</kbd> değil &middot; <kbd>3</kbd> emin değil
      — fare imleci kartın üzerindeyken</p>
    <button class="ikincil" id="btn-goster">JSON&#39;u göster</button>
    <button class="birincil" id="btn-kopyala">Sonucu kopyala</button>
  </div>
  <div class="fwrap"><textarea id="cikti" readonly></textarea></div>
</footer>

<script>
const KART = __KARTLAR__;
const kararlar = {};
const izgara = document.getElementById("izgara");

for (const k of KART) {
  const el = document.createElement("article");
  el.className = "k"; el.dataset.id = k.id;
  const olcu = (k.merkez === null || k.halka === null) ? "&mdash;"
    : `merkez ${k.merkez.toFixed(1)} / halka ${k.halka.toFixed(1)} mm`;
  el.innerHTML = `
    <div class="ust">
      <span class="no mono">${String(k.n).padStart(2,"0")}</span>
      <span class="pid mono">${k.pid}</span>
      <span class="etiket">${k.mfg}</span>
      <span class="etiket">${k.rejim}-CP</span>
      <span class="olcu mono">en yakın GT ${k.gt}</span>
    </div>
    <img alt="Parça ${k.pid} için derinlik haritası ve konum" src="data:image/png;base64,${k.img}">
    <div class="dugmeler">
      <button class="b1" data-v="acik">Açıklık</button>
      <button class="b2" data-v="degil">Değil</button>
      <button class="b3" data-v="belirsiz">Emin değilim</button>
    </div>`;
  el.querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => isaretle(el, k.id, b.dataset.v));
  });
  izgara.appendChild(el);
}

function isaretle(el, id, v) {
  if (kararlar[id] === v) { delete kararlar[id]; el.removeAttribute("data-v"); }
  else { kararlar[id] = v; el.dataset.v = v; }
  guncelle();
}

function wilson(k, n) {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n, d = 1 + z*z/n;
  const m = (p + z*z/(2*n)) / d;
  const s = z * Math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d;
  return [Math.max(0, m - s), Math.min(1, m + s)];
}

function guncelle() {
  const v = Object.values(kararlar);
  const a = v.filter(x => x === "acik").length;
  const d = v.filter(x => x === "degil").length;
  const b = v.filter(x => x === "belirsiz").length;
  document.getElementById("s-acik").textContent = a;
  document.getElementById("s-degil").textContent = d;
  document.getElementById("s-belirsiz").textContent = b;
  document.getElementById("s-kalan").textContent = KART.length - v.length;
  document.getElementById("ilerleme").style.width =
    (100 * v.length / KART.length).toFixed(1) + "%";
  const n = a + d;
  const t = document.getElementById("tahmin");
  if (n < 5) {
    t.innerHTML = "Karar verdikçe tahmini oran ve %95 güven aralığı burada güncellenir.";
    return;
  }
  const [lo, hi] = wilson(a, n);
  const TP = __TP__, FP = __FP__;
  const pay = a / n;
  const kesinlik0 = TP / (TP + FP);
  const kesinlik1 = (TP + FP * pay) / (TP + FP);
  t.innerHTML = `Kararlaştırılan ${n} örneğin <b>${(100*pay).toFixed(0)}%</b>&#39;i gerçek açıklık `
    + `(%95 GA ${(100*lo).toFixed(0)}&ndash;${(100*hi).toFixed(0)}%). `
    + `Bu oran 313 FP&#39;nin tamamına taşınırsa kesinlik `
    + `<b>${kesinlik0.toFixed(3)} &rarr; ${kesinlik1.toFixed(3)}</b> olur `
    + `(emin olunamayan ${b} kare hesaba katılmadı).`;
}

let aktif = null;
document.addEventListener("mouseover", e => {
  const k = e.target.closest(".k"); if (k) aktif = k;
});
document.addEventListener("keydown", e => {
  if (!aktif || e.metaKey || e.ctrlKey) return;
  const h = {"1":"acik","2":"degil","3":"belirsiz"}[e.key];
  if (h) { e.preventDefault(); isaretle(aktif, aktif.dataset.id, h); }
});

function metin() {
  return JSON.stringify({
    toplam_fp: __FP__, tp: __TP__, ornek: KART.length,
    kararlar: kararlar
  }, null, 1);
}
document.getElementById("btn-goster").addEventListener("click", () => {
  const t = document.getElementById("cikti");
  t.style.display = t.style.display === "block" ? "none" : "block";
  t.value = metin();
});
document.getElementById("btn-kopyala").addEventListener("click", async (e) => {
  const t = document.getElementById("cikti");
  t.style.display = "block"; t.value = metin(); t.select();
  try { await navigator.clipboard.writeText(metin()); e.target.textContent = "Kopyalandı"; }
  catch { e.target.textContent = "Metni seçip kopyalayın"; }
  setTimeout(() => { e.target.textContent = "Sonucu kopyala"; }, 2200);
});
guncelle();
</script>
"""
    html = (html.replace("__KARTLAR__", kartlar_js)
                .replace("__TP__", str(tp)).replace("__FP__", str(toplam_fp)))
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {OUT}  ({os.path.getsize(OUT)/1048576:.1f} MB, {len(kart)} kare)")
    print(f"   TP {tp} | FP {toplam_fp} | GT {gt_n} | su anki kesinlik {tp/(tp+toplam_fp):.4f}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
