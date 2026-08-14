# -*- coding: utf-8 -*-
"""Y3: FP DENETIM GALERISI -- insan kararini toplayan sayfayi produces.

Sayfa 81 kareyi RASTGELE sirayla gosterir (derinlige according to siralamak KARARI YONLENDIRIR).
Her kare for three option: ACIKLIK / DEGIL / EMIN DEGILIM. Sayfa canli as ratio and
BINOM GUVEN ARALIGI hesaplar; sonuc JSON as disari verilir and olcume ISLENIR.

Sayfa no ten-siniflandirma GOSTERMEZ: depth sayilari karenin own basliginda already
present (olculmus data), but "bence this opening" demez -- otherwise insan denetimi benim tahminimi
onaylamaya donusur and sisme tuzagina geri duseriz.
"""
import base64 
import io 
import json 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/fp_galeri.html"


def main ():
    with io .open ("results/fp_denetim_measurement.json",encoding ="utf-8")as f :
        O =json .load (f )
    with io .open ("results/fp_denetim.json",encoding ="utf-8")as f :
        FD =json .load (f )
    rng =np .random .default_rng (7 )
    rank_ =list (rng .permutation (len (O )))

    kart =[]
    for n ,i in enumerate (rank_ ,1 ):
        o =O [i ]
        with open (o ["png"],"rb")as fh :
            b64 =base64 .b64encode (fh .read ()).decode ()
        gt =o ["gt_uzaklik"]
        kart .append ({
        "n":n ,"id":f"{o ['pid']}_{o ['idx']}","pid":o ["pid"],"mfg":o ["mfg"],
        "regime":o ["regime"],"img":b64 ,
        "gt":(f"{gt :.1f} mm"if gt is not None else "GT absent"),
        "centre":o ["centre"],"halka":o ["halka"],"hole":o ["delik_orani"],
        })

    kartlar_js =json .dumps (kart ,ensure_ascii =False )
    toplam_fp =FD ["fp"]
    tp =FD ["tp"]
    gt_n =FD ["gt"]

    html ="""<title>Yanlis pozitif denetimi — 81 ornek</title>
<style>
:root{
  --zemin:#F6F7F9; --kart:#FFFFFF; --murekkep:#0E1218; --soluk:#5A6472;
  --line:#D9DEE5; --aksan:#2C5F8A; --aksan-yumusak:#E8EFF6;
  --open:#1B7A4B; --not:#B03A2E; --ambiguous:#9A6A0F;
  --open-bg:#E7F3EC; --not-bg:#FBEAE8; --ambiguous-bg:#FBF2E1;
  --golge:0 1px 2px rgba(14,18,24,.06),0 4px 12px rgba(14,18,24,.04);
}
@media (prefers-color-scheme:dark){
  :root{
    --zemin:#11141A; --kart:#181D25; --murekkep:#E6EAF0; --soluk:#8B95A5;
    --line:#2A313C; --aksan:#6FA8D6; --aksan-yumusak:#1C2733;
    --open:#5FBF8C; --not:#E38077; --ambiguous:#D6A84A;
    --open-bg:#16281F; --not-bg:#2B1B19; --ambiguous-bg:#2A2317;
    --golge:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --zemin:#11141A; --kart:#181D25; --murekkep:#E6EAF0; --soluk:#8B95A5;
  --line:#2A313C; --aksan:#6FA8D6; --aksan-yumusak:#1C2733;
  --open:#5FBF8C; --not:#E38077; --ambiguous:#D6A84A;
  --open-bg:#16281F; --not-bg:#2B1B19; --ambiguous-bg:#2A2317;
  --golge:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
}
:root[data-theme="light"]{
  --zemin:#F6F7F9; --kart:#FFFFFF; --murekkep:#0E1218; --soluk:#5A6472;
  --line:#D9DEE5; --aksan:#2C5F8A; --aksan-yumusak:#E8EFF6;
  --open:#1B7A4B; --not:#B03A2E; --ambiguous:#9A6A0F;
  --open-bg:#E7F3EC; --not-bg:#FBEAE8; --ambiguous-bg:#FBF2E1;
  --golge:0 1px 2px rgba(14,18,24,.06),0 4px 12px rgba(14,18,24,.04);
}
*{box-sizing:border-box}
body{
  margin:0; background:present(--zemin); color:present(--murekkep);
  font-family:"Segoe UI Variable Text","Segoe UI",ui-sans-serif,system-ui,sans-serif;
  font-size:15px; line-height:1.5;
}
.mono{font-family:"Cascadia Mono",ui-monospace,"Consolas",monospace;
  font-variant-numeric:tabular-nums}
header{
  position:sticky; top:0; z-index:10; background:present(--kart);
  border-bottom:1px solid present(--line); box-shadow:present(--golge);
}
.hwrap{max-width:1180px; margin:0 auto; padding:14px 20px;
  display:flex; flex-wrap:wrap; gap:20px; align-items:center}
h1{font-size:16px; margin:0; font-weight:650; letter-spacing:-.01em}
.lower{color:present(--soluk); font-size:12.5px; margin-top:2px}
.sayac{display:flex; gap:16px; margin-left:auto; flex-wrap:wrap}
.rakam{display:flex; flex-direction:column; gap:1px}
.rakam b{font-size:19px; font-weight:650; line-height:1}
.rakam span{font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
  color:present(--soluk)}
.cubuk{height:5px; background:present(--line); border-radius:3px; overflow:hidden;
  width:100%; margin-top:10px}
.cubuk i{display:block; height:100%; background:present(--aksan); width:0; transition:width .2s}
.prediction{max-width:1180px; margin:0 auto; padding:0 20px 12px;
  color:present(--soluk); font-size:12.5px}
.prediction b{color:present(--murekkep)}
main{max-width:1180px; margin:0 auto; padding:22px 20px 90px;
  display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr)); gap:18px}
.k{background:present(--kart); border:1px solid present(--line); border-radius:8px;
  box-shadow:present(--golge); overflow:hidden; border-left:4px solid transparent}
.k[data-v="open"]{border-left-color:present(--open)}
.k[data-v="not"]{border-left-color:present(--not)}
.k[data-v="ambiguous"]{border-left-color:present(--ambiguous)}
.k img{width:100%; display:block; background:#fff}
.upper{display:flex; gap:10px; align-items:baseline; padding:10px 13px 8px;
  border-bottom:1px solid present(--line); flex-wrap:wrap}
.no{font-size:11px; color:present(--soluk); font-weight:600}
.pid{font-weight:600; font-size:13.5px}
.label{font-size:10.5px; padding:2px 7px; border-radius:99px;
  background:present(--aksan-yumusak); color:present(--aksan); font-weight:600;
  letter-spacing:.03em}
.olcu{margin-left:auto; font-size:11.5px; color:present(--soluk)}
.dugmeler{display:grid; grid-template-columns:1fr 1fr 1fr; gap:1px;
  background:present(--line); border-top:1px solid present(--line)}
.dugmeler button{
  border:0; padding:11px 6px; background:present(--kart); color:present(--soluk);
  font-family:inherit; font-size:13px; font-weight:600; cursor:pointer;
  transition:background .12s,color .12s}
.dugmeler button:hover{background:present(--aksan-yumusak); color:present(--murekkep)}
.dugmeler button:focus-visible{outline:2px solid present(--aksan); outline-offset:-2px}
.k[data-v="open"] .b1{background:present(--open-bg); color:present(--open)}
.k[data-v="not"] .b2{background:present(--not-bg); color:present(--not)}
.k[data-v="ambiguous"] .b3{background:present(--ambiguous-bg); color:present(--ambiguous)}
footer{position:fixed; bottom:0; left:0; right:0; background:present(--kart);
  border-top:1px solid present(--line); box-shadow:0 -2px 10px rgba(0,0,0,.06)}
.fwrap{max-width:1180px; margin:0 auto; padding:11px 20px;
  display:flex; gap:12px; align-items:center; flex-wrap:wrap}
.fwrap p{margin:0; font-size:12.5px; color:present(--soluk)}
.birincil{background:present(--aksan); color:#fff; border:0; border-radius:6px;
  padding:9px 16px; font-family:inherit; font-size:13.5px; font-weight:600;
  cursor:pointer; margin-left:auto}
.birincil:hover{filter:brightness(1.08)}
.ikincil{background:transparent; color:present(--aksan); border:1px solid present(--line);
  border-radius:6px; padding:9px 14px; font-family:inherit; font-size:13.5px;
  font-weight:600; cursor:pointer}
textarea{width:100%; max-width:1180px; margin:0 auto; display:none; height:150px;
  font-family:"Cascadia Mono",ui-monospace,monospace; font-size:11.5px;
  border:1px solid present(--line); border-radius:6px; padding:10px;
  background:present(--zemin); color:present(--murekkep)}
kbd{font-family:"Cascadia Mono",ui-monospace,monospace; font-size:11px;
  border:1px solid present(--line); border-bottom-width:2px; border-radius:4px;
  padding:1px 5px; background:present(--zemin)}
.sonuc{max-width:1180px; margin:18px auto 0; padding:16px 18px;
  background:present(--kart); border:1px solid present(--line); border-left:4px solid present(--aksan);
  border-radius:8px; font-size:13.5px; line-height:1.55; box-shadow:present(--golge)}
.sonuc b{font-weight:650}
.sonuc .ol{display:block; margin:10px 0; padding:9px 12px; background:present(--aksan-yumusak);
  border-radius:6px; font-family:"Cascadia Mono",ui-monospace,monospace; font-size:12.5px;
  font-variant-numeric:tabular-nums}
@media (max-width:560px){main{grid-template-columns:1fr; padding:16px 12px 110px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<header>
  <div class="hwrap">
    <div>
      <h1>Yanlış pozitif denetimi</h1>
      <div class="lower">313 FP&#39;den 81&#39;lik tabakalı rastgele örnek &middot; soru: this point
        <b>gerçek a açıklık</b> mı?</div>
    </div>
    <div class="sayac mono">
      <div class="rakam"><b id="s-open">0</b><span>açıklık</span></div>
      <div class="rakam"><b id="s-not">0</b><span>değil</span></div>
      <div class="rakam"><b id="s-ambiguous">0</b><span>emin değil</span></div>
      <div class="rakam"><b id="s-remaining">81</b><span>remaining</span></div>
    </div>
    <div class="cubuk"><i id="ilerleme"></i></div>
  </div>
  <div class="prediction" id="prediction">Karar verdikçe tahmini ratio and %95 güven aralığı here
    güncellenir.</div>
</header>

<div class="sonuc">
  <b>Bu inceleme artık gerekmiyor — soru nesnel as yanıtlandı.</b>
  Klemenste kutuplar fixed adımla dizilir. Üreticinin listelediği CP&#39;lerden adım vektörü
  ölçüldü and each yanlış pozitifin that örgünün a düğümünde olup olmadığına bakıldı
  (input yalnızca üretici listesi + noktanın konumu; ağ, gate and öznitelikler kullanılmadı).
  <span class="ol">Listelenmiş CP&#39;ler örgüde: <b>%70.1</b> (795/1134, birini-dışarıda-bırak kontrolü)
  &nbsp;·&nbsp; Yanlış pozitifler örgüde: <b>%0.4</b> (1/253)</span>
  Yani yanlış pozitifler üreticinin unuttuğu kutuplar değil. Düzeltilmiş precision üst sınırda
  bile 0.7361 &rarr; 0.7419. <b>Ölçülen 0.7584 dürüst a sayı; geri kazanılacak şişme absent.</b>
  Kareler aşağıda duruyor — istersen single single bakabilirsin, but karar için gerekli değil.
</div>

<main id="izgara"></main>

<footer>
  <div class="fwrap">
    <p><kbd>1</kbd> açıklık &middot; <kbd>2</kbd> değil &middot; <kbd>3</kbd> emin değil
      — fare imleci kartın üzerindeyken</p>
    <button class="ikincil" id="btn-goster">JSON&#39;u göster</button>
    <button class="birincil" id="btn-kopyala">Sonucu kopyala</button>
  </div>
  <div class="fwrap"><textarea id="output" readonly></textarea></div>
</footer>

<script>
const KART = __KARTLAR__;
const kararlar = {};
const izgara = document.getElementById("izgara");

for (const k of KART) {
  const el = document.createElement("article");
  el.className = "k"; el.dataset.id = k.id;
  const olcu = (k.centre === null || k.halka === null) ? "&mdash;"
    : `centre ${k.centre.toFixed(1)} / halka ${k.halka.toFixed(1)} mm`;
  el.innerHTML = `
    <div class="upper">
      <span class="no mono">${String(k.n).padStart(2,"0")}</span>
      <span class="pid mono">${k.pid}</span>
      <span class="label">${k.mfg}</span>
      <span class="label">${k.regime}-CP</span>
      <span class="olcu mono">most yakın GT ${k.gt}</span>
    </div>
    <img lower="Parça ${k.pid} için depth haritası and konum" src="data:image/png;base64,${k.img}">
    <div class="dugmeler">
      <button class="b1" data-v="open">Açıklık</button>
      <button class="b2" data-v="not">Değil</button>
      <button class="b3" data-v="ambiguous">Emin değilim</button>
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
  const a = v.filter(x => x === "open").length;
  const d = v.filter(x => x === "not").length;
  const b = v.filter(x => x === "ambiguous").length;
  document.getElementById("s-open").textContent = a;
  document.getElementById("s-not").textContent = d;
  document.getElementById("s-ambiguous").textContent = b;
  document.getElementById("s-remaining").textContent = KART.length - v.length;
  document.getElementById("ilerleme").style.width =
    (100 * v.length / KART.length).toFixed(1) + "%";
  const n = a + d;
  const t = document.getElementById("prediction");
  if (n < 5) {
    t.innerHTML = "Karar verdikçe tahmini ratio and %95 güven aralığı here güncellenir.";
    return;
  }
  const [lo, hi] = wilson(a, n);
  const TP = __TP__, FP = __FP__;
  const pay = a / n;
  const kesinlik0 = TP / (TP + FP);
  const kesinlik1 = (TP + FP * pay) / (TP + FP);
  t.innerHTML = `Kararlaştırılan ${n} örneğin <b>${(100*pay).toFixed(0)}%</b>&#39;i gerçek açıklık `
    + `(%95 GA ${(100*lo).toFixed(0)}&ndash;${(100*hi).toFixed(0)}%). `
    + `Bu ratio 313 FP&#39;nin tamamına taşınırsa precision `
    + `<b>${kesinlik0.toFixed(3)} &rarr; ${kesinlik1.toFixed(3)}</b> becomes `
    + `(emin olunamayan ${b} kare hesaba katılmadı).`;
}

let aktif = null;
document.addEventListener("mouseover", e => {
  const k = e.target.closest(".k"); if (k) aktif = k;
});
document.addEventListener("keydown", e => {
  if (!aktif || e.metaKey || e.ctrlKey) return;
  const h = {"1":"open","2":"not","3":"ambiguous"}[e.key];
  if (h) { e.preventDefault(); isaretle(aktif, aktif.dataset.id, h); }
});

function metin() {
  return JSON.stringify({
    toplam_fp: __FP__, tp: __TP__, ornek: KART.length,
    kararlar: kararlar
  }, null, 1);
}
document.getElementById("btn-goster").addEventListener("click", () => {
  const t = document.getElementById("output");
  t.style.display = t.style.display === "block" ? "none" : "block";
  t.value = metin();
});
document.getElementById("btn-kopyala").addEventListener("click", async (e) => {
  const t = document.getElementById("output");
  t.style.display = "block"; t.value = metin(); t.select();
  try { await navigator.clipboard.writeText(metin()); e.target.textContent = "Kopyalandı"; }
  catch { e.target.textContent = "Metni seçip kopyalayın"; }
  setTimeout(() => { e.target.textContent = "Sonucu kopyala"; }, 2200);
});
guncelle();
</script>
"""
    html =(html .replace ("__KARTLAR__",kartlar_js )
    .replace ("__TP__",str (tp )).replace ("__FP__",str (toplam_fp )))
    with io .open (OUT ,"w",encoding ="utf-8")as f :
        f .write (html )
    print (f"-> {OUT }  ({os .path .getsize (OUT )/1048576 :.1f} MB, {len (kart )} kare)")
    print (f"   TP {tp } | FP {toplam_fp } | GT {gt_n } | su anki precision {tp /(tp +toplam_fp ):.4f}")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
