# -*- coding: utf-8 -*-
"""D5-3 ONKOSULU: YASAK GRUPLARI YENI ANAHTAR UZAYINA TASI (capraz sizinti kontrolu).

SORUN (2026-08-04'te fark edildi, sessiz sizinti):
`protokol.egitim_maskesi` gruplari `_strict_geometry_keys.json`'dan okur. O dosyadaki
anahtarlar ESKI ureticiden (`[8.1, 50.4, 71.8]|v12|f13|c19|p161|h...`). Yeni korpusun
anahtarlari ise `geometri_anahtar.anahtar()`'dan geliyor ve bicimi TAMAMEN FARKLI
(`b16.101.144|d31|h5.10.20...`).

Iki anahtar uzayi **asla string olarak eslesmez**. `geometri_anahtar.dogrula()` yeni
uretecin eski GRUPLAMAYI yeniden urettigini kanitlasa bile bu yetmez: gruplama denkligi
!= anahtar esitligi. Yani OLCUM ya da LOCKED parcasinin IKIZI olan yeni bir parca, iki
ayri uzayda durdugu icin `egitim_maskesi`'nden YAKALANMADAN gecer ve egitime girer.
Sonuc: manset SESSIZCE SISER -- tam da [[geometry-twin-leakage]]'in ogrettigi hata.

COZUM (ucuz olan): tum korpusu yeniden anahtarlamak yerine YALNIZ YASAK gruplarin
uyelerini (610 parca) yeni uretecle anahtarla. Yeni parcanin anahtari bu kumeye
dusuyorsa egitim disi birakilir. Eski parcalarin mevcut mekanizmasi AYNEN korunur.

Cikti: results/_geo_yasak.json  ->  {pid: yeni_anahtar}  (+ anahtar kumesi D5-3'te kullanilir)
"""
import io
import json
import os
import sys
import time

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CIKTI = "results/_geo_yasak.json"


def main():
    import protokol
    protokol.tez_dogrula()
    import olcum_kumesi as OK
    from big_arbiter import eligible
    from geometri_anahtar import anahtar
    from infer_step_cp import step_to_mesh

    gk = OK.geo_anahtarlari()
    yasak_grup = set(OK.locked_gruplari())
    D, _ = OK.kume("results/_der_tam.pkl")
    yasak_grup |= {r["geo"] for r in D}
    hedef_pid = {p for p, g in gk.items() if g in yasak_grup}
    yol = {p: s for _, p, _, s in eligible() if p in hedef_pid}
    print(f"YASAK grup {len(yasak_grup)} | uye parca {len(hedef_pid)} | "
          f"STEP'i bulunan {len(yol)}")

    OUT = {}
    if os.path.exists(CIKTI):
        OUT = json.load(io.open(CIKTI, encoding="utf-8"))
        print(f"devam: {len(OUT)} zaten anahtarlanmis")
    kalan = sorted(p for p in yol if p not in OUT)
    t0 = time.time(); hata = 0
    for k, p in enumerate(kalan, 1):
        if k % 25 == 0:
            hiz = (time.time() - t0) / k
            print(f"  {k}/{len(kalan)}  ({hiz:.1f}s/parca, kalan ~{hiz*(len(kalan)-k)/60:.0f} dk, "
                  f"hata {hata})", flush=True)
            with io.open(CIKTI, "w", encoding="utf-8") as f:
                json.dump(OUT, f)
        try:
            V, F = step_to_mesh(yol[p])
            OUT[p] = anahtar(V, F)
        except Exception as e:
            hata += 1
            if hata <= 5:
                print(f"    {p}: {type(e).__name__}: {str(e)[:60]}", flush=True)
    with io.open(CIKTI, "w", encoding="utf-8") as f:
        json.dump(OUT, f)

    # SAGLAMLIK KONTROLU: yeni anahtar, ESKI yasak gruplamayi koruyor mu?
    # Ayni eski grupta olan iki parca yeni anahtarda da ayni olmali; degilse yeni
    # uretec bu bolgede AYIRIYOR demektir ve koruma eksik kalir -- gorunur olsun.
    ters = {}
    for p, a in OUT.items():
        ters.setdefault(gk.get(p, "?"), set()).add(a)
    bolunen = {g: len(s) for g, s in ters.items() if len(s) > 1}
    print(f"\n{len(OUT)} parca anahtarlandi | hata {hata} | {time.time()-t0:.0f}s")
    print(f"  ESKI grup -> YENI anahtar: {len(ters)} gruptan {len(bolunen)}'i BOLUNDU")
    if bolunen:
        print(f"    (bolunen gruplar korumayi zayiflatir; en cok bolunen "
              f"{sorted(bolunen.values(), reverse=True)[:5]})")
    print(f"  TEKIL YASAK ANAHTAR: {len(set(OUT.values()))}  -> {CIKTI}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
