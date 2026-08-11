# -*- coding: utf-8 -*-
"""OLCUM TUZAKLARI -- her biri gercekten yasandi, her biri yanlis bir hukme yol acti.

Bunlar 'kod calisiyor mu' testi degil; 'olctugum sey olcmek istedigim sey mi' testi.
"""
import os
import sys

import numpy as np
import pytest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sina_kume import W, f1_rejim, f1w


def _satir(rejim, tp, fp, fn, n=1):
    return [(rejim, tp, fp, fn)] * n


def test_f1w_TEK_REJIMDE_o_rejimin_F1ini_verir():
    """DUZELTILDI 2026-08-03. Bu test ONCE ters yonde civiliyordu.

    Eski hali `f1w(cok) == W["cok"] * dogru` diye ISRAR ediyor ve yorumunda "bu davranis
    YANLIS DEGIL ama TUZAK" yaziyordu. Denetimde bit-duzeyinde gorulduki YANLIS: manset,
    DOGAL OLARAK tek rejimli bolmelerde (o parcalarin hepsi dusuk-CP) ayni carpani yiyordu.
    ATANMAMIS 0.6635 raporlandi, gercegi 0.7413; seri 16 0.8333 -> 0.9310; seri 17
    0.6531 -> 0.7297. Ucunde de rapor/0.895 == dusuk_CP_F1 TAM tutuyordu.

    DOGRU DAVRANIS: yalniz bir rejimin verisi varsa korpus-agirlikli bir sayi RAPORLANAMAZ;
    o bolmenin F1'i, var olan rejimin F1'idir. Agirliklar VAR OLAN rejimler uzerinden
    yeniden normalize edilir.
    """
    cok = _satir("cok", 3, 1, 2, 10)
    dogru = f1_rejim(cok)["F1"]["cok"]
    assert abs(f1w(cok) - dogru) < 1e-9, "tek rejimde f1w o rejimin F1'ini vermeli"
    dusuk = _satir("dusuk", 8, 2, 1, 10)
    assert abs(f1w(dusuk) - f1_rejim(dusuk)["F1"]["dusuk"]) < 1e-9


def test_f1_rejim_dogru_kirilim_ve_agirlikli_toplam_verir():
    rows = _satir("cok", 3, 1, 2, 10) + _satir("dusuk", 8, 2, 1, 10)
    r = f1_rejim(rows)
    assert r["n"] == {"dusuk": 10, "cok": 10}
    assert abs(r["agirlikli_F1"] - f1w(rows)) < 1e-9, "agirlikli toplam f1w ile tutmuyor"
    for k in ("dusuk", "cok"):
        tek = f1_rejim([x for x in rows if x[0] == k])["F1"][k]
        assert abs(r["F1"][k] - tek) < 1e-9, f"{k} kirilimi alt kumeyle tutmuyor"


def test_f1_rejim_tek_rejimde_agirligi_YENIDEN_NORMALLESTIRIR():
    """Alt kume tek rejim iceriyorsa agirlikli deger o rejimin kendi F1'i olmali (0.105 ile
    carpilmis hali degil) -- f1w'nin tuzagina dusmeyen davranis."""
    cok = _satir("cok", 3, 1, 2, 10)
    r = f1_rejim(cok)
    assert abs(r["agirlikli_F1"] - r["F1"]["cok"]) < 1e-9
    assert r["F1"]["dusuk"] is None


def test_carpik_kumede_HAM_ve_AGIRLIKLI_PR_ayrisir():
    """Puanlanan kumede cok-CP %30, korpusta %10.5. Ham P/R korpusu temsil etmez; ayni tabloda
    agirlikli F1 ile ham P/R yan yana konursa IKI FARKLI evren raporlanmis olur."""
    rows = _satir("cok", 1, 9, 9, 30) + _satir("dusuk", 9, 1, 1, 70)
    r = f1_rejim(rows)
    assert r["ham_kesinlik"] < r["agirlikli_kesinlik"], "carpiklik etkisi kaybolmus"
    assert abs(r["agirlikli_kesinlik"] - (W["dusuk"] * 0.9 + W["cok"] * 0.1)) < 1e-9


def test_rejim_agirliklari_korpusla_tutuyor():
    """W, korpus oranlarini temsil eder (dusuk 0.895 / cok 0.105). Degisirse manset degisir."""
    assert abs(sum(W.values()) - 1.0) < 1e-9
    assert abs(W["dusuk"] - 0.895) < 1e-6 and abs(W["cok"] - 0.105) < 1e-6


def test_LOCKED_kirliligi_CIKARILANLARI_da_kapsar():
    """2026-08-02 DENETIM BULGUSU: dogrudan kullanilip puanlamadan CIKARILAN LOCKED parcalar
    yeniden 'temiz' sayiliyordu.

    Sebep: `kullanilan_geo` yalniz KALAN parcalardan hesaplaniyordu, yani cikarilan parcanin
    kendi grubu "dokunulmamis" gorunuyordu. Ama o parca KULLANILDI -- olcum onbelleginde var.
    Bir kez dokunulan parca KALICI olarak kirlidir; aksi halde tek-atislik sinav kirlenir ve
    bunu FARK ETMEYIZ.

    Etkisi olculdu: "temiz LOCKED 98" -> DOGRUSU 95."""
    import olcum_kumesi
    import os
    if not os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "_der_tam.pkl")):
        import pytest
        pytest.skip("olcum onbellegi yok")
    _, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    atilan = set(rap["atilan_locked"])
    assert atilan, "bu testin anlamli olmasi icin en az bir LOCKED cikarilmali"
    temiz = set(rap["locked_temiz"])
    assert not (atilan & temiz), \
        f"DOGRUDAN kullanilan LOCKED parcalar TEMIZ sayiliyor: {sorted(atilan & temiz)}"
    assert rap["locked_temiz_n"] <= 100 - len(atilan)


def test_SINAV_egitim_maskesi_LOCKED_gruplarini_atar():
    """FINAL SINAVI icin egitim maskesi LOCKED'in GEOMETRI GRUPLARINI atmali.

    2026-08-03 BULGUSU: dagitilan egitim verisi (zengin_parite_w2.npz) 95 grup-temiz
    LOCKED parcasinin 77'sini iceriyordu (grup uzerinden 80). Sinav dagitilan gate'le
    kosulsaydi 95 parcanin 80'i KIRLI olurdu -> tek atislik hak bosa giderdi.
    Bu test, maskenin gercekten temizledigini garanti eder.
    """
    import numpy as np
    import olcum_kumesi
    npz = os.path.join(KOK, "results", "zengin_parite_w2.npz")
    if not os.path.exists(npz):
        pytest.skip("egitim npz yok")
    d = np.load(npz, allow_pickle=True)
    pids = np.array([str(x) for x in d["pids"]])
    m = olcum_kumesi.sinav_egitim_maskesi(pids)
    assert m.sum() > 0 and m.sum() < len(m), "maske ya hepsini atiyor ya hicbirini"
    _, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    gk = olcum_kumesi.geo_anahtarlari()
    kalan = {gk.get(x, "yok:" + x) for x in np.unique(pids[m])}
    ihlal = [x for x in rap["locked_temiz"] if gk.get(x, "yok:" + x) in kalan]
    assert not ihlal, f"maskeden sonra hala {len(ihlal)} LOCKED grubu egitimde: {ihlal[:5]}"


def test_f1w_TEK_REJIMLI_bolmede_agirligi_YENIDEN_NORMALLESTIRIR():
    """Tek rejimli bir bolmede f1w, eksik rejimi F1=0 sayip 0.895 ile CARPMAMALI.

    2026-08-03 denetiminde bit-duzeyinde yakalandi: ATANMAMIS 0.6635 raporlaniyordu,
    gercegi 0.7413 idi (rapor/0.895 == dusuk_CP_F1 TAM tutuyordu). Ayni sekilde
    seri 16 (0.8333 -> 0.9310) ve seri 17 (0.6531 -> 0.7297).
    HAVUZLANMIS ve uretici-disi bolmeler etkilenmemisti (ikisinde de her iki rejim var).
    """
    from sina_kume import f1w
    # yalniz dusuk-CP parcalar: TP=8 FP=2 FN=2 -> P=0.8 R=0.8 F1=0.8
    tek = [("dusuk", 8, 2, 2)]
    assert abs(f1w(tek) - 0.8) < 1e-6, f"tek rejimde F1 0.8 olmali, {f1w(tek)} geldi"
    # yalniz cok-CP parcalar da ayni sekilde
    tek_cok = [("cok", 8, 2, 2)]
    assert abs(f1w(tek_cok) - 0.8) < 1e-6, f"tek rejimde (cok) 0.8 olmali, {f1w(tek_cok)}"
    # IKI rejim varsa AGIRLIKLI ortalama: 0.895*0.8 + 0.105*0.5
    iki = [("dusuk", 8, 2, 2), ("cok", 5, 5, 5)]
    bek = 0.895 * 0.8 + 0.105 * 0.5
    assert abs(f1w(iki) - bek) < 1e-6, f"iki rejimde {bek} olmali, {f1w(iki)} geldi"
