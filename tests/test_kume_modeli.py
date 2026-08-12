# -*- coding: utf-8 -*-
"""S4 modelinin IDDIASI: parca baglamini goren model, NOKTASAL modelin
COZEMEYECEGI bir gorevi cozer.

Sentetik gorev: POZITIF, `v` degeri parcanin ORTALAMASINA EN YAKIN olan
secenektir.

ILK TASARIM KUSURLUYDU (2026-08-12): gorev "parcadaki EN BUYUK v" idi ve
noktasal model 0.775 aldi. Sebep: parca icinde ARGMAX aliniyor, dolayisiyla
v'nin ARTAN herhangi bir fonksiyonu gorevi zaten cozer -- gorev noktasal
olarak COZULEBILIR durumdaydi, yani hicbir sey olcmuyordu.

"Ortalamaya en yakin" olcutu monoton bir fonksiyonla COZULEMEZ: cevap parcanin
ortalamasina baglidir ve o ortalama parca basina rastgele kaydirilir. Noktasal
model bunu bilemez; DeepSets parca ozetini gordugu icin cozebilir.

Bu test gecmezse S4'un butun gerekcesi cokmus demektir.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kume_modeli as KM  # noqa: E402


def _veri(n_parca=300, tohum=0):
    rng = np.random.default_rng(tohum)
    P = []
    for _ in range(n_parca):
        n = int(rng.integers(10, 40))
        kaydirma = rng.uniform(-5, 5)          # parca basina RASTGELE kayma
        v = rng.normal(kaydirma, 1.0, size=n)
        gurultu = rng.normal(0, 1.0, size=(n, 3))
        X = np.column_stack([v, gurultu]).astype(np.float32)
        y = np.zeros(n, np.float32)
        # ORTALAMAYA EN YAKIN: monoton bir noktasal fonksiyonla cozulemez,
        # cunku cevap parcanin (rastgele kaydirilmis) ortalamasina baglidir.
        y[int(np.argmin(np.abs(v - v.mean())))] = 1.0
        P.append((X, y))
    return P


def _tepe_isabet(skorla, parcalar):
    """En yuksek skorlu secenek gercekten pozitif mi?"""
    d = [1.0 if int(np.argmax(skorla(X))) == int(np.argmax(y)) else 0.0
         for X, y in parcalar]
    return float(np.mean(d))


class _Noktasal(nn.Module):
    """Ayni kapasite, ama PARCA BAGLAMI YOK."""

    def __init__(self, n_giris, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_giris, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)


def _egit_noktasal(parcalar, n_giris, devir=60, tohum=0):
    torch.manual_seed(tohum)
    m = _Noktasal(n_giris)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(tohum)
    sira = np.arange(len(parcalar))
    for _ in range(devir):
        rng.shuffle(sira)
        for i in sira:
            X, y = parcalar[i]
            Xt = torch.as_tensor(X)
            yt = torch.as_tensor(y)
            L, _, _ = KM.kayip(m(Xt), yt, lam=0.5)
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
    return m


def test_kume_modeli_noktasali_belirgin_yener():
    egitim = _veri(400, tohum=0)
    sinav = _veri(120, tohum=99)
    n_giris = egitim[0][0].shape[1]

    # EGITIM BUTCESI: 8 devir YETERSIZDI (ikisi de sans seviyesinde kaliyordu).
    # Tarama: 30 devir/lr1e-3 -> 0.360, 30/lr3e-3 -> 0.440, 60/lr3e-3 -> 0.460.
    ks = KM.egit(egitim, n_giris, devir=60, lr=3e-3, lam=1.0,
                 cihaz="cpu", tohum=0)
    kume_isabet = _tepe_isabet(
        lambda X: KM.tahmin(ks, X, cihaz="cpu"), sinav)

    nk = _egit_noktasal(egitim, n_giris, devir=60, tohum=0)   # AYNI butce

    @torch.no_grad()
    def nok_skor(X):
        return nk(torch.as_tensor(np.asarray(X, np.float32))).numpy()

    nok_isabet = _tepe_isabet(nok_skor, sinav)

    print(f"    kume(DeepSets) {kume_isabet:.3f}  vs  noktasal {nok_isabet:.3f}")
    # Gorev noktasal model icin COZULEMEZ; kume modeli belirgin sekilde asmali.
    assert kume_isabet > 0.30, f"kume modeli gorevi cozemedi: {kume_isabet}"
    assert kume_isabet > nok_isabet + 0.15, (
        f"kume {kume_isabet:.3f} noktasali yeterince yenmedi {nok_isabet:.3f}")


def test_kayip_listwise_bileseni_calisir():
    logit = torch.tensor([2.0, -1.0, -1.0])
    y_dogru = torch.tensor([1.0, 0.0, 0.0])
    y_yanlis = torch.tensor([0.0, 0.0, 1.0])
    L_iyi = KM.kayip(logit, y_dogru)[0]
    L_kotu = KM.kayip(logit, y_yanlis)[0]
    assert float(L_iyi) < float(L_kotu)


def test_tek_secenekli_parca_cokmez():
    m = KM.KumeSkorlayici(4)
    X = torch.zeros(1, 4)
    y = torch.ones(1)
    L, _, _ = KM.kayip(m(X), y)
    assert torch.isfinite(L)


if __name__ == "__main__":
    for ad, f in sorted(globals().items()):
        if ad.startswith("test_"):
            f()
            print(f"  GECTI  {ad}")
    print("hepsi gecti")
