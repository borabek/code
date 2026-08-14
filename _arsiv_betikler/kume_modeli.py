# -*- coding: utf-8 -*-
"""S4: ADAY-KUMESI SKORLAYICI -- secenegi parcanin KENDI POPULASYONUNA gore puanla.

NEDEN (S7 teshisi, 2026-08-12). Secici verimliligi iki kutuplu: UPUN %65.5,
NIT %0.5. Sebep POZITIF-NEGATIF SKOR AYRIMI: NIT'te dogru secenek yanlistan
yalnizca 0.05 daha yuksek puan aliyor. Havuz cevabi tasiyor (tavan 0.9147),
skor gostermiyor.

Bugunku model NOKTASAL: her secenegi TEK BASINA puanliyor, "ayni parcadaki
digerlerine gore nasil" sorusunu goremiyor. Karar kurali ise GORELI
(parca-maksimumunun %85'i). Bu bosluk yapisaldir; oznitelik eklemekle kapanmaz
-- nitekim `ozkalib` blogu (skoru parca icinde yeniden olceklendirme) -0.0482
ile COKTU: skoru SONRADAN normallestirmek islemiyor, model KARARI VERIRKEN
baglami gormeli.

NEDEN DEEPSETS, DIKKAT (ATTENTION) DEGIL. Parca basina 7800'e kadar secenek
var; tam dikkat O(n^2) = 60M cift ve 4 GB VRAM'e sigmaz. DeepSets O(n) ve
ihtiyacimiz olan seyi tam olarak verir: her secenek, parcanin ORTALAMA ve
EN YUKSEK temsiliyle birlikte puanlanir. "Bu secenek, bu parcadaki digerlerine
gore iyi mi?" sorusu budur.

KAYIP: BCE + LISTWISE. Yalniz BCE mutlak kalibrasyon ogretir ama siralamayi
zorlamaz; yalniz listwise siralamayi ogretir ama esik kurali icin gereken
mutlak duzeyi bozar. Ikisi birlikte (lambda ile) kullanilir.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KumeSkorlayici(nn.Module):
    """DeepSets: phi(secenek) -> parca ozeti -> rho(secenek, ozet) -> logit."""

    def __init__(self, n_giris, d=128, p_drop=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_giris, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
        )
        # rho girisi: [kendi, parca ortalamasi, parca maksimumu, kendi - ortalama]
        self.rho = nn.Sequential(
            nn.Linear(4 * d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, X):
        """X: (n_secenek, n_giris) TEK parca. Doner: (n_secenek,) logit."""
        h = self.phi(X)
        ort = h.mean(0, keepdim=True).expand_as(h)
        mak = h.max(0, keepdim=True).values.expand_as(h)
        return self.rho(torch.cat([h, ort, mak, h - ort], 1)).squeeze(-1)


def kayip(logit, y, lam=0.5, bce_maske=None):
    """BCE + LISTWISE.

    Listwise: parca icinde pozitiflerin log-softmax'i. Cok pozitifli parcalar
    icin pozitiflerin ORTALAMASI alinir; boylece 24 CP'li NIT parcasi ile 3
    CP'li UPUN parcasi ayni agirligi tasir.

    `bce_maske`: BCE'nin HANGI seceneklerden hesaplanacagi.

    NEDEN VAR (2026-08-12, S4 v1 olcumunden sonra). HGB kolu pozitif basina 6
    negatifle DENGELENMIS bir orneklemde egitiliyor; DeepSets ise parcanin
    tamamini goruyordu (~3000 secenek, ~10 pozitif = 300:1). BCE ortalama
    oldugu icin negatifler kaybi boguyor ve model her seye ~0 demeyi
    ogreniyor. Iki kol EGITIM DENGESI bakimindan esit degildi -- haksiz kiyas.

    KRITIK AYRIM: maske yalnizca KAYBI daraltir. Ileri gecis (ve dolayisiyla
    parca ozeti/baglam) HER ZAMAN TUM parca uzerinde hesaplanir; aksi halde
    modelin varlik sebebi olan baglam yok olurdu. LISTWISE de tum parca
    uzerinde kalir -- siralama ancak tam kume uzerinde anlamlidir.
    """
    if bce_maske is None:
        bce = F.binary_cross_entropy_with_logits(logit, y)
    else:
        bce = F.binary_cross_entropy_with_logits(logit[bce_maske],
                                                 y[bce_maske])
    if y.sum() > 0 and len(y) > 1:
        ls = -F.log_softmax(logit, dim=0)[y > 0].mean()
    else:
        ls = torch.zeros((), device=logit.device)
    return bce + lam * ls, bce.detach(), ls.detach()


def egit(parcalar, n_giris, d=128, devir=8, lr=1e-3, lam=0.5,
         cihaz=None, tohum=0, ilerle=None, neg_kat=0):
    """`parcalar`: [(X_np, y_np), ...] her biri BIR parca. Doner: model.

    `neg_kat > 0` ise BCE her devirde pozitif basina `neg_kat` negatiften
    hesaplanir (HGB kolunun dengesiyle ESITLENIR). Negatifler her devirde
    YENIDEN cekilir, boylece zamanla hepsi gorulur. Ileri gecis ve listwise
    HER ZAMAN tum parca uzerindedir -- bkz. `kayip` aciklamasi.
    """
    cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tohum)
    m = KumeSkorlayici(n_giris, d).to(cihaz)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(tohum)
    sira = np.arange(len(parcalar))
    for e in range(devir):
        rng.shuffle(sira)
        m.train()
        top = n = 0.0
        for i in sira:
            X, y = parcalar[i]
            if len(X) < 2 or not y.any():
                continue
            Xt = torch.as_tensor(X, dtype=torch.float32, device=cihaz)
            yt = torch.as_tensor(y, dtype=torch.float32, device=cihaz)
            mask = None
            if neg_kat > 0:
                p_ = np.where(y > 0)[0]
                n_ = np.where(y == 0)[0]
                if len(n_) > neg_kat * len(p_):
                    n_ = rng.choice(n_, neg_kat * len(p_), replace=False)
                mask = torch.zeros(len(y), dtype=torch.bool, device=cihaz)
                mask[torch.as_tensor(np.concatenate([p_, n_]),
                                     device=cihaz)] = True
            L, _, _ = kayip(m(Xt), yt, lam, bce_maske=mask)
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            top += float(L); n += 1
        if ilerle:
            ilerle(e, top / max(n, 1))
    return m


@torch.no_grad()
def tahmin(m, X, cihaz=None):
    """Tek parcanin secenek olasiliklari."""
    cihaz = cihaz or next(m.parameters()).device
    m.eval()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=cihaz)
    return torch.sigmoid(m(Xt)).cpu().numpy()
