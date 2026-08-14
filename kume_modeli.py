# -*- coding: utf-8 -*-
"""S4: ADAY-KUMESI SKORLAYICI -- secenegi parcanin KENDI POPULASYONUNA according to puanla.

WHY (S7 teshisi, 2026-08-12). Secici verimliligi two kutuplu: UPUN %65.5,
NIT %0.5. Sebep POZITIF-NEGATIF SKOR AYRIMI: NIT'te correct secenek yanlistan
only 0.05 more high score aliyor. Havuz cevabi tasiyor (ceiling 0.9147),
skor gostermiyor.

Bugunku model NOKTASAL: each secenegi TEK BASINA puanliyor, "same parcadaki
digerlerine according to nasil" sorusunu goremiyor. Karar kurali whereas GORELI
(part-maksimumunun %85'i). Bu bosluk yapisaldir; feature eklemekle kapanmaz
-- nitekim `ozkalib` blogu (skoru part inside yeniden olceklendirme) -0.0482
with COKTU: skoru SONRADAN normallestirmek islemiyor, model KARARI VERIRKEN
baglami gormeli.

WHY DEEPSETS, NOTE (ATTENTION) DEGIL. Parca basina 7800'e up to secenek
present; full dikkat O(n^2) = 60M double and 4 GB VRAM'e sigmaz. DeepSets O(n) and
ihtiyacimiz which is seyi full as gives: each secenek, parcanin ORTALAMA and
EN YUKSEK temsiliyle birlikte puanlanir. "Bu secenek, this parcadaki digerlerine
according to iyi mi?" sorusu budur.

KAYIP: BCE + LISTWISE. Yalniz BCE mutlak kalibrasyon ogretir but siralamayi
zorlamaz; only listwise siralamayi ogretir but threshold kurali for gereken
mutlak duzeyi breaks. Ikisi birlikte (lambda with) is used.
"""
import numpy as np 
import torch 
import torch .nn as nn 
import torch .nn .functional as F 


class KumeSkorlayici (nn .Module ):
    """DeepSets: phi(secenek) -> part ozeti -> rho(secenek, ozet) -> logit."""

    def __init__ (self ,n_giris ,d =128 ,p_drop =0.1 ):
        super ().__init__ ()
        self .phi =nn .Sequential (
        nn .Linear (n_giris ,d ),nn .LayerNorm (d ),nn .GELU (),
        nn .Dropout (p_drop ),
        nn .Linear (d ,d ),nn .LayerNorm (d ),nn .GELU (),
        )
        # rho girisi: [own, part ortalamasi, part maksimumu, own - mean]
        self .rho =nn .Sequential (
        nn .Linear (4 *d ,d ),nn .LayerNorm (d ),nn .GELU (),
        nn .Dropout (p_drop ),
        nn .Linear (d ,d //2 ),nn .GELU (),
        nn .Linear (d //2 ,1 ),
        )

    def forward (self ,X ):
        """X: (n_secenek, n_giris) TEK part. Doner: (n_secenek,) logit."""
        h =self .phi (X )
        ort =h .mean (0 ,keepdim =True ).expand_as (h )
        mak =h .max (0 ,keepdim =True ).values .expand_as (h )
        return self .rho (torch .cat ([h ,ort ,mak ,h -ort ],1 )).squeeze (-1 )


def loss (logit ,y ,lam =0.5 ,bce_maske =None ):
    """BCE + LISTWISE.

    Listwise: part inside pozitiflerin log-softmax'i. Cok pozitifli parts
    for pozitiflerin ORTALAMASI alinir; so 24 CP'li NIT parcasi with 3
    CP'li UPUN parcasi same agirligi carries.

    `bce_maske`: BCE'nin HANGI seceneklerden hesaplanacagi.

    WHY IT EXISTS (2026-08-12, S4 v1 olcumunden after). HGB kolu pozitif basina 6
    negatifle DENGELENMIS a orneklemde egitiliyor; DeepSets whereas parcanin
    tamamini goruyordu (~3000 secenek, ~10 pozitif = 300:1). BCE mean
    oldugu for negatifler kaybi boguyor and model each seye ~0 demeyi
    ogreniyor. Iki arm EGITIM DENGESI bakimindan equal degildi -- haksiz kiyas.

    KRITIK AYRIM: maske only KAYBI daraltir. Ileri gecis (and therefore
    part ozeti/baglam) HER ZAMAN TUM part ten is computed; aksi halde
    modelin varlik sebebi which is baglam absent olurdu. LISTWISE de tum part
    ten kalir -- ranking however full cluster ten anlamlidir.
    """
    if bce_maske is None :
        bce =F .binary_cross_entropy_with_logits (logit ,y )
    else :
        bce =F .binary_cross_entropy_with_logits (logit [bce_maske ],
        y [bce_maske ])
    if y .sum ()>0 and len (y )>1 :
        ls =-F .log_softmax (logit ,dim =0 )[y >0 ].mean ()
    else :
        ls =torch .zeros ((),device =logit .device )
    return bce +lam *ls ,bce .detach (),ls .detach ()


def egit (parts ,n_giris ,d =128 ,devir =8 ,lr =1e-3 ,lam =0.5 ,
cihaz =None ,seed =0 ,ilerle =None ,neg_kat =0 ):
    """`parts`: [(X_np, y_np), ...] each biri BIR part. Doner: model.

    `neg_kat > 0` whereas BCE each devirde pozitif basina `neg_kat` negatiften
    is computed (HGB kolunun dengesiyle ESITLENIR). Negatifler each devirde
    YENIDEN cekilir, so zamanla all of them gorulur. Ileri gecis and listwise
    HER ZAMAN tum part uzerindedir -- see. `loss` aciklamasi.
    """
    cihaz =cihaz or ("cuda"if torch .cuda .is_available ()else "cpu")
    torch .manual_seed (seed )
    m =KumeSkorlayici (n_giris ,d ).to (cihaz )
    opt =torch .optim .AdamW (m .parameters (),lr =lr ,weight_decay =1e-4 )
    rng =np .random .default_rng (seed )
    rank_ =np .arange (len (parts ))
    for e in range (devir ):
        rng .shuffle (rank_ )
        m .train ()
        top =n =0.0 
        for i in rank_ :
            X ,y =parts [i ]
            if len (X )<2 or not y .any ():
                continue 
            Xt =torch .as_tensor (X ,dtype =torch .float32 ,device =cihaz )
            yt =torch .as_tensor (y ,dtype =torch .float32 ,device =cihaz )
            mask =None 
            if neg_kat >0 :
                p_ =np .where (y >0 )[0 ]
                n_ =np .where (y ==0 )[0 ]
                if len (n_ )>neg_kat *len (p_ ):
                    n_ =rng .choice (n_ ,neg_kat *len (p_ ),replace =False )
                mask =torch .zeros (len (y ),dtype =torch .bool ,device =cihaz )
                mask [torch .as_tensor (np .concatenate ([p_ ,n_ ]),
                device =cihaz )]=True 
            L ,_ ,_ =loss (m (Xt ),yt ,lam ,bce_maske =mask )
            opt .zero_grad (set_to_none =True )
            L .backward ()
            torch .nn .utils .clip_grad_norm_ (m .parameters (),1.0 )
            opt .step ()
            top +=float (L );n +=1 
        if ilerle :
            ilerle (e ,top /max (n ,1 ))
    return m 


@torch .no_grad ()
def pred_ (m ,X ,cihaz =None ):
    """Tek parcanin secenek olasiliklari."""
    cihaz =cihaz or next (m .parameters ()).device 
    m .eval ()
    Xt =torch .as_tensor (np .asarray (X ,np .float32 ),device =cihaz )
    return torch .sigmoid (m (Xt )).cpu ().numpy ()
