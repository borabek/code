# -*- coding: utf-8 -*-
"""canonical_alignment: cercevenin donme/oteleme/scale under DEGISMEDIGI.

Modulun tum varlik sebebi this: same klemens baska a eksende modellenmisse
features AYNI cikmali. Test bunu dogrudan olcer.
"""
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

import canonical_alignment as KH # noqa: E402


def _donme (seed =0 ):
    rng =np .random .default_rng (seed )
    A =rng .normal (size =(3 ,3 ))
    Q ,R =np .linalg .qr (A )
    Q =Q *np .sign (np .diag (R ))
    if np .linalg .det (Q )<0 :
        Q [:,0 ]=-Q [:,0 ]
    return Q 


def _sahne (seed =1 ):
    """ASIMETRIK body (carpiklik ~0 olmasin) + candidates."""
    rng =np .random .default_rng (seed )
    V =rng .normal (size =(400 ,3 ))*np .array ([20.0 ,6.0 ,3.0 ])
    V =np .vstack ([V ,rng .normal (size =(120 ,3 ))*3.0 +np .array ([35.0 ,0 ,0 ])])
    P =rng .normal (size =(25 ,3 ))*np .array ([15.0 ,4.0 ,2.0 ])
    D =rng .normal (size =(25 ,3 ))
    return V ,P ,D 


def test_donme_degismezligi ():
    V ,P ,D =_sahne ()
    X0 =KH .oznitelik (P ,D ,V )
    for t in (0 ,1 ,2 ):
        R =_donme (t )
        X1 =KH .oznitelik (P @R .T ,D @R .T ,V @R .T )
        f =float (np .abs (X0 -X1 ).max ())
        assert f <1e-6 ,f"donme {t }: maks diff {f }"


def test_oteleme_degismezligi ():
    V ,P ,D =_sahne ()
    X0 =KH .oznitelik (P ,D ,V )
    o =np .array ([123.0 ,-45.0 ,7.0 ])
    X1 =KH .oznitelik (P +o ,D ,V +o )
    assert np .abs (X0 -X1 ).max ()<1e-6 


def test_olcek_degismezligi ():
    V ,P ,D =_sahne ()
    X0 =KH .oznitelik (P ,D ,V )
    X1 =KH .oznitelik (P *3.7 ,D ,V *3.7 )
    assert np .abs (X0 -X1 ).max ()<1e-6 


def test_mutlak_sutunlar_isaretten_bagimsiz ():
    V ,P ,D =_sahne ()
    X0 =KH .oznitelik (P ,D ,V )
    X1 =KH .oznitelik (P ,-D ,V )
    # signed sutunlar tersine returns, MUTLAK sutunlar same kalir
    assert np .abs (X0 [:,3 :6 ]+X1 [:,3 :6 ]).max ()<1e-9 
    assert np .abs (X0 [:,6 :9 ]-X1 [:,6 :9 ]).max ()<1e-9 


def test_dejenere_ve_bos_cokmez ():
    assert KH .oznitelik (np .zeros ((0 ,3 )),np .zeros ((0 ,3 )),
    np .zeros ((0 ,3 ))).shape ==(0 ,len (KH .OZ_AD ))
    X =KH .oznitelik (np .zeros ((2 ,3 )),np .tile ([0 ,0 ,1.0 ],(2 ,1 )),
    np .zeros ((1 ,3 )))
    assert X .shape ==(2 ,len (KH .OZ_AD ))and np .isfinite (X ).all ()


if __name__ =="__main__":
    for ad ,f in sorted (globals ().items ()):
        if ad .startswith ("test_"):
            f ()
            print (f"  GECTI  {ad }")
    print ("all of them gecti")
