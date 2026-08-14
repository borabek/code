# -*- coding: utf-8 -*-
"""cp_geometry primitifleri -- ANALITIK as known sekillerde sabitlenir.

Neden: 2026-07-28'de this fonksiyonlar rtree yoklugunda sessizce girdilerini geri donduruyordu and
no sey difference etmedi. Burada beklenen degerler ELDE hesaplanabilir olduğu for, fonksiyon
calismayi birakirsa test ANINDA kirmizi becomes. Denetcinin (glb_audit) guvendigi baseline budur.
"""
import numpy as np 
import pytest 
import trimesh 

from cp_geometry import ray_hits ,is_inside ,seat_to_mouth ,outward_along_axis ,exit_length 


@pytest .fixture 
def box ():
    """Merkezde 10x20x30 kutu: yuzeyler x=+-5, y=+-10, z=+-15."""
    return trimesh .creation .box (extents =[10.0 ,20.0 ,30.0 ])


    # ------------------------------------------------------------------ ray_hits
def test_ray_hits_from_center (box ):
    """Merkezden +Z: full as BIR kesisim, 15.0 mm'de."""
    h =ray_hits (box ,[0 ,0 ,0 ],[0 ,0 ,1 ])
    assert len (h )==1 
    assert h [0 ]==pytest .approx (15.0 ,abs =1e-6 )


def test_ray_hits_through_from_outside (box ):
    """Disaridan passing isin: IKI kesisim (giris + cikis), 35 and 65 mm."""
    h =ray_hits (box ,[0 ,0 ,-50 ],[0 ,0 ,1 ])
    assert len (h )==2 
    assert h [0 ]==pytest .approx (35.0 ,abs =1e-6 )
    assert h [1 ]==pytest .approx (65.0 ,abs =1e-6 )


def test_ray_hits_miss (box ):
    """Kutuyu isinlamayan isin: SIFIR kesisim."""
    assert len (ray_hits (box ,[100 ,100 ,100 ],[0 ,0 ,1 ]))==0 


def test_ray_hits_respects_max (box ):
    """max_mm sinirinin otesindeki kesisim sayilmaz."""
    assert len (ray_hits (box ,[0 ,0 ,0 ],[0 ,0 ,1 ],max_mm =10.0 ))==0 
    assert len (ray_hits (box ,[0 ,0 ,0 ],[0 ,0 ,1 ],max_mm =20.0 ))==1 


def test_ray_hits_normalises_direction (box ):
    """Yon vektoru unit olmasa da distance MM cinsinden correct kalir."""
    h =ray_hits (box ,[0 ,0 ,0 ],[0 ,0 ,7.3 ])
    assert h [0 ]==pytest .approx (15.0 ,abs =1e-6 )


    # ------------------------------------------------------------------ is_inside
def test_is_inside (box ):
    assert is_inside (box ,[0 ,0 ,0 ])is True 
    assert is_inside (box ,[0 ,0 ,14.9 ])is True 


def test_is_outside (box ):
    assert is_inside (box ,[0 ,0 ,15.1 ])is False 
    assert is_inside (box ,[0 ,0 ,1000 ])is False 
    assert is_inside (box ,[50 ,0 ,0 ])is False 


    # ------------------------------------------------------------------ seat_to_mouth
def test_seat_to_mouth_from_center (box ):
    """Merkezden +Z ekseni: two direction de 15mm; mouth yuzeyde, offset buyuklugu 15."""
    mouth ,off =seat_to_mouth (box ,[0 ,0 ,0 ],[0 ,0 ,1 ])
    assert abs (off )==pytest .approx (15.0 ,abs =1e-6 )
    assert abs (abs (mouth [2 ])-15.0 )<1e-6 


def test_seat_to_mouth_picks_nearest_exit (box ):
    """z=+5'te: +Z'de 10mm, -Z'de 20mm -> YAKIN which is secilir."""
    mouth ,off =seat_to_mouth (box ,[0 ,0 ,5.0 ],[0 ,0 ,1 ])
    assert off ==pytest .approx (+10.0 ,abs =1e-6 )
    assert mouth [2 ]==pytest .approx (15.0 ,abs =1e-6 )


def test_seat_to_mouth_leaves_outside_point_alone (box ):
    """DISARIDAKI point OYNATILMAZ (otherwise isin karsi duvara carpip noktayi firlatir)."""
    p =[0 ,0 ,100.0 ]
    mouth ,off =seat_to_mouth (box ,p ,[0 ,0 ,1 ])
    assert off ==0.0 
    assert np .allclose (mouth ,p )


def test_seat_to_mouth_zero_direction_is_noop (box ):
    mouth ,off =seat_to_mouth (box ,[0 ,0 ,0 ],[0 ,0 ,0 ])
    assert off ==0.0 and np .allclose (mouth ,[0 ,0 ,0 ])


    # ------------------------------------------------------------------ outward_along_axis
def test_outward_from_surface_point_points_away (box ):
    """Yuzeyin hemen disindaki point: empty tarafa (+Z) bakmali."""
    d =outward_along_axis (box ,[0 ,0 ,15.5 ],[0 ,0 ,1 ])
    assert d [2 ]==pytest .approx (+1.0 ,abs =1e-9 )


def test_outward_from_surface_point_flips_sign_when_needed (box ):
    """Karsi yuzeyde same axis: this times -Z disari must be (sign KENDILIGINDEN returns)."""
    d =outward_along_axis (box ,[0 ,0 ,-15.5 ],[0 ,0 ,1 ])
    assert d [2 ]==pytest .approx (-1.0 ,abs =1e-9 )


def test_outward_inside_point_takes_nearest_exit (box ):
    """ICERIDEKI point: kisa yoldan cikilan taraf disaridir (z=+14 -> +Z, 1mm)."""
    d =outward_along_axis (box ,[0 ,0 ,14.0 ],[0 ,0 ,1 ])
    assert d [2 ]==pytest .approx (+1.0 ,abs =1e-9 )


def test_outward_free_space_prefers_more_room (box ):
    """Serbest alandaki point: onunde more COK bosluk which is taraf disaridir."""
    d =outward_along_axis (box ,[0 ,0 ,40.0 ],[0 ,0 ,1 ])
    assert d [2 ]==pytest .approx (+1.0 ,abs =1e-9 )# +Z: never malzeme absent; -Z: kutu present


    # ------------------------------------------------------------------ exit_length
def test_exit_length_clears_material (box ):
    """Merkezden +Z: 15mm malzeme + 2mm pay = 17mm."""
    L =exit_length (box ,[0 ,0 ,0 ],[0 ,0 ,1 ],margin =2.0 )
    assert L ==pytest .approx (17.0 ,abs =1e-6 )


def test_exit_length_outside_is_just_margin (box ):
    """Bos tarafa bakan disarisi point: only pay up to."""
    L =exit_length (box ,[0 ,0 ,20.0 ],[0 ,0 ,1 ],margin =3.0 )
    assert L ==pytest .approx (3.0 ,abs =1e-6 )


def test_exit_length_respects_min_and_max (box ):
    assert exit_length (box ,[0 ,0 ,20.0 ],[0 ,0 ,1 ],margin =1.0 ,min_len =9.0 )==pytest .approx (9.0 )
    assert exit_length (box ,[0 ,0 ,0 ],[0 ,0 ,1 ],margin =2.0 ,max_len =12.0 )==pytest .approx (12.0 )


def test_exit_length_tip_is_actually_outside (box ):
    """SOZLESME M3'un tabani: measured_path boyla three GERCEKTEN disarida must be."""
    for p in ([0 ,0 ,0 ],[0 ,0 ,14.0 ],[3 ,5 ,-10.0 ]):
        d =outward_along_axis (box ,p ,[0 ,0 ,1 ])
        L =exit_length (box ,p ,d ,margin =2.0 )
        assert not is_inside (box ,np .asarray (p ,float )+d *L )


        # ------------------------------------------------------------------ delikli body (tup)
@pytest .fixture 
def tube ():
    """Ic radius 3, dis 8, height 20 tup: axis on MALZEME YOK (hole)."""
    return trimesh .creation .annulus (r_min =3.0 ,r_max =8.0 ,height =20.0 )


def test_tube_axis_is_hollow (tube ):
    """Delik ekseninde isin no surface kesmez -- 'opening' tanimi budur."""
    assert len (ray_hits (tube ,[0 ,0 ,0 ],[0 ,0 ,1 ]))==0 
    assert is_inside (tube ,[0 ,0 ,0 ])is False 


def test_tube_wall_is_solid (tube ):
    """Duvarin icindeki point ICERIDEDIR (r=5.5, ic 3 with dis 8 arasi)."""
    assert is_inside (tube ,[5.5 ,0 ,0 ])is True 


def test_tube_marker_at_bore_mouth_points_out (tube ):
    """Delik agzindaki isaretci axis along DISARI bakar and ucu govdeye girmez."""
    p =np .array ([0.0 ,0.0 ,10.5 ])# upper agzin hemen ustu
    d =outward_along_axis (tube ,p ,[0 ,0 ,1 ])
    assert d [2 ]==pytest .approx (+1.0 ,abs =1e-9 )
    L =exit_length (tube ,p ,d ,margin =2.0 )
    assert not is_inside (tube ,p +d *L )
