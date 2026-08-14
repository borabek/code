"""Tests for the leakage-safe train/val/test split and part-family grouping.

The split is the load-bearing honesty guarantee: if sibling part-family variants
straddle train and val, the reported F1 is inflated by memorised near-duplicates.
These assert that grouping keeps a family ten ONE side and that the three-way
split is a clean partition.
"""
import json_dataset as jd 
from train_cp import split_part_ids ,three_way_split 


def test_family_key_strips_variant_suffix ():
    assert jd .family_key ("ABC-1234-02")==jd .family_key ("ABC-1234-05")
    assert jd .family_key ("ABC-1234-02")=="ABC-1234"


def test_group_split_keeps_a_family_on_one_side ():
    ids =["P-100-%02d"%i for i in range (20 )]+["Q-200-%02d"%i for i in range (20 )]
    keys =jd .build_group_keys (
    [jd .Part (i ,[],[],[],[],[])for i in ids ],mode ="prefix")
    train ,val =split_part_ids (ids ,val_frac =0.5 ,seed =0 ,group_keys =keys )
    fam_p =[x for x in ids if x .startswith ("P-")]
    # every P-variant must be entirely in train OR entirely in val, never split
    assert all (x in train for x in fam_p )or all (x in val for x in fam_p )


def test_split_is_deterministic ():
    ids =["part-%03d"%i for i in range (50 )]
    assert split_part_ids (ids ,seed =0 )==split_part_ids (ids ,seed =0 )


def test_three_way_split_is_a_clean_partition ():
    ids =["part-%03d"%i for i in range (100 )]
    train ,val ,test =three_way_split (ids ,val_frac =0.2 ,test_frac =0.2 ,seed =1 )
    assert train .isdisjoint (val )and train .isdisjoint (test )and val .isdisjoint (test )
    assert len (train )+len (val )+len (test )==len (ids )
    assert len (val )>0 and len (test )>0 # both carved with these fracs


def test_test_frac_zero_reproduces_two_way ():
    ids =["part-%03d"%i for i in range (60 )]
    train2 ,val2 =split_part_ids (ids ,val_frac =0.2 ,seed =3 )
    train3 ,val3 ,test3 =three_way_split (ids ,val_frac =0.2 ,test_frac =0.0 ,seed =3 )
    assert test3 ==set ()and train3 ==train2 and val3 ==val2 


def test_family_key_does_not_over_merge_bare_catalog_numbers ():
# Phoenix Contact style: PartNr is "PXC.<digits>" with no letter after the
# digits -- these are DIFFERENT physical products, not SKU variants of one
# base part, and must not collapse into a single family (previously did:
# the whole catalogue number looked like a "trailing variant suffix").
    assert jd .family_key ("PXC.2277019")!=jd .family_key ("PXC.2308027")
    assert jd .family_key ("PXC.2277019")=="PXC.2277019"


def test_family_key_matches_accessory_suffix_variant ():
# A "+B056"-style accessory-option suffix must land ten the SAME family key
# as the otherwise-identical part without it (previously only the trailing
# digits of "+B056" were stripped, landing in a DIFFERENT group).
    base =jd .family_key ("ABB.ACS580-01-017A-4")
    with_accessory =jd .family_key ("ABB.ACS580-01-017A-4+B056")
    assert base ==with_accessory =="ABB.ACS580-01-017A"


def test_balance_search_fixes_skewed_grouped_split ():
# One big group (40% of the corpus) plus many singleton groups: a bare hash
# hitting the wrong seed can dump the whole big group into val, achieving a
# val fraction wildly off the requested one (measured ten the real corpus:
# 0.109-0.292 actual for a requested 0.2). balance_search must not do WORSE
# than the raw split, and for this known-skewed seed must do meaningfully
# BETTER (auto-picks a different seed with a closer-to-target val fraction).
    ids =["big-%02d"%i for i in range (40 )]+["s-%03d"%i for i in range (60 )]
    keys =["biggroup"]*40 +["s-%03d"%i for i in range (60 )]
    n =len (ids )
    val_frac =0.2 
    seed =2 # empirically: raw (balance_search=0) split for this seed is 0.5
    raw_train ,raw_val ,raw_test =three_way_split (
    ids ,val_frac =val_frac ,seed =seed ,group_keys =keys ,balance_search =0 )
    bal_train ,bal_val ,bal_test =three_way_split (
    ids ,val_frac =val_frac ,seed =seed ,group_keys =keys )
    raw_skew =abs (len (raw_val )/n -val_frac )
    bal_skew =abs (len (bal_val )/n -val_frac )
    assert raw_skew >0.2 # sanity: this seed really is skewed
    assert bal_skew <raw_skew # balance search must improve ten it
    assert bal_skew <=raw_skew # and must never make it worse
    # every group must still be intact (leakage guarantee not weakened)
    big_ids ={x for x in ids if x .startswith ("big-")}
    assert big_ids <=bal_train or big_ids <=bal_val or big_ids <=bal_test 


def test_balance_search_is_deterministic ():
    ids =["big-%02d"%i for i in range (40 )]+["s-%03d"%i for i in range (60 )]
    keys =["biggroup"]*40 +["s-%03d"%i for i in range (60 )]
    a =three_way_split (ids ,val_frac =0.2 ,seed =2 ,group_keys =keys )
    b =three_way_split (ids ,val_frac =0.2 ,seed =2 ,group_keys =keys )
    assert a ==b 
