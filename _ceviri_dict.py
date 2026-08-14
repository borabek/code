# -*- coding: utf-8 -*-
"""TR -> EN sozlugu. YALNIZ ayirt edici, alana ozgu terimler.

Kural: single harfli / kisa / ambiguous names (f, oku, yap, sec, kur, sha, rec)
DISARIDA -- kelime siniriyla bile carpisma riski present.
Uzun which is ONCE gelmeli (match_hungarian, match_greedy'dan before).
"""
SOZLUK ={
# --- cekirdek fonksiyonlar (very cagrilan) ---
"match_hungarian":"match_hungarian","match_greedy":"match_greedy",
"match_signed":"match_signed","match3":"match3",
"decision_score":"decision_score","decision_mask":"decision_mask",
"derive_candidates":"derive_candidates","product_output":"product_output",
"canonical_chain":"canonical_chain","canonical_block":"canonical_block",
"ring_normals":"ring_normals","ring_sign":"ring_sign",
"suppress_crowd":"suppress_crowd","crowd_mask":"crowd_mask",
"pose_correct":"pose_correct","angle_correct":"angle_correct",
"pick_member_direction":"pick_member_direction",
"pick_direction_from_dictionary":"pick_direction_from_dictionary",
"candidates":"candidates","vertex_normals_at":"vertex_normals_at",
"step_map":"step_map","within_part":"within_part",
"assign_tier":"assign_tier","post_gate_chain":"post_gate_chain",
"local_frame":"local_frame","merged_pool":"merged_pool",
"thin_pool":"thin_pool","candidate_label":"candidate_label",
"project_label":"project_label","axis_sample":"axis_sample",
"warn_position":"warn_position","pick_gate":"pick_gate",
"estimate":"estimate","self_check":"self_check",
# --- sik gecen isimler ---
"measurement":"measurement","measured":"measured","criterion":"criterion",
"part":"part","parts":"parts","candidate":"candidate",
"pool":"pool","direction":"direction","axis":"axis",
"sign":"sign","signed":"signed","unsigned":"unsigned",
"lateral":"lateral","axial":"axial","depth":"depth",
"mouth":"mouth","opening":"opening","body":"body",
"cluster":"cluster","clustering":"clustering","seed":"seed",
"fold":"fold","split":"split","exam":"exam","metadata":"metadata",
"receipt":"receipt","rationale":"rationale","verdict":"verdict",
"evidence":"evidence","ceiling":"ceiling","baseline":"baseline",
"gate":"gate","threshold":"threshold","ratio":"ratio",
"confidence":"confidence","precision":"precision","noise":"noise",
"deviation":"deviation","error":"error","loss":"loss",
"training":"training","inference":"inference","ensemble":"ensemble",
"router":"router","selector":"selector","guard":"guard",
"leakage":"leakage","cache":"cache","backup":"backup",
"version":"version","deployed":"deployed","closed":"closed",
"smoke_test":"smoke_test","rollback":"rollback",
"headline":"headline","probe":"probe","arm":"arm",
"dense":"dense","sparse":"sparse","regime":"regime",
"manufacturer":"manufacturer","brand":"brand","corpus":"corpus",
}
