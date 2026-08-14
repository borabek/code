#!/usr/bin/env bash
# Y19 — GIRDI OZNITELIGI: xyz (dissal) vs hks (icsel)
#
# BULGU. `train_seg_extra.py` cfg'yi SABIT `input_features="xyz"` yaziyordu;
# `hks` secenegi DiffusionNet'te destekli oldugu halde hic denenemiyordu.
# Canli kontrol noktalari da xyz.
#
# NEDEN ONEMLI. `xyz` DISSALDIR -- model MUTLAK KONUMA baglanir. `hks`
# (isi cekirdegi imzasi, 16 kanal) ICSELDIR: donme ve otelemeye duyarsiz,
# yalnizca yuzeyin kendi geometrisini tasir. Gorulmemis brand kosulunda
# istenen tam olarak budur.
#
# TEK DEGISKEN: input_features. Etiket dizinleri, seed, k-eig, epoch AYNI.
# Taban = A kolu receteси (189 part, 200 epoch, val Conn_IoU 0.6232).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --input-features hks \
  --checkpoint-out results/seg_extra/y19_hks_s0.pt
