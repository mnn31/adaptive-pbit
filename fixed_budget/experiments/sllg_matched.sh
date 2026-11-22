#!/usr/bin/env bash
# Author: Manan Gupta <mnn@yogins.com>
# Build the matched-macrospin record set from a mumax3 summary, start to end.
#
#   bash experiments/sllg_matched.sh OUT [GPU] [MUMAX_SUMMARY]
#
# Run inside fixed_budget/. OUT must not hold records yet: every part skips
# files that exist, so a set built on an older summary has to be moved aside
# first. Rerunning after an interruption continues where it stopped (pass
# RESUME=1). The macrospin-trained networks are trained before any run that
# tests them, and the last step fails if a cell of the table is missing.
# On one RTX 4090: the match takes two minutes of CPU, the device parts
# about 45 minutes, the six training runs about 15 minutes and the macrospin
# classifier runs about 85 minutes; the telegraph columns run on CPU.
set -euo pipefail

OUT=${1:?output directory, e.g. results/sllg_matched}
GPU=${2:-cuda:1}
MUMAX=${3:-results/micromagnetics/summary.json}

if [ -e "$OUT/match.json" ] && [ "${RESUME:-0}" != 1 ]; then
    echo "$OUT already holds a record set; move it aside or pass RESUME=1" >&2
    exit 1
fi

D="python experiments/sllg_device.py --out $OUT --gpu $GPU"
C="python experiments/sllg_classifier.py --dev-dir $OUT --gpu $GPU"

# device: barrier fit, then its characterization
[ -e "$OUT/match.json" ] || $D --part match --mumax "$MUMAX"
[ -e "$OUT/device_dt.json" ] || $D --part dt
$D --part bias
[ -e "$OUT/device_step.json" ] || $D --part step
$D --part summary

# networks trained on this device's macrospin (tested by the transfer group)
python experiments/sllg_train.py --dev-dir "$OUT" --r0 0.2 0.5 --seed 0 1 2 \
    --gpu "$GPU"

# telegraph-model columns (CPU); the shared ones are skipped if on record
$C --part telegraph --groups settle stream crossing cap transfer frontier
$C --part telegraph --groups stream crossing --seeds 0 1 2 3 4 5
$C --part tcap --groups cap stream crossing
$C --part tlaw --groups cap stream crossing transfer frontier
$C --part ideal
$C --part equilibrium
$C --part inputs

# macrospin neurons
$C --part sllg --groups settle stream crossing cap frontier transfer
$C --part dtcheck
$C --part table --strict
