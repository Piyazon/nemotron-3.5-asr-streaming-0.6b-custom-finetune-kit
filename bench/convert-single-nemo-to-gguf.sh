#!/bin/bash
set -e

###############################################################################
# convert-single-nemo-to-gguf.sh
#
# Converts a single .nemo file to GGUF (F16 or quantized).
#
# Usage:  bash convert-single-nemo-to-gguf.sh <path-to-file.nemo> [q4_k|q8_0]
###############################################################################

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-file.nemo> [q4_k|q8_0]"
    exit 1
fi

NEMO_FILE="$1"
QUANT="${2:-}"

if [[ "$NEMO_FILE" != /* ]]; then
    NEMO_FILE="$(cd "$(dirname "$NEMO_FILE")" && pwd)/$(basename "$NEMO_FILE")"
fi

if [[ ! -f "$NEMO_FILE" ]]; then
    echo "Error: file not found: $NEMO_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-$KIT_ROOT/../CrispASR/models/convert-nemotron-to-gguf.py}"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "Error: conversion script not found: $CONVERT_SCRIPT"
    exit 1
fi

NEMO_DIR="$(dirname "$NEMO_FILE")"
BASENAME="$(basename "${NEMO_FILE%.nemo}")"

if [[ -n "$QUANT" ]]; then
    OUT_FILE="${NEMO_DIR}/${BASENAME}-${QUANT}.gguf"
else
    OUT_FILE="${NEMO_DIR}/${BASENAME}.gguf"
fi

echo "Input:    $NEMO_FILE"
echo "Output:   $OUT_FILE"
echo "Quant:    ${QUANT:-F16 (none)}"
echo ""

CMD="python3 \"$CONVERT_SCRIPT\" --nemo \"$NEMO_FILE\" --output \"$OUT_FILE\""
if [[ -n "$QUANT" ]]; then
    CMD="$CMD --quant $QUANT"
fi

echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "Done. Output file:"
ls -lh "$OUT_FILE"
