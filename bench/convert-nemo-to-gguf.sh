#!/bin/bash
set -e

###############################################################################
# convert-nemo-to-gguf.sh
#
# Converts .nemo checkpoint(s) to GGUF (F16 or quantized).
#
# Usage:
#   bash convert-nemo-to-gguf.sh <path-to-file.nemo> [q4_k|q8_0]     # single file
#   bash convert-nemo-to-gguf.sh --all [q4_k|q8_0]                   # all best-N checkpoints
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$KIT_ROOT/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/test"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-$KIT_ROOT/../CrispASR/models/convert-nemotron-to-gguf.py}"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "Error: conversion script not found: $CONVERT_SCRIPT"
    exit 1
fi

do_convert() {
    local nemo_file="$1"
    local quant="$2"

    if [[ "$nemo_file" != /* ]]; then
        nemo_file="$(cd "$(dirname "$nemo_file")" && pwd)/$(basename "$nemo_file")"
    fi

    if [[ ! -f "$nemo_file" ]]; then
        echo "Error: file not found: $nemo_file"
        return 1
    fi

    local out_file="${nemo_file%.nemo}.gguf"
    if [[ -n "$quant" ]]; then
        out_file="${nemo_file%.nemo}-${quant}.gguf"
    fi

    echo "Input:    $nemo_file"
    echo "Output:   $out_file"
    echo "Quant:    ${quant:-F16 (none)}"
    echo ""

    local cmd="python3 \"$CONVERT_SCRIPT\" --nemo \"$nemo_file\" --output \"$out_file\""
    if [[ -n "$quant" ]]; then
        cmd="$cmd --quant $quant"
    fi

    echo "Running: $cmd"
    echo ""
    eval "$cmd"

    echo ""
    echo "Done. Output file:"
    ls -lh "$out_file"
    echo ""
}

if [[ "${1:-}" == "--all" ]]; then
    QUANT="${2:-}"

    BEST_NEMOS=($(find "$CKPT_DIR" -maxdepth 1 -name 'nemotron-asr-best*-wer-*.nemo' 2>/dev/null | sort))
    if [[ ${#BEST_NEMOS[@]} -eq 0 ]]; then
        echo "Error: no best-N .nemo checkpoints found in $CKPT_DIR"
        exit 1
    fi

    echo "============================================================"
    echo "  Batch converting ${#BEST_NEMOS[@]} checkpoint(s) to GGUF"
    echo "  Quant: ${QUANT:-F16 (none)}"
    echo "============================================================"
    echo ""

    for nemo in "${BEST_NEMOS[@]}"; do
        echo "--- $(basename "$nemo") ---"
        do_convert "$nemo" "$QUANT"
    done

    FINAL_NEMO="$CKPT_DIR/nemotron-asr-finetuned.nemo"
    if [[ -f "$FINAL_NEMO" ]]; then
        echo "--- $(basename "$FINAL_NEMO") ---"
        do_convert "$FINAL_NEMO" "$QUANT"
    fi

    echo "All conversions complete."
    exit 0
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-file.nemo> [q4_k|q8_0]"
    echo "       $0 --all [q4_k|q8_0]            # convert all best-N checkpoints"
    exit 1
fi

do_convert "$1" "${2:-}"
