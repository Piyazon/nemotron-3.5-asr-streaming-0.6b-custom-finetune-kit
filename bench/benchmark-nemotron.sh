#!/bin/bash
set -e

###############################################################################
# benchmark-nemotron.sh
#
# End-to-end orchestrator: finds all best-N .nemo checkpoints, converts each
# to Q8_0 GGUF, then benchmarks all models against the test manifest.
#
#   1. Pretrained base (auto-downloaded via crispasr)
#   2. Best #1 checkpoint (lowest WER)
#   3. Best #2 checkpoint
#   4. Best #3 checkpoint
#   5. Epoch-40 finetuned model (existing Q8_0 GGUF, optional)
#
# Requires: CrispASR binary + convert-nemotron-to-gguf.py
# Usage:    bash benchmark-nemotron.sh
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT_DIR="$KIT_ROOT/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/test"
EPOCH40_GGUF="$KIT_ROOT/checkpoints_bestworking/FastConformer-Transducer-BPE-Prompt-Streaming/test/nemotron-asr-finetuned-q8_0_epoch40.gguf"

MANIFEST="$KIT_ROOT/custom_asr_data/test_manifest.json"
AUDIO_BASE="$KIT_ROOT/custom_asr_data/wavs"

# ---- CrispASR paths (set CRISPASR_BIN / CONVERT_SCRIPT envvars or edit here) ----
CRISPASR="${CRISPASR_BIN:-crispasr}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-$KIT_ROOT/../CrispASR/models/convert-nemotron-to-gguf.py}"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "Error: conversion script not found at $CONVERT_SCRIPT"
    echo "Set CONVERT_SCRIPT envvar or place convert-nemotron-to-gguf.py at ../CrispASR/models/"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: test manifest not found at $MANIFEST"
    exit 1
fi

# ============================================================================
# Discover best-N .nemo checkpoints and convert to Q8_0 GGUF
# ============================================================================
echo ""
echo "============================================================"
echo "  Phase 0: Converting best-N checkpoints to Q8_0 GGUF"
echo "============================================================"

BEST_NEMOS=($(find "$CKPT_DIR" -maxdepth 1 -name 'nemotron-asr-best*-wer-*.nemo' 2>/dev/null | sort))
if [[ ${#BEST_NEMOS[@]} -eq 0 ]]; then
    echo "Error: no best-N .nemo checkpoints found in $CKPT_DIR"
    echo "Run the finetuning script first (it saves top-3 as nemotron-asr-best{1,2,3}-wer-*.nemo)"
    exit 1
fi

echo "Found ${#BEST_NEMOS[@]} best checkpoint(s):"
for nemo in "${BEST_NEMOS[@]}"; do
    echo "  - $(basename "$nemo")"
done
echo ""

TMP_ROOT="$(mktemp -d)"
GGUF_DIR="$TMP_ROOT/gguf"
mkdir -p "$GGUF_DIR"

CONVERTED_GGUFS=()
for nemo in "${BEST_NEMOS[@]}"; do
    out_gguf="$GGUF_DIR/$(basename "${nemo%.nemo}-q8_0.gguf")"
    echo "--> Converting $(basename "$nemo") -> Q8_0 ..."
    python3 "$CONVERT_SCRIPT" --nemo "$nemo" --output "$out_gguf" --quant q8_0 2>&1 | tail -1
    CONVERTED_GGUFS+=("$out_gguf")
    echo "    -> $(du -h "$out_gguf" | cut -f1)"
done

# ============================================================================
# Assemble the full model list (labels + paths) for benchmarking
# ============================================================================
MODEL_LABELS=("Pretrained Base (auto)" "Best #1 (lowest WER)" "Best #2" "Best #3" "Epoch-40 Finetuned")
MODEL_PATHS=()

MODEL_PATHS+=("auto")

for gguf in "${CONVERTED_GGUFS[@]}"; do
    MODEL_PATHS+=("$gguf")
done

HAS_EPOCH40=true
if [[ -f "$EPOCH40_GGUF" ]]; then
    MODEL_PATHS+=("$EPOCH40_GGUF")
    echo ""
    echo "  Epoch-40 GGUF: $(du -h "$EPOCH40_GGUF" | cut -f1)"
else
    HAS_EPOCH40=false
    echo ""
    echo "  WARNING: Epoch-40 GGUF not found — skipping."
fi

LABEL_LIST=("${MODEL_LABELS[0]}")
for i in 1 2 3; do
    if [[ ${#CONVERTED_GGUFS[@]} -ge $i ]]; then
        LABEL_LIST+=("${MODEL_LABELS[$i]}")
    fi
done
if [[ "$HAS_EPOCH40" == true ]]; then
    LABEL_LIST+=("${MODEL_LABELS[4]}")
fi

TOTAL_MODELS=${#MODEL_PATHS[@]}

FILE_IDS=($(python3 -c "
import json, os
for line in open('$MANIFEST'):
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    path = obj['audio_filepath']
    print(os.path.splitext(os.path.basename(path))[0])
"))

echo ""
echo "============================================================"
echo "  Benchmarking $TOTAL_MODELS models against ${#FILE_IDS[@]} test audios"
echo "============================================================"
echo "Found ${#FILE_IDS[@]} test audio files in manifest"
echo ""

RESULT_DIRS=()
for i in $(seq 0 $((TOTAL_MODELS - 1))); do
    d="$TMP_ROOT/model_${i}"
    mkdir -p "$d"
    RESULT_DIRS+=("$d")
done

trap 'rm -rf "$TMP_ROOT"' EXIT

# ============================================================================
# Benchmark each model (runs crispasr on all audios)
# ============================================================================
for i in $(seq 0 $((TOTAL_MODELS - 1))); do
    label="${LABEL_LIST[$i]}"
    path="${MODEL_PATHS[$i]}"
    out_dir="${RESULT_DIRS[$i]}"

    echo "=========================================="
    echo "Model $((i+1))/$TOTAL_MODELS: $label"
    if [[ "$path" != "auto" ]]; then
        echo "$path"
    else
        echo "(auto-downloaded pretrained base)"
    fi
    echo "=========================================="

    for fid in "${FILE_IDS[@]}"; do
        WAV="$AUDIO_BASE/${fid}.wav"
        if [[ ! -f "$WAV" ]]; then
            echo "  SKIP $fid (file not found: $WAV)"
            continue
        fi
        echo -n "  $fid ... "
        if [[ "$path" == "auto" ]]; then
            $CRISPASR --backend nemotron -m auto --auto-download -f "$WAV" \
                > "$out_dir/${fid}.txt" 2>"$out_dir/${fid}.stderr.txt" && echo "OK" || echo "FAILED"
        else
            $CRISPASR --backend nemotron -m "$path" -f "$WAV" \
                > "$out_dir/${fid}.txt" 2>"$out_dir/${fid}.stderr.txt" && echo "OK" || echo "FAILED"
        fi
    done
    echo ""
done

# ============================================================================
# Multi-model WER comparison (all models side by side)
# ============================================================================
echo "============================================================"
echo "  MULTI-MODEL WER COMPARISON"
echo "============================================================"

python3 - "$MANIFEST" "$TOTAL_MODELS" "${RESULT_DIRS[@]}" "${LABEL_LIST[@]}" << 'PYEOF'
import sys, json, re, os
from difflib import SequenceMatcher

manifest_path = sys.argv[1]
total_models  = int(sys.argv[2])
result_dirs   = sys.argv[3:3+total_models]
model_labels  = sys.argv[3+total_models:]

def normalize(s):
    s = re.sub(r'<[^>]+>', '', s).lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def wer(ref, hyp):
    rw = normalize(ref).split()
    hw = normalize(hyp).split()
    n, m = len(rw), len(hw)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if rw[i-1] == hw[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m], n

def read_transcript(d, fid):
    p = os.path.join(d, fid + ".txt")
    if not os.path.exists(p):
        return None
    return open(p).read().strip()

entries = []
for line in open(manifest_path):
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    fid = os.path.splitext(os.path.basename(obj['audio_filepath']))[0]
    entries.append((fid, obj.get('text', '')))

totals_err = [0] * total_models
totals_ref = [0] * total_models

col_width = 18
hdr = f"{'File':<12}"
for i in range(total_models):
    lbl = model_labels[i] if i < len(model_labels) else f"Model-{i+1}"
    hdr += f"{lbl:<{col_width}}"
hdr += f" {'Best Model':^16}"

print(hdr)
print("-" * len(hdr))

for fid, gt in entries:
    transcripts = [read_transcript(result_dirs[i], fid) for i in range(total_models)]
    if None in transcripts:
        print(f"{fid:<12} {'N/A':<{col_width * total_models}} skipped")
        continue

    wers = []
    row = f"{fid:<12}"
    for i in range(total_models):
        e, n = wer(gt, transcripts[i])
        totals_err[i] += e; totals_ref[i] += n
        pct = f"{e/n*100:.1f}%" if n else "N/A"
        wers.append((e, n))
        row += f" {e}/{n}({pct}){'':<{col_width - len(f'{e}/{n}({pct})') - 2}}"

    best_idx = min(range(total_models), key=lambda k: wers[k][0] if wers[k][1] else 999)
    best_label = (model_labels[best_idx] if best_idx < len(model_labels) else f"M{best_idx+1}")[:8]
    row += f" {best_label}"
    print(row)

print("-" * len(hdr))

row = f"{'TOTAL':<12}"
for i in range(total_models):
    e, n = totals_err[i], totals_ref[i]
    pct = f"{e/n*100:.1f}%" if n else "N/A"
    lbl = model_labels[i] if i < len(model_labels) else f"M{i+1}"
    row += f" {e}/{n}({pct}){'':<{col_width - len(f'{e}/{n}({pct})') - 2}}"

overall_best = min(range(total_models), key=lambda k: totals_err[k] if totals_ref[k] else 999)
best_label = (model_labels[overall_best] if overall_best < len(model_labels) else f"M{overall_best+1}")[:8]
row += f" {best_label}"
print(row)

print()
print("=" * len(hdr))
print("  RANKING (lowest WER = best)")
print("=" * len(hdr))
ranking = sorted(range(total_models), key=lambda k: totals_err[k]/totals_ref[k] if totals_ref[k] else 999)
for rank, idx in enumerate(ranking, 1):
    lbl = model_labels[idx] if idx < len(model_labels) else f"Model-{idx+1}"
    e, n = totals_err[idx], totals_ref[idx]
    pct = f"{e/n*100:.1f}%" if n else "N/A"
    marker = " <-- BEST" if rank == 1 else ""
    print(f"  #{rank}. {lbl:<24} WER: {pct}  ({e}/{n}){marker}")

print()
print("=" * 160)
col_hdr = f"{'File':<10}"
for i in range(total_models):
    lbl = model_labels[i] if i < len(model_labels) else f"M{i+1}"
    col_hdr += f" | {lbl}"
print(col_hdr)
print("-" * 160)

def highlight(ref_words, hyp_words):
    rw = ref_words.split()
    hw = hyp_words.split()
    result = []
    sm = SequenceMatcher(None, rw, hw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            result.append(' '.join(hw[j1:j2]))
        elif tag == 'replace':
            for w in hw[j1:j2]:
                result.append("[" + w + "]")
        elif tag == 'delete':
            missing = ' '.join(rw[i1:i2])
            result.append("<<" + missing + ">>")
        elif tag == 'insert':
            extra = ' '.join(hw[j1:j2])
            result.append("(+" + extra + ")")
    return ' '.join(result)

for fid, gt in entries:
    transcripts = [read_transcript(result_dirs[i], fid) or "N/A" for i in range(total_models)]
    gt_n = normalize(gt)
    row = f"{fid:<10}"
    for i in range(total_models):
        if transcripts[i] == "N/A":
            row += " | N/A"
        else:
            hl = highlight(gt_n, normalize(transcripts[i]))
            if len(hl) > 60: hl = hl[:57] + "..."
            row += f" | {hl}"
    print(row)

print()
print("[word]  = wrong/substituted word")
print("<<words>> = missing from hypothesis")
print("(+words) = extra in hypothesis")
PYEOF

# ============================================================================
# Final summary — copy GGUFs back to checkpoint dir for persistence
# ============================================================================
echo ""
echo "============================================================"
echo "  FINAL SUMMARY — GGUF Files"
echo "============================================================"
echo ""
for i in $(seq 0 $((TOTAL_MODELS - 1))); do
    path="${MODEL_PATHS[$i]}"
    label="${LABEL_LIST[$i]:-Model-$((i+1))}"
    if [[ "$path" == "auto" ]]; then
        echo "  Model $((i+1)): $label (auto-downloaded, no local file)"
    elif [[ -f "$path" ]]; then
        echo "  Model $((i+1)): $label"
        echo "          $(du -h "$path" | cut -f1)  $path"
    fi
done
echo ""

echo "Copying Q8_0 GGUF files to checkpoint directory ..."
for gguf in "${CONVERTED_GGUFS[@]}"; do
    cp "$gguf" "$CKPT_DIR/"
    echo "  -> $CKPT_DIR/$(basename "$gguf")"
done

echo ""
echo "Done."
