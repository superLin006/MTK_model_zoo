#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/moonshine_streaming_test"
MODELS="$ROOT_DIR/models"

# Copy MTK runtime libraries from device-local zipformer test directory
cp /data/local/tmp/zipformer_mtk_test/libapu_mdw.so        "$ROOT_DIR/lib/" 2>/dev/null || true
cp /data/local/tmp/zipformer_mtk_test/libapu_mdw_batch.so  "$ROOT_DIR/lib/" 2>/dev/null || true
cp /data/local/tmp/zipformer_mtk_test/libbase.so           "$ROOT_DIR/lib/" 2>/dev/null || true
cp /data/local/tmp/zipformer_mtk_test/libcutils.so         "$ROOT_DIR/lib/" 2>/dev/null || true
cp /data/local/tmp/zipformer_mtk_test/libdmabufheap.so     "$ROOT_DIR/lib/" 2>/dev/null || true
cp /data/local/tmp/zipformer_mtk_test/libneuron_runtime.8.so "$ROOT_DIR/lib/" 2>/dev/null || true

export LD_LIBRARY_PATH="$ROOT_DIR/lib:$LD_LIBRARY_PATH"

run_one() {
    audio="$1"
    echo "=== 流式测试: $(basename "$audio") ==="
    "$EXEC" \
        "$MODELS/moonshine_encoder_chunk.dla" \
        "$MODELS/moonshine_decoder.dla" \
        "$MODELS/embed_tokens.npy" \
        "$MODELS/proj_weight.npy" \
        "$MODELS/log_k.npy" \
        "$MODELS/vocab.txt" \
        "$audio"
}

ARG1="${1:-}"
if [ -z "$ARG1" ]; then
    for f in "$SCRIPT_DIR"/*.wav; do [ -f "$f" ] && run_one "$f"; done
else
    case "$ARG1" in /*) run_one "$ARG1" ;; *) run_one "$SCRIPT_DIR/$ARG1" ;; esac
fi
