#!/usr/bin/env bash
# =============================================================
# decode_xpu.sh — Run ONNX decoder/eval on Intel Arc with CPU fallback
# =============================================================
# Usage:
#   ./docker/decode_xpu.sh
#   ./docker/decode_xpu.sh python3 scripts/eval_model.py
#   ./docker/decode_xpu.sh python3 scripts/eval_model.py 1 2 3
#
# Env:
#   MODEL_DEVICE=auto|xpu|gpu|cpu   (default: auto)
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="inertialink-xpu"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "[inertialink] Building Docker image '$IMAGE_NAME' (first run, ~5 min)..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.xpu" \
        -t "$IMAGE_NAME" \
        "$PROJECT_DIR"
else
    echo "[inertialink] Using existing image '$IMAGE_NAME'. To rebuild: docker rmi $IMAGE_NAME"
fi

VIDEO_GID=$(getent group video 2>/dev/null | cut -d: -f3 || true)
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3 || true)

GROUP_FLAGS=""
[ -n "$VIDEO_GID" ] && GROUP_FLAGS="$GROUP_FLAGS --group-add $VIDEO_GID"
[ -n "$RENDER_GID" ] && GROUP_FLAGS="$GROUP_FLAGS --group-add $RENDER_GID"

if [ "$#" -eq 0 ]; then
    set -- python3 scripts/eval_model.py
fi

MODEL_DEVICE="${MODEL_DEVICE:-auto}"
echo "[inertialink] Starting decoder container (MODEL_DEVICE=$MODEL_DEVICE)..."

TTY_FLAGS=""
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS="-it"
fi

docker run --rm $TTY_FLAGS \
    --device=/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    $GROUP_FLAGS \
    --ipc=host \
    -e MODEL_DEVICE="$MODEL_DEVICE" \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    "$@"
