#!/usr/bin/env bash
# =============================================================
# train_xpu.sh — Build and run InertiaLink training on Intel Arc
# =============================================================
# Usage:
#   ./docker/train_xpu.sh                  # train (default)
#   ./docker/train_xpu.sh bash             # interactive shell
#   ./docker/train_xpu.sh python scripts/augment_seed_data.py
#
# First-time host setup (run once, then log out/in):
#   sudo usermod -aG video,render $USER
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="inertialink-xpu"

# ── Build image if it doesn't exist yet ──────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "[inertialink] Building Docker image '$IMAGE_NAME' (first run, ~5 min)..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.xpu" \
        -t "$IMAGE_NAME" \
        "$PROJECT_DIR"
else
    echo "[inertialink] Using existing image '$IMAGE_NAME'. To rebuild: docker rmi $IMAGE_NAME"
fi

# ── Resolve video/render group IDs for /dev/dri passthrough ──
VIDEO_GID=$(getent group video  2>/dev/null | cut -d: -f3 || true)
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3 || true)

GROUP_FLAGS=""
[ -n "$VIDEO_GID"  ] && GROUP_FLAGS="$GROUP_FLAGS --group-add $VIDEO_GID"
[ -n "$RENDER_GID" ] && GROUP_FLAGS="$GROUP_FLAGS --group-add $RENDER_GID"

# ── Default command if none given ────────────────────────────
if [ "$#" -eq 0 ]; then
    set -- python scripts/train_bilstm.py
fi

# ── Run ──────────────────────────────────────────────────────
# - Mount project root so generated data/ and models/ land on the host
# - Pass /dev/dri for GPU access
# - --ipc=host improves shared-memory throughput during training
echo "[inertialink] Starting training container on Intel Arc GPU..."
docker run --rm -it \
    --device=/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    $GROUP_FLAGS \
    --ipc=host \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    "$@"
