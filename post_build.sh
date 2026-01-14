#!/bin/bash
# Post-build script for radiance_meshes
# Builds optional CUDA extensions: tiny-cuda-nn (tcnn), fused-ssim, and pyGDel3D

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCNN_DIR="${SCRIPT_DIR}/submodules/tcnn"
FUSED_SSIM_DIR="${SCRIPT_DIR}/submodules/fused-ssim"
GDEL3D_DIR="${SCRIPT_DIR}/submodules/pyGDel3D"

# ------------------------------
# ARGUMENT PARSING
# ------------------------------

BUILD_TCNN=false
BUILD_FUSED_SSIM=false
BUILD_GDEL3D=false
BUILD_ALL=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build optional CUDA extensions for radiance_meshes."
    echo ""
    echo "Options:"
    echo "  --tcnn         Build tiny-cuda-nn PyTorch bindings"
    echo "  --fused-ssim   Build fused-ssim CUDA extension"
    echo "  --gdel3d       Build pyGDel3D (Delaunay triangulation)"
    echo "  --all          Build all optional extensions"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Build everything"
    echo "  $0 --tcnn                   # Build only tcnn"
    echo "  $0 --fused-ssim             # Build only fused-ssim"
    echo "  $0 --tcnn --fused-ssim      # Build both explicitly"
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --tcnn)
            BUILD_TCNN=true
            shift
            ;;
        --fused-ssim)
            BUILD_FUSED_SSIM=true
            shift
            ;;
        --gdel3d)
            BUILD_GDEL3D=true
            shift
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ "$BUILD_ALL" = true ]; then
    BUILD_TCNN=true
    BUILD_FUSED_SSIM=true
    BUILD_GDEL3D=true
fi

if [ "$BUILD_TCNN" = false ] && [ "$BUILD_FUSED_SSIM" = false ] && [ "$BUILD_GDEL3D" = false ]; then
    echo "No build targets specified. Use --help for usage."
    exit 0
fi

# ------------------------------
# CUDA DETECTION
# ------------------------------

find_cuda() {
    CUDA_PATH=""
    for cuda_version in 12.8 12.6 12.4 12.2 12.1 12.0 11.8; do
        if [ -d "/usr/local/cuda-${cuda_version}" ]; then
            CUDA_PATH="/usr/local/cuda-${cuda_version}"
            break
        fi
    done

    if [ -z "${CUDA_PATH}" ] && [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    fi

    if [ -z "${CUDA_PATH}" ]; then
        echo "ERROR: CUDA installation not found in /usr/local/cuda*"
        exit 1
    fi

    echo "Using CUDA: ${CUDA_PATH}"
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
}

# ------------------------------
# BUILD FUNCTIONS
# ------------------------------

build_tcnn() {
    echo ""
    echo "=== Building tiny-cuda-nn PyTorch bindings ==="

    # Initialize git submodules (tcnn has nested dependencies)
    echo "[1/2] Initializing tcnn submodules..."
    cd "${TCNN_DIR}"
    git submodule update --init --recursive

    # Build and install tcnn PyTorch bindings
    echo "[2/2] Building tcnn PyTorch bindings..."
    cd "${TCNN_DIR}/bindings/torch"
    rm -rf build/ *.egg-info src/*.o 2>/dev/null || true

    python setup.py install

    echo ""
    echo "=== tiny-cuda-nn build complete ==="
    echo "Verify with: python -c 'import tinycudann; print(tinycudann.__version__)'"
}

build_fused_ssim() {
    echo ""
    echo "=== Building fused-ssim CUDA extension ==="

    cd "${FUSED_SSIM_DIR}"

    # Clean previous build artifacts
    echo "[1/2] Cleaning previous build..."
    rm -rf build/ *.egg-info 2>/dev/null || true

    # Build and install
    echo "[2/2] Building fused-ssim..."
    python setup.py install

    echo ""
    echo "=== fused-ssim build complete ==="
    echo "Verify with: python -c 'import torch; from fused_ssim_cuda import fusedssim; print(\"OK\")'"
}

build_gdel3d() {
    echo ""
    echo "=== Building pyGDel3D (Delaunay triangulation) ==="

    cd "${GDEL3D_DIR}"

    # Clean previous build artifacts
    echo "[1/2] Cleaning previous build..."
    rm -rf build/ *.egg-info 2>/dev/null || true

    # Build and install
    echo "[2/2] Building pyGDel3D..."
    python setup.py install

    echo ""
    echo "=== pyGDel3D build complete ==="
    echo "Verify with: python -c 'import gdel3d; print(\"OK\")'"
}

# ------------------------------
# MAIN
# ------------------------------

find_cuda

if [ "$BUILD_TCNN" = true ]; then
    build_tcnn
fi

if [ "$BUILD_FUSED_SSIM" = true ]; then
    build_fused_ssim
fi

if [ "$BUILD_GDEL3D" = true ]; then
    build_gdel3d
fi

echo ""
echo "=== All requested builds complete ==="
