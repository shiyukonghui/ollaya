#!/bin/sh
# Build the local GPU pack (`lib/ollaya/cuda_v13/<dlls>` + `lib/ollaya/ollaya-cuda-runner.exe`)
# from wheels on PyPI, so an offline/dev machine can run NeoHorse (and every other ONNX model) on
# the CUDA execution provider without Microsoft's GitHub release zip (unreachable here).
#
# What it does (idempotent):
#   1. Downloads the ORT GPU wheel with `pip download` (no venv is modified):
#        onnxruntime-gpu==1.28.0  -> onnxruntime.dll, onnxruntime_providers_shared.dll,
#                                   onnxruntime_providers_cuda.dll
#      The Windows wheel on PyPI is Microsoft's CUDA 13 build (its provider imports
#      cublas64_13/cublasLt64_13), i.e. the same build the official `_cuda13` zip carries.
#      ADR 0004 pins 1.28.2, but Microsoft publishes that only as a GitHub release asset, which is
#      not reachable from this machine; 1.28.0 is the same minor (the same operators and C API, and
#      `ort` rc.13's api-24 needs >= 1.24), so it is the closest wheel PyPI serves.
#   2. Downloads the NVIDIA CUDA 13 runtime wheels at the exact versions the official pack pins
#      (packaging/cuda-requirements.txt) -- the unified `nvidia-cublas`, `nvidia-cuda-runtime`,
#      `nvidia-cufft`, `nvidia-curand`, `nvidia-cuda-nvrtc`, `nvidia-nvjitlink` packages plus
#      `nvidia-cudnn-cu13` -- and keeps only the DLLs `scripts/package.sh`'s `is_cuda_lib` keeps.
#   3. Lays the pack out like the archive: `<pack>/cuda_v13/` (every DLL) and, when a
#      `--features ollaya-runner/cuda-dynamic` build is available, `<pack>/ollaya-cuda-runner.exe`
#      next to it. Point `OLLAYA_LIBRARY_PATH` at `<pack>` and the daemon starts GPU runners from
#      `cuda_v13/ollaya-runner-<hash>.exe` with `ORT_DYLIB_PATH` (crates/ollaya-server/src/launch.rs).
#
# Settings, all optional:
#   OLLAYA_CUDA_PACK  the pack root to write (default <repo>/out/cuda-pack)
#   ORT_GPU_VERSION   the onnxruntime-gpu version      (default 1.28.0)
#   PYTHON            a Python 3 interpreter with pip  (default `python`)
#   NO_BUILD=1        do not build/copy the cuda runner, only assemble the DLLs

set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
PACK=${OLLAYA_CUDA_PACK:-$repo/out/cuda-pack}
ORT_GPU_VERSION=${ORT_GPU_VERSION:-1.28.0}
PYTHON=${PYTHON:-python}
CUDA_V=13
CUFFT_V=12
DL=$PACK/dl

command -v "$PYTHON" >/dev/null 2>&1 ||
    { echo "ERROR: no $PYTHON; set PYTHON to a Python 3 interpreter" >&2; exit 1; }

mkdir -p "$DL"

# The NVIDIA wheels, at the versions packaging/cuda-requirements.txt pins for the CUDA 13 pack.
NVIDIA_WHEELS="nvidia-cublas==13.8.0.4 nvidia-cuda-runtime==13.4.92 nvidia-cuda-nvrtc==13.4.92 \
nvidia-cufft==12.4.0.43 nvidia-curand==10.4.4.72 nvidia-nvjitlink==13.4.92 nvidia-cudnn-cu13==9.26.0.51"

echo "== downloading wheels into $DL"
PYTHONUTF8=1 "$PYTHON" -m pip download --no-deps -d "$DL" \
    "onnxruntime-gpu==$ORT_GPU_VERSION" $NVIDIA_WHEELS

echo "== assembling $PACK/cuda_v$CUDA_V"
CUDA_V=$CUDA_V CUFFT_V=$CUFFT_V PACK="$PACK" PYTHONUTF8=1 "$PYTHON" - <<'PYEOF'
import glob, os, sys, zipfile

pack = os.path.abspath(os.environ["PACK"])
cuda_v = os.environ["CUDA_V"]
cufft_v = os.environ["CUFFT_V"]
dl = os.path.join(pack, "dl")
out = os.path.join(pack, "cuda_v" + cuda_v)
os.makedirs(out, exist_ok=True)

# The same filter as scripts/package.sh's is_cuda_lib for windows-amd64, CUDA 13.
def keep(name):
    wanted = [
        "cudart64_%s.dll" % cuda_v, "cublas64_%s.dll" % cuda_v, "cublasLt64_%s.dll" % cuda_v,
        "cufft64_%s.dll" % cufft_v, "curand64_10.dll",
        "nvrtc64_%s0_0.dll" % cuda_v, "nvJitLink_%s0_0.dll" % cuda_v,
        "cudnn64_9.dll", "cudnn_graph64_9.dll",
    ]
    if name in wanted:
        return True
    if name.startswith("cudnn") and name.endswith("64_9.dll"):
        return True
    if name.startswith("nvrtc-builtins64_%s" % cuda_v) and name.endswith(".dll"):
        return True
    return False

def dll_base(name):
    return os.path.basename(name)

ort_providers = ["onnxruntime.dll", "onnxruntime_providers_shared.dll",
                 "onnxruntime_providers_cuda.dll"]
ort = glob.glob(os.path.join(dl, "onnxruntime_gpu-*.whl"))
if len(ort) != 1:
    sys.exit("ERROR: expected one onnxruntime_gpu wheel in %s, found %d" % (dl, len(ort)))

placed = []
with zipfile.ZipFile(ort[0]) as z:
    for name in z.namelist():
        if dll_base(name) in ort_providers and name.startswith("onnxruntime/capi/"):
            dst = os.path.join(out, dll_base(name))
            with open(dst, "wb") as f:
                f.write(z.read(name))
            placed.append(dll_base(name))

for whl in sorted(glob.glob(os.path.join(dl, "nvidia_*.whl"))):
    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            # Windows wheels keep the DLLs in nvidia/<pkg>/bin/; leave anything else (static and
            # import libs, headers, .lib files) behind.
            if "/bin/" not in name.replace("\\", "/"):
                continue
            if not name.lower().endswith(".dll"):
                continue
            if not keep(dll_base(name)):
                continue
            dst = os.path.join(out, dll_base(name))
            with open(dst, "wb") as f:
                f.write(z.read(name))
            placed.append(dll_base(name))

missing = [p for p in ort_providers if p not in placed]
if missing:
    sys.exit("ERROR: the ORT wheel did not provide %s" % ", ".join(missing))
required = ["cudart64_%s.dll" % cuda_v, "cublas64_%s.dll" % cuda_v, "cublasLt64_%s.dll" % cuda_v,
            "cufft64_%s.dll" % cufft_v, "curand64_10.dll", "cudnn64_9.dll", "cudnn_graph64_9.dll"]
cuda_missing = [p for p in required if not os.path.isfile(os.path.join(out, p))]
if cuda_missing:
    sys.exit("ERROR: the NVIDIA wheels did not provide %s" % ", ".join(cuda_missing))
print("  %d DLLs in %s" % (len(placed), out))
for p in sorted(set(placed)):
    print("    %-30s %8.1f MB" % (p, os.path.getsize(os.path.join(out, p)) / 1e6))
PYEOF

echo "== pack ready"
ls -la "$PACK/cuda_v$CUDA_V" | head -30
echo
echo "run a GPU runner with:"
echo "  export OLLAYA_LIBRARY_PATH=$PACK"
echo "  export ORT_DYLIB_PATH=$PACK/cuda_v$CUDA_V/onnxruntime.dll   # only for direct/dev runs"
