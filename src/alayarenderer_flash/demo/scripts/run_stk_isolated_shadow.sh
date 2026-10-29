#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# STK checkout layout (see demo/stk/README.md): demo/stk/rendering_game/third_party/{stk-code,stk-assets,stk-deps}.
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR%/scripts}/stk/rendering_game}"
STK_BUILD="${STK_BUILD:-${PROJECT_DIR}/third_party/stk-code/cmake_build}"
STK_DEPS="${STK_DEPS:-${PROJECT_DIR}/third_party/stk-deps}"
STK_BINARY="${STK_BINARY:-${SCRIPT_DIR}/bin/supertuxkart}"
PRODUCTION_CAPTURE_DIR="${PRODUCTION_CAPTURE_DIR:-/dev/shm/stk_browser_preview_live}"

: "${OUT_DIR:?Set OUT_DIR to a dedicated Shadow capture directory}"
: "${INPUT_STATE_PATH:?Set INPUT_STATE_PATH to the matching Shadow input.json}"

out_abs="$(readlink -m "${OUT_DIR}")"
input_abs="$(readlink -m "${INPUT_STATE_PATH}")"
production_abs="$(readlink -m "${PRODUCTION_CAPTURE_DIR}")"
if [[ "${out_abs}" == "${production_abs}" ||
      "${out_abs}" == "${production_abs}/"* ||
      "${input_abs}" == "${production_abs}" ||
      "${input_abs}" == "${production_abs}/"* ]]; then
  printf 'refusing to use production capture path: %s\n' \
    "${PRODUCTION_CAPTURE_DIR}" >&2
  exit 2
fi
if [[ ! -x "${STK_BINARY}" ]]; then
  printf 'candidate binary is missing or not executable: %s\n' \
    "${STK_BINARY}" >&2
  exit 1
fi

TRACK="${TRACK:-black_forest}"
PROFILE_TIME="${PROFILE_TIME:-86400}"
NUMKARTS="${NUMKARTS:-8}"
SCREENSIZE="${SCREENSIZE:-1280x720}"
GFX_PRESET="${GFX_PRESET:-5}"
CAPTURE_EVERY="${CAPTURE_EVERY:-1}"
CAPTURE_WARMUP="${CAPTURE_WARMUP:-0}"
CAPTURE_LIMIT="${CAPTURE_LIMIT:-0}"
CAPTURE_VIS="${CAPTURE_VIS:-0}"
CAPTURE_LATEST_ONLY="${CAPTURE_LATEST_ONLY:-1}"
SOURCE_SESSION_ID="${SOURCE_SESSION_ID:-${TRACK}-shadow-$(date +%s%N)}"

STK_HOME="${STK_HOME:-${SCRIPT_DIR}/runtime/home}"
STK_HOME_TEMPLATE="${STK_HOME_TEMPLATE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/stk_home_template}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-stk-watermark-${UID}}"
EGL_VENDOR_FILE="${EGL_VENDOR_FILE:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

mkdir -p "${OUT_DIR}" "$(dirname "${INPUT_STATE_PATH}")" \
  "${STK_HOME}/.config" "${STK_HOME}/.local/share" \
  "${STK_HOME}/.cache" "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

# A clean STK home exits before profile racing on first-run setup. Seed only
# configuration into the private Shadow home; the shared home stays read-only.
if [[ ! -d "${STK_HOME}/.config/supertuxkart" &&
      -d "${STK_HOME_TEMPLATE}/.config/supertuxkart" ]]; then
  cp -a "${STK_HOME_TEMPLATE}/.config/supertuxkart" \
    "${STK_HOME}/.config/"
fi
if [[ ! -d "${STK_HOME}/.local/share/supertuxkart" &&
      -d "${STK_HOME_TEMPLATE}/.local/share/supertuxkart" ]]; then
  cp -a "${STK_HOME_TEMPLATE}/.local/share/supertuxkart" \
    "${STK_HOME}/.local/share/"
fi
if [[ ! -f "${EGL_VENDOR_FILE}" ]]; then
  printf 'NVIDIA EGL vendor file is missing: %s\n' "${EGL_VENDOR_FILE}" >&2
  exit 1
fi

export HOME="${STK_HOME}"
unset XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME
export XDG_RUNTIME_DIR
export PATH="${STK_DEPS}/bin:${PATH}"
export LD_LIBRARY_PATH="${STK_DEPS}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${STK_DEPS}/lib:${LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE}"
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=offscreen
# Pace profile-mode capture to the renderer (live deployment uses 34 FPS).
export ALAYA_STK_CAPTURE_FPS="${ALAYA_STK_CAPTURE_FPS:-34}"
export ALSOFT_DRIVERS=null
export STK_GBUFFER_CAPTURE_DIR="${OUT_DIR}"
export STK_BROWSER_INPUT_FILE="${INPUT_STATE_PATH}"
export STK_GBUFFER_CAPTURE_EVERY="${CAPTURE_EVERY}"
export STK_GBUFFER_CAPTURE_WARMUP="${CAPTURE_WARMUP}"
export STK_GBUFFER_CAPTURE_LIMIT="${CAPTURE_LIMIT}"
export STK_GBUFFER_CAPTURE_VIS="${CAPTURE_VIS}"
export STK_GBUFFER_CAPTURE_LATEST_ONLY="${CAPTURE_LATEST_ONLY}"
if [[ -n "${STK_BROWSER_CONTROL_KART:-}" ]]; then
  export STK_BROWSER_CONTROL_KART
fi

cat >"${OUT_DIR}/run_config.env" <<EOF
TRACK=${TRACK}
PROFILE_TIME=${PROFILE_TIME}
NUMKARTS=${NUMKARTS}
SCREENSIZE=${SCREENSIZE}
GFX_PRESET=${GFX_PRESET}
CAPTURE_EVERY=${CAPTURE_EVERY}
CAPTURE_WARMUP=${CAPTURE_WARMUP}
CAPTURE_LIMIT=${CAPTURE_LIMIT}
CAPTURE_VIS=${CAPTURE_VIS}
CAPTURE_LATEST_ONLY=${CAPTURE_LATEST_ONLY}
OUT_DIR=${OUT_DIR}
INPUT_STATE_PATH=${INPUT_STATE_PATH}
SOURCE_SESSION_ID=${SOURCE_SESSION_ID}
STK_BINARY=${STK_BINARY}
EOF

cd "${STK_BUILD}"
exec "${STK_BINARY}" \
  -R \
  --track="${TRACK}" \
  --profile-time="${PROFILE_TIME}" \
  --numkarts="${NUMKARTS}" \
  --screensize="${SCREENSIZE}" \
  --gfx-preset="${GFX_PRESET}" \
  --enable-dynamic-lights \
  --enable-ssao \
  --disable-ssr \
  --rtt-scale=100 \
  --log=3
