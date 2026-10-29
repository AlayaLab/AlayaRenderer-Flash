#!/usr/bin/env bash
set -euo pipefail
export ALAYA_PREVIEW_REFERENCE_ROOT="${ALAYA_PREVIEW_REFERENCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ALAYA_PREVIEW_GAME_HTML="${ALAYA_PREVIEW_GAME_HTML:-${ALAYA_PREVIEW_REFERENCE_ROOT}/assets/game.html}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}" || printf %s "${PYTHON_BIN:-python3}")"
CAPTURE_DIR="${CAPTURE_DIR:-/dev/shm/stk_browser_preview_live}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_DIR}/.runtime/alaya_live_control}"
WINDOW_DIR="${WINDOW_DIR:-/dev/shm/stk_alaya_windows}"
RENDERED_DIR="${RENDERED_DIR:-/dev/shm/stk_alaya_rendered}"
INPUT_STATE_PATH="${INPUT_STATE_PATH:-${CAPTURE_DIR}/input.json}"
PREVIEW_HOST="${PREVIEW_HOST:-127.0.0.1}"
PREVIEW_PORT="${PREVIEW_PORT:-8091}"
PREVIEW_MAIN_WIDTH="${PREVIEW_MAIN_WIDTH:-720}"
PREVIEW_GBUFFER_WIDTH="${PREVIEW_GBUFFER_WIDTH:-360}"
PREVIEW_JPEG_QUALITY="${PREVIEW_JPEG_QUALITY:-72}"
PREVIEW_MAX_FPS="${PREVIEW_MAX_FPS:-36}"
PREVIEW_GBUFFER_FPS="${PREVIEW_GBUFFER_FPS:-6}"
READ_ONLY_OBSERVER="${READ_ONLY_OBSERVER:-0}"
ALAYA_VALIDATE_ONLY="${ALAYA_VALIDATE_ONLY:-0}"

case "${READ_ONLY_OBSERVER}" in
  1|true|TRUE|yes|YES|on|ON) observer=1 ;;
  0|false|FALSE|no|NO|off|OFF) observer=0 ;;
  *)
    printf 'READ_ONLY_OBSERVER must be a boolean, got %s\n' "${READ_ONLY_OBSERVER}" >&2
    exit 2
    ;;
esac

export STK_PROMPT_CONTROL_DIR="${CONTROL_DIR}"
export STK_ALAYA_WINDOW_DIR="${WINDOW_DIR}"

args=(
  "${SCRIPT_DIR}/../preview.py"
  --capture-dir "${CAPTURE_DIR}"
  --rendered-dir "${RENDERED_DIR}"
  --input-state "${INPUT_STATE_PATH}"
  --host "${PREVIEW_HOST}"
  --port "${PREVIEW_PORT}"
  --main-width "${PREVIEW_MAIN_WIDTH}"
  --gbuffer-width "${PREVIEW_GBUFFER_WIDTH}"
  --jpeg-quality "${PREVIEW_JPEG_QUALITY}"
  --max-fps "${PREVIEW_MAX_FPS}"
  --gbuffer-fps "${PREVIEW_GBUFFER_FPS}"
)
if [[ "${observer}" == 1 ]]; then
  args+=(--read-only-observer)
fi

required_paths=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/../preview.py"
  "${CAPTURE_DIR}"
  "${RENDERED_DIR}"
  "${CONTROL_DIR}/live_control.json"
  "${WINDOW_DIR}/manifest.json"
  "${INPUT_STATE_PATH}"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required preview path is missing: %s\n' "${path}" >&2
    exit 1
  fi
done

if [[ "${ALAYA_VALIDATE_ONLY}" == 1 ]]; then
  printf 'command='
  printf '%q ' "${PYTHON_BIN}" "${args[@]}"
  printf '\n'
  exit 0
fi

exec "${PYTHON_BIN}" "${args[@]}"
