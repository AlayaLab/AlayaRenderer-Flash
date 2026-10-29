#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

LAUNCHER_SCHEMA_VERSION=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage:
  alaya_stk_checkpoint_service.sh [--config FILE] doctor [--json]
  alaya_stk_checkpoint_service.sh [--config FILE] start
  alaya_stk_checkpoint_service.sh [--config FILE] status [--json]
  alaya_stk_checkpoint_service.sh [--config FILE] logs [SERVICE] [--follow] [--lines N]
  alaya_stk_checkpoint_service.sh [--config FILE] stop

Services: stk, bridge, worker, preview, or all.

The config file is an operator-owned Bash environment file. Model paths, the
validated STK binary, and their expected SHA256 values are required by default
for doctor/start.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
  if [[ -n "${DOCTOR_WARNINGS:-}" ]]; then
    DOCTOR_WARNINGS="${DOCTOR_WARNINGS}"$'\n'
  fi
  DOCTOR_WARNINGS="${DOCTOR_WARNINGS:-}$*"
}

CONFIG_FILE="${ALAYA_CONFIG_FILE:-}"
if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 3 ]] || die '--config requires FILE and COMMAND'
  CONFIG_FILE="$2"
  shift 2
fi

COMMAND="${1:-}"
[[ -n "${COMMAND}" ]] || { usage >&2; exit 2; }
shift || true

if [[ -n "${CONFIG_FILE}" ]]; then
  [[ -r "${CONFIG_FILE}" ]] || die "config is not readable: ${CONFIG_FILE}"
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")"
fi

OUTPUT_JSON=0
LOG_SERVICE=all
LOG_FOLLOW=0
LOG_LINES="${ALAYA_LOG_LINES:-200}"

case "${COMMAND}" in
  doctor|status)
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --json) OUTPUT_JSON=1 ;;
        *) die "unknown ${COMMAND} option: $1" ;;
      esac
      shift
    done
    ;;
  logs)
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --follow|-f) LOG_FOLLOW=1 ;;
        --lines)
          [[ $# -ge 2 ]] || die '--lines requires an integer'
          LOG_LINES="$2"
          shift
          ;;
        stk|bridge|worker|preview|all) LOG_SERVICE="$1" ;;
        *) die "unknown logs option or service: $1" ;;
      esac
      shift
    done
    [[ "${LOG_LINES}" =~ ^[1-9][0-9]*$ ]] || die '--lines must be a positive integer'
    ;;
  start|stop)
    [[ $# -eq 0 ]] || die "${COMMAND} takes no positional arguments"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac

LAUNCHER_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALAYA_STATE_DIR="${ALAYA_STATE_DIR:-${LAUNCHER_PROJECT_DIR}/.runtime/stk-flash-service}"
CURRENT_RUNTIME_POINTER="${ALAYA_CURRENT_RUNTIME_POINTER:-${ALAYA_STATE_DIR}/current-runtime}"
HOST_PYTHON="${ALAYA_HOST_PYTHON:-python3}"

canonical_path() {
  "${HOST_PYTHON}" - "$1" <<'PY'
import os
import sys
print(os.path.realpath(os.path.expanduser(sys.argv[1])))
PY
}

utc_now() {
  "${HOST_PYTHON}" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
PY
}

compact_utc_now() {
  "${HOST_PYTHON}" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
PY
}

atomic_write_text() {
  local path="$1"
  local value="$2"
  local directory temporary
  directory="$(dirname "${path}")"
  mkdir -p "${directory}"
  temporary="${directory}/.$(basename "${path}").$$.tmp"
  printf '%s\n' "${value}" >"${temporary}"
  chmod 600 "${temporary}"
  mv -f "${temporary}" "${path}"
}

hash_path() {
  "${HOST_PYTHON}" - "$1" <<'PY'
import hashlib
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"missing path: {root}")

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()

if root.is_file():
    print(file_hash(root).hex())
    raise SystemExit(0)

digest = hashlib.sha256()
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
    relative = path.relative_to(root).as_posix().encode("utf-8")
    if path.is_symlink():
        digest.update(b"L\0" + relative + b"\0")
        digest.update(os.readlink(path).encode("utf-8") + b"\0")
    elif path.is_file():
        digest.update(b"F\0" + relative + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        digest.update(file_hash(path))
print(digest.hexdigest())
PY
}

hash_kind() {
  if [[ -d "$1" ]]; then
    printf 'sha256-tree-v1\n'
  else
    printf 'sha256\n'
  fi
}

json_argv() {
  "${HOST_PYTHON}" - "$@" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
}

require_command() {
  local command="$1"
  if [[ "${command}" == */* ]]; then
    [[ -x "${command}" ]] || die "required executable is missing: ${command}"
  else
    command -v "${command}" >/dev/null 2>&1 || die "required command is missing: ${command}"
  fi
}

require_path() {
  local label="$1"
  local path="$2"
  [[ -e "${path}" ]] || die "${label} is missing: ${path}"
}

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || die "${label} file is missing: ${path}"
}

require_directory() {
  local label="$1"
  local path="$2"
  [[ -d "${path}" ]] || die "${label} directory is missing: ${path}"
}

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "required configuration is empty: ${name}"
}

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

verify_pin() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ -z "${expected}" ]]; then
    if [[ "${ALAYA_REQUIRE_PINNED_WEIGHTS}" == "1" ]]; then
      die "${label} expected SHA256 is required"
    fi
    warn "${label} is not pinned; actual SHA256=${actual}"
    return
  fi
  [[ "$(lowercase "${actual}")" == "$(lowercase "${expected}")" ]] || \
    die "${label} SHA256 mismatch: expected ${expected}, got ${actual}"
}

load_deploy_config() {
  ALAYA_PIPELINE_DIR="${ALAYA_PIPELINE_DIR:-${SCRIPT_DIR}}"
  ALAYA_BRIDGE_LAUNCHER="${ALAYA_BRIDGE_LAUNCHER:-${ALAYA_PIPELINE_DIR}/run_alaya_live_bridge.sh}"
  ALAYA_WORKER_SCRIPT="${ALAYA_WORKER_SCRIPT:-${ALAYA_PIPELINE_DIR}/../worker.py}"
  ALAYA_PREVIEW_LAUNCHER="${ALAYA_PREVIEW_LAUNCHER:-${ALAYA_PIPELINE_DIR}/run_alaya_live_preview.sh}"
  ALAYA_STK_LAUNCHER="${ALAYA_STK_LAUNCHER:-}"
  ALAYA_STK_BINARY="${ALAYA_STK_BINARY:-}"
  ALAYA_STK_BINARY_SHA256="${ALAYA_STK_BINARY_SHA256:-}"
  ALAYA_STK_HOME="${ALAYA_STK_HOME:-}"
  ALAYA_STK_HOME_TEMPLATE="${ALAYA_STK_HOME_TEMPLATE:-${LAUNCHER_PROJECT_DIR}/stk_home_template}"
  ALAYA_STK_GFX_PRESET="${ALAYA_STK_GFX_PRESET:-3}"
  ALAYA_STK_HANDOVER_GRACE="${ALAYA_STK_HANDOVER_GRACE:-5}"

  ALAYA_PYTHON_BIN="${ALAYA_PYTHON_BIN:-python3}"
  ALAYA_PREVIEW_PYTHON="${ALAYA_PREVIEW_PYTHON:-${ALAYA_PYTHON_BIN}}"
  ALAYA_FORWARD_RENDERER_ROOT="${ALAYA_FORWARD_RENDERER_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
  ALAYA_DEMO_ROOT="${ALAYA_DEMO_ROOT:-${SCRIPT_DIR}}"
  ALAYA_TAEHV_ROOT="${ALAYA_TAEHV_ROOT:-$(cd "${SCRIPT_DIR}/../../decoders" && pwd)}"
  ALAYA_TMUX_BIN="${ALAYA_TMUX_BIN:-tmux}"
  ALAYA_CURL_BIN="${ALAYA_CURL_BIN:-curl}"
  ALAYA_NVIDIA_SMI_BIN="${ALAYA_NVIDIA_SMI_BIN:-nvidia-smi}"

  ALAYA_BASE_MODEL="${ALAYA_BASE_MODEL:-}"
  ALAYA_MODEL_CHECKPOINT="${ALAYA_MODEL_CHECKPOINT:-}"
  ALAYA_GROUPED_ENCODER="${ALAYA_GROUPED_ENCODER:-${ALAYA_MODEL_CHECKPOINT:-}/gbuf_small_encoder.safetensors}"
  ALAYA_TINY_DECODER="${ALAYA_TINY_DECODER:-${ALAYA_MODEL_CHECKPOINT:-}/tiny_decoder/taew2_1_ft.safetensors}"
  ALAYA_CAUSAL_CONTRACT_FILE="${ALAYA_CAUSAL_CONTRACT_FILE:-$(cd "${SCRIPT_DIR}/../.." && pwd)/gbuffer.py}"
  ALAYA_BASE_MODEL_SHA256="${ALAYA_BASE_MODEL_SHA256:-}"
  ALAYA_MODEL_CHECKPOINT_SHA256="${ALAYA_MODEL_CHECKPOINT_SHA256:-}"
  ALAYA_GROUPED_ENCODER_SHA256="${ALAYA_GROUPED_ENCODER_SHA256:-}"
  ALAYA_TINY_DECODER_SHA256="${ALAYA_TINY_DECODER_SHA256:-}"
  ALAYA_CAUSAL_CONTRACT_SHA256="${ALAYA_CAUSAL_CONTRACT_SHA256:-}"
  ALAYA_REQUIRE_PINNED_WEIGHTS="${ALAYA_REQUIRE_PINNED_WEIGHTS:-1}"
  ALAYA_MODEL_PROFILE="${ALAYA_MODEL_PROFILE:-stk-checkpoint-600-grouped}"
  ALAYA_EXPECTED_CHECKPOINT_NAME="${ALAYA_EXPECTED_CHECKPOINT_NAME:-}"
  ALAYA_CAUSAL_CONTRACT_ID="${ALAYA_CAUSAL_CONTRACT_ID:-stk-finetune-gbuffer-pixel-window-v1}"

  ALAYA_TRACK="${ALAYA_TRACK:-black_forest}"
  ALAYA_PROMPT="${ALAYA_PROMPT:-cinematic realistic racing game}"
  ALAYA_SOURCE_WIDTH="${ALAYA_SOURCE_WIDTH:-960}"
  ALAYA_SOURCE_HEIGHT="${ALAYA_SOURCE_HEIGHT:-540}"
  ALAYA_MODEL_WIDTH="${ALAYA_MODEL_WIDTH:-832}"
  ALAYA_MODEL_HEIGHT="${ALAYA_MODEL_HEIGHT:-448}"
  ALAYA_PREVIEW_WIDTH="${ALAYA_PREVIEW_WIDTH:-720}"
  ALAYA_PREVIEW_GBUFFER_WIDTH="${ALAYA_PREVIEW_GBUFFER_WIDTH:-360}"
  ALAYA_NUM_INFERENCE_STEPS="${ALAYA_NUM_INFERENCE_STEPS:-4}"
  ALAYA_LATENT_WINDOW="${ALAYA_LATENT_WINDOW:-4}"
  ALAYA_INITIAL_PIXEL_FRAMES="${ALAYA_INITIAL_PIXEL_FRAMES:-13}"
  ALAYA_RESUMED_PIXEL_FRAMES="${ALAYA_RESUMED_PIXEL_FRAMES:-16}"
  ALAYA_SMALL_ENCODER_MODE="${ALAYA_SMALL_ENCODER_MODE:-grouped}"
  ALAYA_SMALL_ENCODER_DIM="${ALAYA_SMALL_ENCODER_DIM:-128}"
  ALAYA_GROUPED_ENCODER_BRANCHES="${ALAYA_GROUPED_ENCODER_BRANCHES:-5}"
  ALAYA_WINDOW_TRANSPORT="${ALAYA_WINDOW_TRANSPORT:-shm_slots}"
  ALAYA_TARGET_FPS="${ALAYA_TARGET_FPS:-30.0}"
  ALAYA_PUBLISH_FPS="${ALAYA_PUBLISH_FPS:-32.0}"
  ALAYA_PUBLISH_SOURCE_FPS="${ALAYA_PUBLISH_SOURCE_FPS:-30.0}"
  ALAYA_PUBLISH_QUEUE_FRAMES="${ALAYA_PUBLISH_QUEUE_FRAMES:-48}"
  ALAYA_PUBLISH_BUFFER_FRAMES="${ALAYA_PUBLISH_BUFFER_FRAMES:-2.5}"
  ALAYA_PUBLISH_MIN_FPS="${ALAYA_PUBLISH_MIN_FPS:-12.0}"
  ALAYA_SMOOTH_PLAYER_INPUT="${ALAYA_SMOOTH_PLAYER_INPUT:-0}"
  ALAYA_MAX_WORKER_LAG_WINDOWS="${ALAYA_MAX_WORKER_LAG_WINDOWS:-2}"
  ALAYA_COMPILE_MODE="${ALAYA_COMPILE_MODE:-all}"
  ALAYA_WARMUP_CHUNKS="${ALAYA_WARMUP_CHUNKS:-2}"

  ALAYA_STK_EGL_DEVICE="${ALAYA_STK_EGL_DEVICE:-0}"
  ALAYA_WORKER_GPU="${ALAYA_WORKER_GPU:-3}"
  ALAYA_PREVIEW_HOST="${ALAYA_PREVIEW_HOST:-127.0.0.1}"
  ALAYA_PREVIEW_PORT="${ALAYA_PREVIEW_PORT:-8095}"
  ALAYA_PUBLIC_PORT="${ALAYA_PUBLIC_PORT:-${ALAYA_PREVIEW_PORT}}"
  ALAYA_PREVIEW_MAX_FPS="${ALAYA_PREVIEW_MAX_FPS:-36}"
  ALAYA_PREVIEW_ALLOWED_ORIGINS="${ALAYA_PREVIEW_ALLOWED_ORIGINS:-}"
  ALAYA_PREVIEW_ADMIN_TOKEN_FILE="${ALAYA_PREVIEW_ADMIN_TOKEN_FILE:-}"

  ALAYA_RUNTIME_BASE="${ALAYA_RUNTIME_BASE:-${XDG_RUNTIME_DIR:-/tmp}/alaya-stk-checkpoint-service}"
  ALAYA_READY_TIMEOUT_STK="${ALAYA_READY_TIMEOUT_STK:-60}"
  ALAYA_READY_TIMEOUT_BRIDGE="${ALAYA_READY_TIMEOUT_BRIDGE:-60}"
  ALAYA_READY_TIMEOUT_WORKER="${ALAYA_READY_TIMEOUT_WORKER:-900}"
  ALAYA_READY_TIMEOUT_PREVIEW="${ALAYA_READY_TIMEOUT_PREVIEW:-60}"
  ALAYA_STOP_TIMEOUT="${ALAYA_STOP_TIMEOUT:-15}"
  ALAYA_RESTART_DELAY="${ALAYA_RESTART_DELAY:-2}"
  ALAYA_HEALTH_STALE_AFTER="${ALAYA_HEALTH_STALE_AFTER:-5}"
  ALAYA_DOCTOR_SKIP_GPU="${ALAYA_DOCTOR_SKIP_GPU:-0}"
  ALAYA_DOCTOR_SKIP_PORT="${ALAYA_DOCTOR_SKIP_PORT:-0}"
}

prepare_runtime_identity() {
  if [[ -z "${ALAYA_RUNTIME_ID:-}" ]]; then
    ALAYA_RUNTIME_ID="stk-checkpoint-$(compact_utc_now)-$$"
  fi
  [[ "${ALAYA_RUNTIME_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die 'ALAYA_RUNTIME_ID has unsafe characters'
  ALAYA_RUNTIME_ROOT="${ALAYA_RUNTIME_ROOT:-${ALAYA_RUNTIME_BASE}/${ALAYA_RUNTIME_ID}}"
  ALAYA_RUNTIME_ROOT="$(canonical_path "${ALAYA_RUNTIME_ROOT}")"
  ALAYA_STK_HOME="${ALAYA_STK_HOME:-${ALAYA_RUNTIME_ROOT}/stk_home}"
  ALAYA_STK_HOME="$(canonical_path "${ALAYA_STK_HOME}")"
  ALAYA_TMUX_SESSION="${ALAYA_TMUX_SESSION:-alaya-stk-${ALAYA_RUNTIME_ID}}"
  ALAYA_TMUX_SESSION="$(printf '%s' "${ALAYA_TMUX_SESSION}" | tr -cd 'A-Za-z0-9._-')"
  [[ -n "${ALAYA_TMUX_SESSION}" ]] || die 'tmux session name is empty'

  CAPTURE_DIR="${ALAYA_RUNTIME_ROOT}/capture"
  CONTROL_DIR="${ALAYA_RUNTIME_ROOT}/control"
  WINDOW_DIR="${ALAYA_RUNTIME_ROOT}/windows"
  RENDERED_DIR="${ALAYA_RUNTIME_ROOT}/rendered"
  INPUT_STATE_PATH="${ALAYA_RUNTIME_ROOT}/input/input.json"
  # Logs grow without bound, so keep them on disk rather than in the size-capped /dev/shm.
  LOG_DIR="${ALAYA_LOG_DIR:-${ALAYA_STATE_DIR}/logs/${ALAYA_RUNTIME_ID}}"
  ALAYA_LOG_MAX_BYTES="${ALAYA_LOG_MAX_BYTES:-52428800}"
  ALAYA_LOG_BACKUPS="${ALAYA_LOG_BACKUPS:-3}"
  PID_DIR="${ALAYA_RUNTIME_ROOT}/pids"
  RUN_DIR="${ALAYA_RUNTIME_ROOT}/run"
  CACHE_ROOT="${ALAYA_CACHE_ROOT:-${ALAYA_RUNTIME_ROOT}/cache}"
  TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
  TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
  # HF_HOME may be pinned separately (ALAYA_HF_HOME) so a read-only bundle can
  # serve pre-seeded kernels while the writable torch/triton cache lives in
  # CACHE_ROOT (which can point outside the mount).
  HF_HOME="${ALAYA_HF_HOME:-${CACHE_ROOT}/huggingface}"
  MANIFEST_PATH="${ALAYA_RUNTIME_ROOT}/deployment-manifest.json"
  SOURCE_SESSION_ID="${ALAYA_SOURCE_SESSION_ID:-${ALAYA_TRACK}-${ALAYA_RUNTIME_ID}}"
  PREVIEW_STATE_URL="http://${ALAYA_PREVIEW_HOST}:${ALAYA_PREVIEW_PORT}/state"
  PREVIEW_MODEL_URL="http://${ALAYA_PREVIEW_HOST}:${ALAYA_PREVIEW_PORT}/model.jpg"
}

check_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

check_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

check_positive_number() {
  local name="$1"
  local value="$2"
  "${HOST_PYTHON}" - "${name}" "${value}" <<'PY' || die "${name} must be a positive finite number: ${value}"
import math
import sys
try:
    value = float(sys.argv[2])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
}

check_port() {
  local name="$1"
  local value="$2"
  check_positive_integer "${name}" "${value}"
  (( value <= 65535 )) || die "${name} must be <= 65535"
}

port_is_free() {
  "${HOST_PYTHON}" - "${ALAYA_PREVIEW_HOST}" "${ALAYA_PREVIEW_PORT}" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
finally:
    sock.close()
PY
}

gpu_exists() {
  local gpu="$1"
  "${ALAYA_NVIDIA_SMI_BIN}" --query-gpu=index --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -Fx "${gpu}" >/dev/null
}

doctor_impl() {
  local quiet="${1:-0}"
  DOCTOR_WARNINGS=""
  load_deploy_config
  prepare_runtime_identity

  require_command "${HOST_PYTHON}"
  require_command "${ALAYA_PYTHON_BIN}"
  require_command "${ALAYA_PREVIEW_PYTHON}"
  require_command "${ALAYA_TMUX_BIN}"
  require_command "${ALAYA_CURL_BIN}"
  if [[ "${ALAYA_DOCTOR_SKIP_GPU}" != "1" ]]; then
    require_command "${ALAYA_NVIDIA_SMI_BIN}"
  fi

  require_variable ALAYA_STK_LAUNCHER
  require_variable ALAYA_STK_BINARY
  require_variable ALAYA_BASE_MODEL
  require_variable ALAYA_MODEL_CHECKPOINT
  require_variable ALAYA_GROUPED_ENCODER
  require_variable ALAYA_TINY_DECODER
  require_variable ALAYA_CAUSAL_CONTRACT_FILE
  require_variable ALAYA_CAUSAL_CONTRACT_ID
  require_variable ALAYA_FORWARD_RENDERER_ROOT
  require_variable ALAYA_DEMO_ROOT
  require_variable ALAYA_TAEHV_ROOT

  require_file 'STK launcher' "${ALAYA_STK_LAUNCHER}"
  [[ -x "${ALAYA_STK_BINARY}" ]] || die "STK binary is not executable: ${ALAYA_STK_BINARY}"
  require_file 'bridge launcher' "${ALAYA_BRIDGE_LAUNCHER}"
  require_file 'worker script' "${ALAYA_WORKER_SCRIPT}"
  require_file 'preview launcher' "${ALAYA_PREVIEW_LAUNCHER}"
  require_directory 'base model' "${ALAYA_BASE_MODEL}"
  require_directory 'model checkpoint' "${ALAYA_MODEL_CHECKPOINT}"
  [[ -f "${ALAYA_MODEL_CHECKPOINT}/transformer_partial.safetensors" || -f "${ALAYA_MODEL_CHECKPOINT}/transformer_partial.pth" ]] || \
    die "missing transformer_partial.{safetensors,pth} in ${ALAYA_MODEL_CHECKPOINT}"
  require_file 'grouped encoder' "${ALAYA_GROUPED_ENCODER}"
  require_file 'tiny decoder' "${ALAYA_TINY_DECODER}"
  require_file 'causal contract' "${ALAYA_CAUSAL_CONTRACT_FILE}"
  require_directory 'forward renderer source' "${ALAYA_FORWARD_RENDERER_ROOT}"
  require_directory 'Alaya demo helpers' "${ALAYA_DEMO_ROOT}"
  require_directory 'TAEHV source' "${ALAYA_TAEHV_ROOT}"
  require_directory 'STK config template' "${ALAYA_STK_HOME_TEMPLATE}/.config/supertuxkart"
  require_directory 'STK data template' "${ALAYA_STK_HOME_TEMPLATE}/.local/share/supertuxkart"

  case "${ALAYA_STK_HOME}" in
    "${ALAYA_RUNTIME_ROOT}"/*) ;;
    *) die "STK home must be inside the independent runtime root: ${ALAYA_STK_HOME}" ;;
  esac

  if [[ -n "${ALAYA_EXPECTED_CHECKPOINT_NAME}" ]]; then
    [[ "$(basename "${ALAYA_MODEL_CHECKPOINT}")" == "${ALAYA_EXPECTED_CHECKPOINT_NAME}" ]] || \
      die "checkpoint basename must be ${ALAYA_EXPECTED_CHECKPOINT_NAME}: ${ALAYA_MODEL_CHECKPOINT}"
  fi
  grep -q -- '--small-encoder-mode' "${ALAYA_WORKER_SCRIPT}" || die 'worker does not expose grouped encoder selection'
  grep -q -- '--small-encoder' "${ALAYA_WORKER_SCRIPT}" || die 'worker does not expose grouped encoder weights'
  grep -q -- '--tiny-decoder' "${ALAYA_WORKER_SCRIPT}" || die 'worker does not expose tiny decoder selection'
  grep -q -- '--checkpoint' "${ALAYA_WORKER_SCRIPT}" || die 'worker does not expose checkpoint selection'
  grep -q -- '--compile-mode' "${ALAYA_WORKER_SCRIPT}" || die 'worker does not expose compile mode selection'
  "${ALAYA_PYTHON_BIN}" -c 'from alayarenderer_flash.runtime import stk_causal_window_spec' || \
    die 'alayarenderer_flash is not importable from ALAYA_PYTHON_BIN'
  "${ALAYA_PYTHON_BIN}" -c 'import cv2' 2>/dev/null || \
    die 'OpenCV is missing; the bridge falls back to a CPU path that caps output near 20 FPS. Run: uv sync --extra demo'

  [[ "${ALAYA_NUM_INFERENCE_STEPS}" == "4" ]] || die 'canonical profile requires exactly 4 denoising steps'
  [[ "${ALAYA_LATENT_WINDOW}" == "4" ]] || die 'canonical profile requires latent_window=4'
  [[ "${ALAYA_INITIAL_PIXEL_FRAMES}" == "13" ]] || die 'canonical profile requires 13 initial pixel frames'
  [[ "${ALAYA_RESUMED_PIXEL_FRAMES}" == "16" ]] || die 'canonical profile requires 16 resumed pixel frames'
  [[ "${ALAYA_SMALL_ENCODER_MODE}" == "grouped" ]] || die 'canonical profile requires grouped encoder mode'
  [[ "${ALAYA_SMALL_ENCODER_DIM}" == "128" ]] || die 'canonical grouped encoder requires dim=128'
  [[ "${ALAYA_GROUPED_ENCODER_BRANCHES}" == "5" ]] || die 'canonical grouped encoder requires five branches'
  [[ "${ALAYA_WINDOW_TRANSPORT}" == "shm_slots" ]] || die 'canonical profile requires shm_slots transport'
  [[ "${ALAYA_STK_GFX_PRESET}" == "3" ]] || die 'canonical profile requires STK gfx preset 3'
  [[ "${ALAYA_COMPILE_MODE}" == "all" ]] || die 'canonical profile requires compile_mode=all'
  [[ "${ALAYA_WARMUP_CHUNKS}" == "2" ]] || die 'canonical profile requires warmup_chunks=2'
  [[ "${ALAYA_SOURCE_WIDTH}" == "960" && "${ALAYA_SOURCE_HEIGHT}" == "540" ]] || \
    die 'canonical profile requires 960x540 STK capture'
  [[ "${ALAYA_MODEL_WIDTH}" == "832" && "${ALAYA_MODEL_HEIGHT}" == "448" ]] || \
    die 'canonical profile requires 832x448 model input'

  check_positive_integer ALAYA_SOURCE_WIDTH "${ALAYA_SOURCE_WIDTH}"
  check_positive_integer ALAYA_SOURCE_HEIGHT "${ALAYA_SOURCE_HEIGHT}"
  check_positive_integer ALAYA_MODEL_WIDTH "${ALAYA_MODEL_WIDTH}"
  check_positive_integer ALAYA_MODEL_HEIGHT "${ALAYA_MODEL_HEIGHT}"
  check_positive_integer ALAYA_PREVIEW_WIDTH "${ALAYA_PREVIEW_WIDTH}"
  check_positive_integer ALAYA_PREVIEW_GBUFFER_WIDTH "${ALAYA_PREVIEW_GBUFFER_WIDTH}"
  check_positive_integer ALAYA_PUBLISH_QUEUE_FRAMES "${ALAYA_PUBLISH_QUEUE_FRAMES}"
  check_positive_integer ALAYA_PREVIEW_MAX_FPS "${ALAYA_PREVIEW_MAX_FPS}"
  check_positive_integer ALAYA_STK_HANDOVER_GRACE "${ALAYA_STK_HANDOVER_GRACE}"
  check_positive_integer ALAYA_HEALTH_STALE_AFTER "${ALAYA_HEALTH_STALE_AFTER}"
  check_positive_integer ALAYA_READY_TIMEOUT_STK "${ALAYA_READY_TIMEOUT_STK}"
  check_positive_integer ALAYA_READY_TIMEOUT_BRIDGE "${ALAYA_READY_TIMEOUT_BRIDGE}"
  check_positive_integer ALAYA_READY_TIMEOUT_WORKER "${ALAYA_READY_TIMEOUT_WORKER}"
  check_positive_integer ALAYA_READY_TIMEOUT_PREVIEW "${ALAYA_READY_TIMEOUT_PREVIEW}"
  check_positive_integer ALAYA_STOP_TIMEOUT "${ALAYA_STOP_TIMEOUT}"
  check_positive_integer ALAYA_RESTART_DELAY "${ALAYA_RESTART_DELAY}"
  check_nonnegative_integer ALAYA_STK_EGL_DEVICE "${ALAYA_STK_EGL_DEVICE}"
  check_nonnegative_integer ALAYA_WORKER_GPU "${ALAYA_WORKER_GPU}"
  check_positive_number ALAYA_TARGET_FPS "${ALAYA_TARGET_FPS}"
  check_positive_number ALAYA_PUBLISH_FPS "${ALAYA_PUBLISH_FPS}"
  check_positive_number ALAYA_PUBLISH_SOURCE_FPS "${ALAYA_PUBLISH_SOURCE_FPS}"
  check_positive_number ALAYA_PUBLISH_BUFFER_FRAMES "${ALAYA_PUBLISH_BUFFER_FRAMES}"
  check_positive_number ALAYA_PUBLISH_MIN_FPS "${ALAYA_PUBLISH_MIN_FPS}"
  check_port ALAYA_PREVIEW_PORT "${ALAYA_PREVIEW_PORT}"
  check_port ALAYA_PUBLIC_PORT "${ALAYA_PUBLIC_PORT}"
  [[ "${ALAYA_REQUIRE_PINNED_WEIGHTS}" == "0" || "${ALAYA_REQUIRE_PINNED_WEIGHTS}" == "1" ]] || \
    die 'ALAYA_REQUIRE_PINNED_WEIGHTS must be 0 or 1'

  if [[ "${ALAYA_DOCTOR_SKIP_GPU}" != "1" ]]; then
    gpu_exists "${ALAYA_STK_EGL_DEVICE}" || die "STK EGL GPU does not exist: ${ALAYA_STK_EGL_DEVICE}"
    gpu_exists "${ALAYA_WORKER_GPU}" || die "Worker GPU does not exist: ${ALAYA_WORKER_GPU}"
  fi
  if [[ "${ALAYA_DOCTOR_SKIP_PORT}" != "1" ]]; then
    port_is_free || die "Preview port is already in use: ${ALAYA_PREVIEW_HOST}:${ALAYA_PREVIEW_PORT}"
  fi

  BASE_MODEL_ACTUAL_SHA256="$(hash_path "${ALAYA_BASE_MODEL}")"
  MODEL_CHECKPOINT_ACTUAL_SHA256="$(hash_path "${ALAYA_MODEL_CHECKPOINT}")"
  GROUPED_ENCODER_ACTUAL_SHA256="$(hash_path "${ALAYA_GROUPED_ENCODER}")"
  TINY_DECODER_ACTUAL_SHA256="$(hash_path "${ALAYA_TINY_DECODER}")"
  CAUSAL_CONTRACT_ACTUAL_SHA256="$(hash_path "${ALAYA_CAUSAL_CONTRACT_FILE}")"
  STK_BINARY_ACTUAL_SHA256="$(hash_path "${ALAYA_STK_BINARY}")"
  STK_LAUNCHER_ACTUAL_SHA256="$(hash_path "${ALAYA_STK_LAUNCHER}")"
  LAUNCHER_ACTUAL_SHA256="$(hash_path "${SCRIPT_PATH}")"

  verify_pin 'base model' "${BASE_MODEL_ACTUAL_SHA256}" "${ALAYA_BASE_MODEL_SHA256}"
  verify_pin 'model checkpoint' "${MODEL_CHECKPOINT_ACTUAL_SHA256}" "${ALAYA_MODEL_CHECKPOINT_SHA256}"
  verify_pin 'grouped encoder' "${GROUPED_ENCODER_ACTUAL_SHA256}" "${ALAYA_GROUPED_ENCODER_SHA256}"
  verify_pin 'tiny decoder' "${TINY_DECODER_ACTUAL_SHA256}" "${ALAYA_TINY_DECODER_SHA256}"
  verify_pin 'causal contract' "${CAUSAL_CONTRACT_ACTUAL_SHA256}" "${ALAYA_CAUSAL_CONTRACT_SHA256}"
  verify_pin 'STK binary' "${STK_BINARY_ACTUAL_SHA256}" "${ALAYA_STK_BINARY_SHA256}"

  export BASE_MODEL_ACTUAL_SHA256 MODEL_CHECKPOINT_ACTUAL_SHA256
  export GROUPED_ENCODER_ACTUAL_SHA256 TINY_DECODER_ACTUAL_SHA256
  export CAUSAL_CONTRACT_ACTUAL_SHA256 LAUNCHER_ACTUAL_SHA256
  export STK_BINARY_ACTUAL_SHA256 STK_LAUNCHER_ACTUAL_SHA256

  if [[ "${quiet}" != "1" ]]; then
    if [[ "${OUTPUT_JSON}" == "1" ]]; then
      export_deploy_environment
      export DOCTOR_WARNINGS
      "${HOST_PYTHON}" - <<'PY'
import json
import os
payload = {
    "schema_version": 1,
    "ok": True,
    "model_profile": os.environ["ALAYA_MODEL_PROFILE"],
    "runtime_root_plan": os.environ["ALAYA_RUNTIME_ROOT"],
    "tmux_session_plan": os.environ["ALAYA_TMUX_SESSION"],
    "weights": {
        "base_model": os.environ["BASE_MODEL_ACTUAL_SHA256"],
        "checkpoint": os.environ["MODEL_CHECKPOINT_ACTUAL_SHA256"],
        "grouped_encoder": os.environ["GROUPED_ENCODER_ACTUAL_SHA256"],
        "tiny_decoder": os.environ["TINY_DECODER_ACTUAL_SHA256"],
        "causal_contract": os.environ["CAUSAL_CONTRACT_ACTUAL_SHA256"],
        "stk_binary": os.environ["STK_BINARY_ACTUAL_SHA256"],
    },
    "gpu_mapping": {
        "stk_egl_device": int(os.environ["ALAYA_STK_EGL_DEVICE"]),
        "worker_cuda_visible_device": int(os.environ["ALAYA_WORKER_GPU"]),
        "worker_device_inside_visibility": "cuda:0",
    },
    "dimensions": {
        "source": [int(os.environ["ALAYA_SOURCE_WIDTH"]), int(os.environ["ALAYA_SOURCE_HEIGHT"])],
        "model": [int(os.environ["ALAYA_MODEL_WIDTH"]), int(os.environ["ALAYA_MODEL_HEIGHT"])],
    },
    "denoising_steps": int(os.environ["ALAYA_NUM_INFERENCE_STEPS"]),
    "compile_mode": os.environ["ALAYA_COMPILE_MODE"],
    "warnings": [line for line in os.environ.get("DOCTOR_WARNINGS", "").splitlines() if line],
}
print(json.dumps(payload, sort_keys=True))
PY
    else
      printf 'doctor: OK\n'
      printf 'profile=%s checkpoint=%s steps=%s encoder=%s tiny_decoder=%s\n' \
        "${ALAYA_MODEL_PROFILE}" "${ALAYA_MODEL_CHECKPOINT}" \
        "${ALAYA_NUM_INFERENCE_STEPS}" "${ALAYA_GROUPED_ENCODER}" \
        "${ALAYA_TINY_DECODER}"
      printf 'runtime_plan=%s tmux_session=%s preview=%s\n' \
        "${ALAYA_RUNTIME_ROOT}" "${ALAYA_TMUX_SESSION}" "${PREVIEW_STATE_URL}"
    fi
  fi
}

export_deploy_environment() {
  export LAUNCHER_SCHEMA_VERSION SCRIPT_PATH CONFIG_FILE
  export ALAYA_PIPELINE_DIR ALAYA_BRIDGE_LAUNCHER ALAYA_WORKER_SCRIPT
  export ALAYA_PREVIEW_LAUNCHER ALAYA_STK_LAUNCHER ALAYA_STK_BINARY
  export ALAYA_STK_BINARY_SHA256 ALAYA_STK_HOME ALAYA_STK_HOME_TEMPLATE
  export ALAYA_STK_GFX_PRESET ALAYA_STK_HANDOVER_GRACE
  export ALAYA_PYTHON_BIN ALAYA_PREVIEW_PYTHON ALAYA_TMUX_BIN ALAYA_CURL_BIN
  export ALAYA_FORWARD_RENDERER_ROOT ALAYA_DEMO_ROOT ALAYA_TAEHV_ROOT
  export ALAYA_MODEL_PROFILE ALAYA_EXPECTED_CHECKPOINT_NAME
  export ALAYA_BASE_MODEL ALAYA_MODEL_CHECKPOINT ALAYA_GROUPED_ENCODER ALAYA_TINY_DECODER
  export ALAYA_CAUSAL_CONTRACT_FILE ALAYA_CAUSAL_CONTRACT_ID
  export ALAYA_BASE_MODEL_SHA256 ALAYA_MODEL_CHECKPOINT_SHA256
  export ALAYA_GROUPED_ENCODER_SHA256 ALAYA_TINY_DECODER_SHA256 ALAYA_CAUSAL_CONTRACT_SHA256
  export ALAYA_TRACK ALAYA_PROMPT ALAYA_SOURCE_WIDTH ALAYA_SOURCE_HEIGHT
  export ALAYA_MODEL_WIDTH ALAYA_MODEL_HEIGHT ALAYA_PREVIEW_WIDTH ALAYA_PREVIEW_GBUFFER_WIDTH
  export ALAYA_NUM_INFERENCE_STEPS ALAYA_LATENT_WINDOW ALAYA_INITIAL_PIXEL_FRAMES
  export ALAYA_RESUMED_PIXEL_FRAMES ALAYA_SMALL_ENCODER_MODE ALAYA_SMALL_ENCODER_DIM
  export ALAYA_GROUPED_ENCODER_BRANCHES ALAYA_WINDOW_TRANSPORT ALAYA_TARGET_FPS
  export ALAYA_PUBLISH_FPS ALAYA_PUBLISH_SOURCE_FPS ALAYA_PUBLISH_QUEUE_FRAMES
  export ALAYA_PUBLISH_BUFFER_FRAMES ALAYA_PUBLISH_MIN_FPS ALAYA_COMPILE_MODE
  export ALAYA_WARMUP_CHUNKS ALAYA_STK_EGL_DEVICE ALAYA_WORKER_GPU
  export ALAYA_HEALTH_STALE_AFTER
  export ALAYA_PREVIEW_HOST ALAYA_PREVIEW_PORT ALAYA_PUBLIC_PORT ALAYA_PREVIEW_MAX_FPS
  export ALAYA_PREVIEW_ALLOWED_ORIGINS ALAYA_PREVIEW_ADMIN_TOKEN_FILE
  export ALAYA_RUNTIME_ID ALAYA_RUNTIME_ROOT ALAYA_TMUX_SESSION SOURCE_SESSION_ID
  export CAPTURE_DIR CONTROL_DIR WINDOW_DIR RENDERED_DIR INPUT_STATE_PATH
  export LOG_DIR PID_DIR RUN_DIR CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR HF_HOME
  export MANIFEST_PATH PREVIEW_STATE_URL PREVIEW_MODEL_URL
  export BASE_MODEL_ACTUAL_SHA256 MODEL_CHECKPOINT_ACTUAL_SHA256
  export GROUPED_ENCODER_ACTUAL_SHA256 TINY_DECODER_ACTUAL_SHA256
  export CAUSAL_CONTRACT_ACTUAL_SHA256 STK_BINARY_ACTUAL_SHA256
  export STK_LAUNCHER_ACTUAL_SHA256 LAUNCHER_ACTUAL_SHA256
}

configure_commands() {
  local stk_binary_env=()
  if [[ -n "${ALAYA_STK_BINARY}" ]]; then
    stk_binary_env=("STK_BINARY=${ALAYA_STK_BINARY}")
  fi
  STK_COMMAND=(
    env
    "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}"
    "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}"
    "OUT_DIR=${CAPTURE_DIR}"
    "INPUT_STATE_PATH=${INPUT_STATE_PATH}"
    "STK_HOME=${ALAYA_STK_HOME}"
    "STK_HOME_TEMPLATE=${ALAYA_STK_HOME_TEMPLATE}"
    "XDG_RUNTIME_DIR=${ALAYA_RUNTIME_ROOT}/xdg"
    "SOURCE_SESSION_ID=${SOURCE_SESSION_ID}"
    "TRACK=${ALAYA_TRACK}"
    "SCREENSIZE=${ALAYA_SOURCE_WIDTH}x${ALAYA_SOURCE_HEIGHT}"
    "GFX_PRESET=${ALAYA_STK_GFX_PRESET}"
    "SDL_HINT_EGL_DEVICE=${ALAYA_STK_EGL_DEVICE}"
    "${stk_binary_env[@]}"
    bash "${ALAYA_STK_LAUNCHER}"
  )
  BRIDGE_COMMAND=(
    env
    "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}"
    "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}"
    "PROJECT_DIR=${ALAYA_PIPELINE_DIR}"
    "PYTHON_BIN=${ALAYA_PYTHON_BIN}"
    "CAPTURE_DIR=${CAPTURE_DIR}"
    "CONTROL_DIR=${CONTROL_DIR}"
    "WINDOW_DIR=${WINDOW_DIR}"
    "INPUT_STATE_PATH=${INPUT_STATE_PATH}"
    "EXPECTED_WIDTH=${ALAYA_SOURCE_WIDTH}"
    "EXPECTED_HEIGHT=${ALAYA_SOURCE_HEIGHT}"
    "ALAYA_30FPS_SHADOW=1"
    "LATENT_WINDOW=4"
    "TARGET_FPS=${ALAYA_TARGET_FPS}"
    "WINDOW_TRANSPORT=shm_slots"
    "MAX_WORKER_LAG_WINDOWS=${ALAYA_MAX_WORKER_LAG_WINDOWS}"
    bash "${ALAYA_BRIDGE_LAUNCHER}"
  )
  WORKER_COMMAND=(
    env
    "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}"
    "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}"
    "CUDA_VISIBLE_DEVICES=${ALAYA_WORKER_GPU}"
    "HF_HOME=${HF_HOME}"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "PYTHONNOUSERSITE=1"
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    "ALAYA_FORWARD_RENDERER_ROOT=${ALAYA_FORWARD_RENDERER_ROOT}"
    "ALAYA_DEMO_ROOT=${ALAYA_DEMO_ROOT}"
    "ALAYA_TAEHV_ROOT=${ALAYA_TAEHV_ROOT}"
    "ALAYA_ADAPTIVE_PUBLISH=1"
    "${ALAYA_PYTHON_BIN}" "${ALAYA_WORKER_SCRIPT}"
    --control-dir "${CONTROL_DIR}"
    --input-dir "${WINDOW_DIR}"
    --rendered-dir "${RENDERED_DIR}"
    --base-model "${ALAYA_BASE_MODEL}"
    --checkpoint "${ALAYA_MODEL_CHECKPOINT}"
    --small-encoder "${ALAYA_GROUPED_ENCODER}"
    --tiny-decoder "${ALAYA_TINY_DECODER}"
    --device cuda:0
    --height "${ALAYA_MODEL_HEIGHT}"
    --width "${ALAYA_MODEL_WIDTH}"
    --latent-window 4
    --num-inference-steps 4
    --window-transport shm_slots
    --small-encoder-mode grouped
    --small-encoder-dim 128
    --compile-mode "${ALAYA_COMPILE_MODE}"
    --warmup-chunks "${ALAYA_WARMUP_CHUNKS}"
    --max-attempts 3
    --poll-interval 0.01
    --publish-fps "${ALAYA_PUBLISH_FPS}"
    --publish-queue-frames "${ALAYA_PUBLISH_QUEUE_FRAMES}"
    --adaptive-publish
    --publish-source-fps "${ALAYA_PUBLISH_SOURCE_FPS}"
    --publish-buffer-frames "${ALAYA_PUBLISH_BUFFER_FRAMES}"
    --publish-min-fps "${ALAYA_PUBLISH_MIN_FPS}"
  )
  if [[ "${ALAYA_SMOOTH_PLAYER_INPUT}" != "1" ]]; then
    WORKER_COMMAND+=(--input-state "${INPUT_STATE_PATH}")
  fi
  PREVIEW_COMMAND=(
    env
    "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}"
    "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}"
    "PYTHON_BIN=${ALAYA_PREVIEW_PYTHON}"
    "CAPTURE_DIR=${CAPTURE_DIR}"
    "CONTROL_DIR=${CONTROL_DIR}"
    "WINDOW_DIR=${WINDOW_DIR}"
    "RENDERED_DIR=${RENDERED_DIR}"
    "INPUT_STATE_PATH=${INPUT_STATE_PATH}"
    "PREVIEW_HOST=${ALAYA_PREVIEW_HOST}"
    "PREVIEW_PORT=${ALAYA_PREVIEW_PORT}"
    "PREVIEW_MAIN_WIDTH=${ALAYA_PREVIEW_WIDTH}"
    "PREVIEW_GBUFFER_WIDTH=${ALAYA_PREVIEW_GBUFFER_WIDTH}"
    "PREVIEW_MAX_FPS=${ALAYA_PREVIEW_MAX_FPS}"
    "ALAYA_PREVIEW_ALLOWED_ORIGINS=${ALAYA_PREVIEW_ALLOWED_ORIGINS}"
    "ALAYA_PREVIEW_ADMIN_TOKEN_FILE=${ALAYA_PREVIEW_ADMIN_TOKEN_FILE}"
    "READ_ONLY_OBSERVER=0"
    "STK_TRACK_SCRIPT_PATH=${ALAYA_STK_LAUNCHER}"
    "STK_TRACK_PID_PATH=${PID_DIR}/stk.child.pid"
    "STK_TRACK_LOG_PATH=${LOG_DIR}/stk_live_preview_stk.log"
    "STK_HOME=${ALAYA_STK_HOME}"
    "STK_HOME_TEMPLATE=${ALAYA_STK_HOME_TEMPLATE}"
    "GFX_PRESET=${ALAYA_STK_GFX_PRESET}"
    "XDG_RUNTIME_DIR=${ALAYA_RUNTIME_ROOT}/xdg"
    "SDL_HINT_EGL_DEVICE=${ALAYA_STK_EGL_DEVICE}"
    "${stk_binary_env[@]}"
    bash "${ALAYA_PREVIEW_LAUNCHER}"
  )

  STK_COMMAND_JSON="$(json_argv "${STK_COMMAND[@]}")"
  BRIDGE_COMMAND_JSON="$(json_argv "${BRIDGE_COMMAND[@]}")"
  WORKER_COMMAND_JSON="$(json_argv "${WORKER_COMMAND[@]}")"
  PREVIEW_COMMAND_JSON="$(json_argv "${PREVIEW_COMMAND[@]}")"
  export STK_COMMAND_JSON BRIDGE_COMMAND_JSON WORKER_COMMAND_JSON PREVIEW_COMMAND_JSON
}

write_supervisor_runner() {
  local service="$1"
  shift
  local runner="${RUN_DIR}/${service}.sh"
  local temporary="${RUN_DIR}/.${service}.sh.$$.tmp"
  local supervisor_pid="${PID_DIR}/${service}.pid"
  local child_pid="${PID_DIR}/${service}.child.pid"
  local log_path="${LOG_DIR}/${service}.log"
  local liveness_file="" liveness_capture="" liveness_timeout="" liveness_grace="" liveness_poll="" liveness_pyspy=""
  if [[ "${service}" == "worker" ]]; then
    liveness_file="${RENDERED_DIR}/latest.json"
    liveness_capture="${CAPTURE_DIR}/latest.json"
    liveness_timeout="${ALAYA_WORKER_LIVENESS_TIMEOUT:-90}"
    liveness_grace="${ALAYA_WORKER_LIVENESS_GRACE:-360}"
    liveness_poll="${ALAYA_WORKER_LIVENESS_POLL:-15}"
    liveness_pyspy="${ALAYA_PYSPY_BIN:-$(command -v py-spy || true)}"
  fi
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'umask 077\n'
    printf 'export ALAYA_RUNTIME_ID=%q\n' "${ALAYA_RUNTIME_ID}"
    printf 'export ALAYA_RUNTIME_ROOT=%q\n' "${ALAYA_RUNTIME_ROOT}"
    printf 'supervisor_pid_file=%q\n' "${supervisor_pid}"
    printf 'child_pid_file=%q\n' "${child_pid}"
    printf 'log_path=%q\n' "${log_path}"
    printf 'restart_delay=%q\n' "${ALAYA_RESTART_DELAY}"
    printf 'liveness_file=%q\n' "${liveness_file}"
    printf 'liveness_capture=%q\n' "${liveness_capture}"
    printf 'liveness_timeout=%q\n' "${liveness_timeout}"
    printf 'liveness_grace=%q\n' "${liveness_grace}"
    printf 'liveness_poll=%q\n' "${liveness_poll}"
    printf 'liveness_pyspy=%q\n' "${liveness_pyspy}"
    printf 'command=('
    printf '%q ' "$@"
    printf ')\n'
    cat <<'EOF'
mkdir -p "$(dirname "${supervisor_pid_file}")" "$(dirname "${log_path}")"
printf '%s\n' "$$" >"${supervisor_pid_file}.tmp"
mv -f "${supervisor_pid_file}.tmp" "${supervisor_pid_file}"
exec >>"${log_path}" 2>&1
printf '[%s] supervisor started pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$"
child=""
stopping=0
stop_child() {
  stopping=1
  if [[ -n "${child}" ]] && kill -0 "${child}" 2>/dev/null; then
    kill -TERM "${child}" 2>/dev/null || true
    wait "${child}" 2>/dev/null || true
  fi
  rm -f "${child_pid_file}"
  exit 0
}
trap stop_child INT TERM HUP
while true; do
  "${command[@]}" &
  child=$!
  printf '%s\n' "${child}" >"${child_pid_file}.tmp"
  mv -f "${child_pid_file}.tmp" "${child_pid_file}"
  launched_at=$(date +%s)
  if [[ -n "${liveness_file}" ]]; then
    while kill -0 "${child}" 2>/dev/null; do
      sleep "${liveness_poll}"
      [[ "${stopping}" == "1" ]] && break
      kill -0 "${child}" 2>/dev/null || break
      now=$(date +%s)
      (( now - launched_at < liveness_grace )) && continue
      rendered_age=$(( now - $(stat -c %Y "${liveness_file}" 2>/dev/null || echo "${now}") ))
      capture_age=0
      if [[ -n "${liveness_capture}" && -e "${liveness_capture}" ]]; then
        capture_age=$(( now - $(stat -c %Y "${liveness_capture}" 2>/dev/null || echo "${now}") ))
      fi
      if (( rendered_age >= liveness_timeout )) && (( capture_age < liveness_timeout )); then
        printf '[%s] LIVENESS worker hung: rendered_age=%ss capture_age=%ss timeout=%ss; killing child %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rendered_age}" "${capture_age}" "${liveness_timeout}" "${child}"
        dump_file="$(dirname "${log_path}")/worker_hang_$(date -u +%Y%m%dT%H%M%SZ).txt"
        if [[ -n "${liveness_pyspy}" && -x "${liveness_pyspy}" ]]; then
          printf '[%s] capturing worker stack via py-spy -> %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${dump_file}"
          timeout 60 "${liveness_pyspy}" dump --pid "${child}" >"${dump_file}" 2>&1 || printf '(py-spy dump failed rc=%s)\n' "$?" >>"${dump_file}"
        fi
        kill -TERM "${child}" 2>/dev/null || true
        sleep 5
        kill -0 "${child}" 2>/dev/null && kill -KILL "${child}" 2>/dev/null || true
        break
      fi
    done
    wait "${child}" 2>/dev/null
    rc=$?
  else
    wait "${child}"
    rc=$?
  fi
  rm -f "${child_pid_file}"
  child=""
  if [[ "${stopping}" == "1" ]]; then
    exit 0
  fi
  printf '[%s] child exited rc=%s; restarting in %ss\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${restart_delay}"
  sleep "${restart_delay}"
done
EOF
  } >"${temporary}"
  chmod 700 "${temporary}"
  mv -f "${temporary}" "${runner}"
}

write_stk_supervisor_runner() {
  local runner="${RUN_DIR}/stk.sh"
  local temporary="${RUN_DIR}/.stk.sh.$$.tmp"
  local supervisor_pid="${PID_DIR}/stk.pid"
  local child_pid="${PID_DIR}/stk.child.pid"
  local log_path="${LOG_DIR}/stk.log"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'umask 077\n'
    printf 'export ALAYA_RUNTIME_ID=%q\n' "${ALAYA_RUNTIME_ID}"
    printf 'export ALAYA_RUNTIME_ROOT=%q\n' "${ALAYA_RUNTIME_ROOT}"
    printf 'supervisor_pid_file=%q\n' "${supervisor_pid}"
    printf 'child_pid_file=%q\n' "${child_pid}"
    printf 'log_path=%q\n' "${log_path}"
    printf 'restart_delay=%q\n' "${ALAYA_RESTART_DELAY}"
    printf 'handover_grace=%q\n' "${ALAYA_STK_HANDOVER_GRACE}"
    printf 'live_control_path=%q\n' "${CONTROL_DIR}/live_control.json"
    printf 'command=('
    printf '%q ' "$@"
    printf ')\n'
    cat <<'EOF'
mkdir -p "$(dirname "${supervisor_pid_file}")" "$(dirname "${log_path}")"
printf '%s\n' "$$" >"${supervisor_pid_file}.tmp"
mv -f "${supervisor_pid_file}.tmp" "${supervisor_pid_file}"
exec >>"${log_path}" 2>&1
printf '[%s] STK supervisor started pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$"

current_pid=""
owned_child=0
stopping=0

pid_matches_runtime() {
  local pid="$1"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/${pid}/environ" ]] || return 1
  tr '\0' '\n' <"/proc/${pid}/environ" \
    | grep -Fx "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}" >/dev/null
}

pid_alive() {
  local pid="$1"
  local stat_line rest state
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat_line <"/proc/${pid}/stat" || return 1
  rest="${stat_line##*) }"
  state="${rest%% *}"
  [[ "${state}" != "Z" ]]
}

write_child_pid() {
  printf '%s\n' "$1" >"${child_pid_file}.tmp"
  mv -f "${child_pid_file}.tmp" "${child_pid_file}"
}

clear_child_pid_if_current() {
  local expected="$1"
  local recorded=""
  if [[ -r "${child_pid_file}" ]]; then
    IFS= read -r recorded <"${child_pid_file}" || true
  fi
  if [[ "${recorded}" == "${expected}" ]]; then
    rm -f "${child_pid_file}"
  fi
}

stop_current() {
  stopping=1
  trap - INT TERM HUP
  if [[ -n "${current_pid}" ]] && pid_alive "${current_pid}" \
      && pid_matches_runtime "${current_pid}"; then
    kill -TERM "${current_pid}" 2>/dev/null || true
    local deadline=$(( $(date +%s) + 5 ))
    while pid_alive "${current_pid}" && (( $(date +%s) < deadline )); do
      sleep 0.1
    done
    if pid_alive "${current_pid}" && pid_matches_runtime "${current_pid}"; then
      kill -KILL "${current_pid}" 2>/dev/null || true
    fi
  fi
  [[ -n "${current_pid}" ]] && clear_child_pid_if_current "${current_pid}"
  exit 0
}
trap stop_current INT TERM HUP

# After a crash, relaunch the track the viewer is on (from live_control.json)
# rather than the configured default. A different track would leave the bridge
# waiting for a control rebind that only the preview's own switch performs.
active_track() {
  [[ -r "${live_control_path}" ]] || return 0
  grep -o '"track_id": *"[A-Za-z0-9_-]*"' "${live_control_path}" 2>/dev/null \
    | head -n 1 | sed 's/.*"\([A-Za-z0-9_-]*\)"$/\1/'
}

relaunch_command() {
  local track arg
  track="$(active_track)"
  relaunch=()
  for arg in "${command[@]}"; do
    if [[ -n "${track}" && "${arg}" == TRACK=* ]]; then
      arg="TRACK=${track}"
    elif [[ -n "${track}" && "${arg}" == SOURCE_SESSION_ID=* ]]; then
      arg="SOURCE_SESSION_ID=${track}-$(date +%s%N)"
    fi
    relaunch+=("${arg}")
  done
}

launch_owned_child() {
  if [[ "${1:-}" == "relaunch" ]]; then
    relaunch_command
    command=("${relaunch[@]}")
  fi
  "${command[@]}" &
  current_pid=$!
  owned_child=1
  write_child_pid "${current_pid}"
  printf '[%s] launched managed STK pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${current_pid}"
}

adopt_preview_replacement() {
  local exited_pid="$1"
  local deadline=$(( $(date +%s) + handover_grace ))
  local candidate=""
  while (( $(date +%s) <= deadline )); do
    if [[ -r "${child_pid_file}" ]]; then
      IFS= read -r candidate <"${child_pid_file}" || candidate=""
    fi
    if [[ "${candidate}" != "${exited_pid}" ]] \
        && pid_alive "${candidate}" \
        && pid_matches_runtime "${candidate}"; then
      current_pid="${candidate}"
      owned_child=0
      printf '[%s] adopted Preview-launched STK pid=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${current_pid}"
      return 0
    fi
    sleep 0.1
  done
  return 1
}

launch_owned_child
while true; do
  exited_pid="${current_pid}"
  if [[ "${owned_child}" == "1" ]]; then
    wait "${current_pid}"
    rc=$?
  else
    while pid_alive "${current_pid}"; do
      sleep 0.2
    done
    rc=unknown
  fi
  clear_child_pid_if_current "${exited_pid}"
  current_pid=""
  owned_child=0
  [[ "${stopping}" == "0" ]] || exit 0

  if adopt_preview_replacement "${exited_pid}"; then
    continue
  fi

  printf '[%s] managed STK exited rc=%s; no Preview handover appeared in %ss; restarting in %ss\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${handover_grace}" "${restart_delay}"
  sleep "${restart_delay}"
  [[ "${stopping}" == "0" ]] || exit 0
  launch_owned_child relaunch
done
EOF
  } >"${temporary}"
  chmod 700 "${temporary}"
  mv -f "${temporary}" "${runner}"
}

write_initial_documents() {
  export SOURCE_SESSION_ID
  "${HOST_PYTHON}" - <<'PY'
import json
import os
import pathlib
import time

def atomic(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)

identity = {
    "track_id": os.environ["ALAYA_TRACK"],
    "source_session_id": os.environ["SOURCE_SESSION_ID"],
    "reset_epoch": 1,
    "source_reset_epoch": 1,
}
atomic(os.environ["INPUT_STATE_PATH"], {
    "mode": "auto",
    "keys": [],
    "server_time": time.time(),
    "source": "canonical-launcher",
    "client_id": "server:canonical-launcher",
    "intent": "auto",
    "input_seq": 0,
    "input_event_id": f"{os.environ['ALAYA_RUNTIME_ID']}:bootstrap:auto:1",
    "boundary_source_seq": 0,
    "boundary_source_frame": 0,
    **identity,
})
atomic(os.path.join(os.environ["CONTROL_DIR"], "live_control.json"), {
    "schema_version": 2,
    "reset_epoch": 1,
    "reset_reason": "canonical_start",
    "render_request_id": 1,
    "request_reason": "canonical_start",
    "prompt_revision": 1,
    "prompt": os.environ["ALAYA_PROMPT"],
    "track_id": identity["track_id"],
    "source_session_id": identity["source_session_id"],
    "reset_source_seq": 0,
    "reset_source_frame": 0,
    "requested_at": time.time(),
    "render_request_ts": time.time(),
    "reset_at": time.time(),
})
PY
}

rebind_live_control_to_capture() {
  "${HOST_PYTHON}" - <<'PY'
import json
import os
import pathlib
import time

capture = json.loads(pathlib.Path(os.environ["CAPTURE_DIR"], "latest.json").read_text())
path = pathlib.Path(os.environ["CONTROL_DIR"], "live_control.json")
value = json.loads(path.read_text())
seq = int(capture.get("seq", -1))
frame = int(capture.get("source_frame", seq))
value.update({
    "source_session_id": os.environ["SOURCE_SESSION_ID"],
    "track_id": os.environ["ALAYA_TRACK"],
    "reset_source_seq": seq,
    "reset_source_frame": frame,
    "requested_at": time.time(),
    "render_request_ts": time.time(),
    "reset_at": time.time(),
})
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
}

write_manifest() {
  local service_status="$1"
  local error_text="${2:-}"
  export_deploy_environment
  export service_status error_text
  local temporary="${ALAYA_RUNTIME_ROOT}/.deployment-manifest.json.$$.tmp"
  "${HOST_PYTHON}" - "${temporary}" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

def env(name, default=""):
    return os.environ.get(name, default)

def integer(name):
    return int(env(name))

def weight(label, path_name, actual_name, expected_name):
    path = pathlib.Path(env(path_name))
    return {
        "label": label,
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "hash_algorithm": "sha256-tree-v1" if path.is_dir() else "sha256",
        "sha256": env(actual_name),
        "expected_sha256": env(expected_name) or None,
    }

runtime = pathlib.Path(env("ALAYA_RUNTIME_ROOT"))
processes = {}
for name, command_name in (
    ("stk", "STK_COMMAND_JSON"),
    ("bridge", "BRIDGE_COMMAND_JSON"),
    ("worker", "WORKER_COMMAND_JSON"),
    ("preview", "PREVIEW_COMMAND_JSON"),
):
    processes[name] = {
        "command": json.loads(env(command_name)),
        "runner": str(runtime / "run" / f"{name}.sh"),
        "supervisor_pid_file": str(runtime / "pids" / f"{name}.pid"),
        "child_pid_file": str(runtime / "pids" / f"{name}.child.pid"),
        "log": str(runtime / "logs" / f"{name}.log"),
    }

payload = {
    "schema_version": integer("LAUNCHER_SCHEMA_VERSION"),
    "service": "alaya-stk-checkpoint-service",
    "status": env("service_status"),
    "error": env("error_text"),
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "runtime": {
        "id": env("ALAYA_RUNTIME_ID"),
        "root": str(runtime),
        "tmux_session": env("ALAYA_TMUX_SESSION"),
        "config_file": env("CONFIG_FILE") or None,
        "launcher": env("SCRIPT_PATH"),
        "launcher_sha256": env("LAUNCHER_ACTUAL_SHA256"),
    },
    "stk": {
        "launcher": env("ALAYA_STK_LAUNCHER"),
        "launcher_sha256": env("STK_LAUNCHER_ACTUAL_SHA256"),
        "binary": env("ALAYA_STK_BINARY"),
        "binary_sha256": env("STK_BINARY_ACTUAL_SHA256"),
        "expected_binary_sha256": env("ALAYA_STK_BINARY_SHA256") or None,
        "home": env("ALAYA_STK_HOME"),
        "home_template": env("ALAYA_STK_HOME_TEMPLATE"),
        "track": env("ALAYA_TRACK"),
        "gfx_preset": integer("ALAYA_STK_GFX_PRESET"),
        "track_handover_grace_seconds": integer("ALAYA_STK_HANDOVER_GRACE"),
    },
    "model": {
        "profile": env("ALAYA_MODEL_PROFILE"),
        "weights": {
            "base_model": weight("base_model", "ALAYA_BASE_MODEL", "BASE_MODEL_ACTUAL_SHA256", "ALAYA_BASE_MODEL_SHA256"),
            "checkpoint": weight("checkpoint", "ALAYA_MODEL_CHECKPOINT", "MODEL_CHECKPOINT_ACTUAL_SHA256", "ALAYA_MODEL_CHECKPOINT_SHA256"),
            "grouped_encoder": weight("grouped_encoder", "ALAYA_GROUPED_ENCODER", "GROUPED_ENCODER_ACTUAL_SHA256", "ALAYA_GROUPED_ENCODER_SHA256"),
            "tiny_decoder": weight("tiny_decoder", "ALAYA_TINY_DECODER", "TINY_DECODER_ACTUAL_SHA256", "ALAYA_TINY_DECODER_SHA256"),
            "causal_contract": weight("causal_contract", "ALAYA_CAUSAL_CONTRACT_FILE", "CAUSAL_CONTRACT_ACTUAL_SHA256", "ALAYA_CAUSAL_CONTRACT_SHA256"),
        },
        "denoising_steps": integer("ALAYA_NUM_INFERENCE_STEPS"),
        "latent_window": integer("ALAYA_LATENT_WINDOW"),
        "pixel_cadence": {
            "initial_frames": integer("ALAYA_INITIAL_PIXEL_FRAMES"),
            "resumed_frames": integer("ALAYA_RESUMED_PIXEL_FRAMES"),
        },
        "causal_window_contract": {
            "bridge_new_pixel_frames": {
                "initial": integer("ALAYA_INITIAL_PIXEL_FRAMES"),
                "resumed": integer("ALAYA_RESUMED_PIXEL_FRAMES"),
            },
            "encoder_input_frames": {"initial": 13, "resumed": 17},
            "latent_offset": {"initial": 0, "resumed": 1},
            "encoded_latents": {"initial": 4, "resumed": 5},
            "selected_latents": 4,
        },
        "encoder": {
            "mode": env("ALAYA_SMALL_ENCODER_MODE"),
            "dimension": integer("ALAYA_SMALL_ENCODER_DIM"),
            "branches": integer("ALAYA_GROUPED_ENCODER_BRANCHES"),
        },
        "decoder": "tiny",
        "causal_contract_id": env("ALAYA_CAUSAL_CONTRACT_ID"),
        "compile_mode": env("ALAYA_COMPILE_MODE"),
        "warmup_chunks": integer("ALAYA_WARMUP_CHUNKS"),
    },
    "dimensions": {
        "stk_source": [integer("ALAYA_SOURCE_WIDTH"), integer("ALAYA_SOURCE_HEIGHT")],
        "model_input": [integer("ALAYA_MODEL_WIDTH"), integer("ALAYA_MODEL_HEIGHT")],
        "preview_main_width": integer("ALAYA_PREVIEW_WIDTH"),
        "preview_gbuffer_width": integer("ALAYA_PREVIEW_GBUFFER_WIDTH"),
    },
    "transport": {
        "window": env("ALAYA_WINDOW_TRANSPORT"),
        "target_fps": float(env("ALAYA_TARGET_FPS")),
        "publish_fps": float(env("ALAYA_PUBLISH_FPS")),
    },
    "gpu_mapping": {
        "stk_egl_device": integer("ALAYA_STK_EGL_DEVICE"),
        "worker_cuda_visible_device": integer("ALAYA_WORKER_GPU"),
        "worker_device_inside_visibility": "cuda:0",
    },
    "ports": {
        "preview_bind_host": env("ALAYA_PREVIEW_HOST"),
        "preview_port": integer("ALAYA_PREVIEW_PORT"),
        "public_forward_port": integer("ALAYA_PUBLIC_PORT"),
    },
    "cache": {
        "root": env("CACHE_ROOT"),
        "torchinductor": env("TORCHINDUCTOR_CACHE_DIR"),
        "triton": env("TRITON_CACHE_DIR"),
        "huggingface": env("HF_HOME"),
    },
    "paths": {
        "capture": env("CAPTURE_DIR"),
        "control": env("CONTROL_DIR"),
        "windows": env("WINDOW_DIR"),
        "rendered": env("RENDERED_DIR"),
        "input_state": env("INPUT_STATE_PATH"),
    },
    "health": {
        "stale_after_seconds": integer("ALAYA_HEALTH_STALE_AFTER"),
        "preview_state": env("PREVIEW_STATE_URL"),
        "preview_model": env("PREVIEW_MODEL_URL"),
        "capture_latest": str(runtime / "capture" / "latest.json"),
        "bridge_manifest": str(runtime / "windows" / "manifest.json"),
        "worker_status": str(runtime / "control" / "worker_status.json"),
        "rendered_latest": str(runtime / "rendered" / "latest.json"),
    },
    "processes": processes,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
os.chmod(path, 0o600)
PY
  mv -f "${temporary}" "${MANIFEST_PATH}"
}

update_manifest_status() {
  local runtime_root="$1"
  local status="$2"
  local error_text="${3:-}"
  local manifest="${runtime_root}/deployment-manifest.json"
  [[ -f "${manifest}" ]] || return 0
  "${HOST_PYTHON}" - "${manifest}" "${status}" "${error_text}" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
value["status"] = sys.argv[2]
value["error"] = sys.argv[3]
value["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
}

json_condition() {
  local path="$1"
  local kind="$2"
  "${HOST_PYTHON}" - "${path}" "${kind}" "${SOURCE_SESSION_ID}" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
session = sys.argv[3]
try:
    value = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
if kind == "capture":
    ok = int(value.get("seq", -1)) >= 0 and bool(value.get("input_event_id"))
elif kind == "bridge":
    ok = bool(value.get("ready")) and value.get("source_session_id") == session
elif kind == "rendered":
    ok = value.get("source_session_id") == session and int(value.get("source_seq", -1)) >= 0
else:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

wait_json_condition() {
  local path="$1"
  local kind="$2"
  local timeout="$3"
  local started now
  started="$(date +%s)"
  while true; do
    if json_condition "${path}" "${kind}"; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= timeout )); then
      printf 'timed out waiting for %s readiness: %s\n' "${kind}" "${path}" >&2
      return 1
    fi
    sleep 0.2
  done
}

wait_http_ready() {
  local timeout="$1"
  local started now
  started="$(date +%s)"
  while true; do
    if "${ALAYA_CURL_BIN}" -fsS --max-time 2 "${PREVIEW_STATE_URL}" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= timeout )); then
      printf 'timed out waiting for Preview health: %s\n' "${PREVIEW_STATE_URL}" >&2
      return 1
    fi
    sleep 0.25
  done
}

tmux_has_session() {
  "${ALAYA_TMUX_BIN}" has-session -t "$1" >/dev/null 2>&1
}

resolve_current_runtime() {
  local runtime="${ALAYA_RUNTIME_ROOT:-}"
  if [[ -z "${runtime}" && -r "${CURRENT_RUNTIME_POINTER}" ]]; then
    IFS= read -r runtime <"${CURRENT_RUNTIME_POINTER}"
  fi
  [[ -n "${runtime}" ]] || die 'no current runtime; set ALAYA_RUNTIME_ROOT or run start'
  runtime="$(canonical_path "${runtime}")"
  [[ -f "${runtime}/deployment-manifest.json" ]] || die "runtime manifest is missing: ${runtime}"
  printf '%s\n' "${runtime}"
}

manifest_value() {
  local manifest="$1"
  local expression="$2"
  "${HOST_PYTHON}" - "${manifest}" "${expression}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for part in sys.argv[2].split('.'):
    value = value[part]
print(value)
PY
}

pid_belongs_to_runtime() {
  local pid="$1"
  local runtime_id="$2"
  local expected_runtime_root="$3"
  local argument
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  if [[ -r "/proc/${pid}/environ" ]] \
      && tr '\0' '\n' <"/proc/${pid}/environ" \
        | grep -Fx "ALAYA_RUNTIME_ID=${runtime_id}" >/dev/null; then
    return 0
  fi
  [[ -n "${expected_runtime_root}" && -r "/proc/${pid}/cmdline" ]] || return 1
  while IFS= read -r argument; do
    case "${argument}" in
      "${expected_runtime_root}/run/stk.sh"|\
      "${expected_runtime_root}/run/bridge.sh"|\
      "${expected_runtime_root}/run/worker.sh"|\
      "${expected_runtime_root}/run/preview.sh") return 0 ;;
    esac
  done < <(tr '\0' '\n' <"/proc/${pid}/cmdline")
  return 1
}

stop_runtime_processes() {
  local runtime_root="$1"
  local manifest="${runtime_root}/deployment-manifest.json"
  local runtime_id session pid_file pid
  runtime_id="$(manifest_value "${manifest}" runtime.id)"
  session="$(manifest_value "${manifest}" runtime.tmux_session)"
  if tmux_has_session "${session}"; then
    "${ALAYA_TMUX_BIN}" kill-session -t "${session}" >/dev/null 2>&1 || true
  fi
  sleep 0.5
  for pid_file in "${runtime_root}"/pids/*.child.pid "${runtime_root}"/pids/*.pid; do
    [[ -r "${pid_file}" ]] || continue
    IFS= read -r pid <"${pid_file}" || true
    if kill -0 "${pid}" 2>/dev/null \
        && pid_belongs_to_runtime "${pid}" "${runtime_id}" "${runtime_root}"; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  local started now alive
  started="$(date +%s)"
  while true; do
    alive=0
    for pid_file in "${runtime_root}"/pids/*.child.pid "${runtime_root}"/pids/*.pid; do
      [[ -r "${pid_file}" ]] || continue
      IFS= read -r pid <"${pid_file}" || true
      if kill -0 "${pid}" 2>/dev/null \
          && pid_belongs_to_runtime "${pid}" "${runtime_id}" "${runtime_root}"; then
        alive=1
      fi
    done
    [[ "${alive}" == "0" ]] && break
    now="$(date +%s)"
    (( now - started < ALAYA_STOP_TIMEOUT )) || break
    sleep 0.2
  done
  if [[ "${alive:-0}" == "1" ]]; then
    for pid_file in "${runtime_root}"/pids/*.child.pid "${runtime_root}"/pids/*.pid; do
      [[ -r "${pid_file}" ]] || continue
      IFS= read -r pid <"${pid_file}" || true
      if kill -0 "${pid}" 2>/dev/null \
          && pid_belongs_to_runtime "${pid}" "${runtime_id}" "${runtime_root}"; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    done
  fi
}

start_impl() {
  doctor_impl 1
  export_deploy_environment

  if [[ -r "${CURRENT_RUNTIME_POINTER}" ]]; then
    local previous
    IFS= read -r previous <"${CURRENT_RUNTIME_POINTER}" || true
    if [[ -n "${previous}" && -f "${previous}/deployment-manifest.json" ]]; then
      local previous_session previous_runtime_id previous_pid_file previous_pid
      previous_session="$(manifest_value "${previous}/deployment-manifest.json" runtime.tmux_session)"
      if tmux_has_session "${previous_session}"; then
        die "a current runtime is still active: ${previous} (${previous_session})"
      fi
      previous_runtime_id="$(manifest_value "${previous}/deployment-manifest.json" runtime.id)"
      for previous_pid_file in "${previous}"/pids/*.child.pid "${previous}"/pids/*.pid; do
        [[ -r "${previous_pid_file}" ]] || continue
        IFS= read -r previous_pid <"${previous_pid_file}" || true
        if kill -0 "${previous_pid}" 2>/dev/null \
            && pid_belongs_to_runtime \
              "${previous_pid}" "${previous_runtime_id}" "${previous}"; then
          die "a managed process from the previous runtime is still alive: ${previous_pid} (${previous})"
        fi
      done
    fi
  fi
  [[ ! -e "${ALAYA_RUNTIME_ROOT}" ]] || die "runtime root already exists: ${ALAYA_RUNTIME_ROOT}"

  mkdir -p "${CAPTURE_DIR}" "${CONTROL_DIR}" "${WINDOW_DIR}" "${RENDERED_DIR}"
  mkdir -p "$(dirname "${INPUT_STATE_PATH}")" "${LOG_DIR}" "${PID_DIR}" "${RUN_DIR}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}"
  mkdir -p "${ALAYA_STK_HOME}" "${ALAYA_RUNTIME_ROOT}/xdg"
  chmod 700 "${ALAYA_RUNTIME_ROOT}/xdg"

  configure_commands
  write_stk_supervisor_runner "${STK_COMMAND[@]}"
  write_supervisor_runner bridge "${BRIDGE_COMMAND[@]}"
  write_supervisor_runner worker "${WORKER_COMMAND[@]}"
  write_supervisor_runner preview "${PREVIEW_COMMAND[@]}"
  write_manifest starting

  STARTING=1
  start_cleanup() {
    local rc=$?
    trap - ERR INT TERM
    if [[ "${STARTING:-0}" == "1" ]]; then
      set +e
      stop_runtime_processes "${ALAYA_RUNTIME_ROOT}"
      update_manifest_status "${ALAYA_RUNTIME_ROOT}" failed "start failed with exit code ${rc}"
      printf 'start failed; runtime retained for diagnostics: %s\n' "${ALAYA_RUNTIME_ROOT}" >&2
    fi
    exit "${rc}"
  }
  trap start_cleanup ERR INT TERM

  write_initial_documents

  "${ALAYA_TMUX_BIN}" new-session -d \
    -e "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}" \
    -e "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}" \
    -s "${ALAYA_TMUX_SESSION}" -n stk "${RUN_DIR}/stk.sh"
  "${ALAYA_TMUX_BIN}" set-option -t "${ALAYA_TMUX_SESSION}" remain-on-exit on >/dev/null
  wait_json_condition "${CAPTURE_DIR}/latest.json" capture "${ALAYA_READY_TIMEOUT_STK}"
  rebind_live_control_to_capture

  "${ALAYA_TMUX_BIN}" new-window -d \
    -e "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}" \
    -e "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}" \
    -t "${ALAYA_TMUX_SESSION}" -n bridge "${RUN_DIR}/bridge.sh"
  wait_json_condition "${WINDOW_DIR}/manifest.json" bridge "${ALAYA_READY_TIMEOUT_BRIDGE}"

  "${ALAYA_TMUX_BIN}" new-window -d \
    -e "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}" \
    -e "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}" \
    -t "${ALAYA_TMUX_SESSION}" -n worker "${RUN_DIR}/worker.sh"
  wait_json_condition "${RENDERED_DIR}/latest.json" rendered "${ALAYA_READY_TIMEOUT_WORKER}"

  "${ALAYA_TMUX_BIN}" new-window -d \
    -e "ALAYA_RUNTIME_ID=${ALAYA_RUNTIME_ID}" \
    -e "ALAYA_RUNTIME_ROOT=${ALAYA_RUNTIME_ROOT}" \
    -t "${ALAYA_TMUX_SESSION}" -n preview "${RUN_DIR}/preview.sh"
  wait_http_ready "${ALAYA_READY_TIMEOUT_PREVIEW}"

  # Size-cap the service logs (child stdout/stderr of every process) in place.
  "${ALAYA_TMUX_BIN}" new-window -d -t "${ALAYA_TMUX_SESSION}" -n logcap \
    "${ALAYA_PYTHON_BIN}" -m alayarenderer_flash.demo.logcap "${LOG_DIR}" \
    --max-bytes "${ALAYA_LOG_MAX_BYTES}" --backups "${ALAYA_LOG_BACKUPS}"

  write_manifest running
  mkdir -p "${ALAYA_STATE_DIR}"
  atomic_write_text "${CURRENT_RUNTIME_POINTER}" "${ALAYA_RUNTIME_ROOT}"
  STARTING=0
  trap - ERR INT TERM
  printf 'started runtime=%s\n' "${ALAYA_RUNTIME_ROOT}"
  printf 'tmux=%s preview=%s\n' "${ALAYA_TMUX_SESSION}" "${PREVIEW_STATE_URL}"
}

status_impl() {
  load_deploy_config
  local runtime_root manifest
  runtime_root="$(resolve_current_runtime)"
  manifest="${runtime_root}/deployment-manifest.json"
  export runtime_root manifest OUTPUT_JSON ALAYA_TMUX_BIN
  "${HOST_PYTHON}" - <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

root = pathlib.Path(os.environ["runtime_root"])
manifest_path = pathlib.Path(os.environ["manifest"])
manifest = json.loads(manifest_path.read_text())
runtime_id = manifest["runtime"]["id"]
session = manifest["runtime"]["tmux_session"]
tmux_bin = os.environ["ALAYA_TMUX_BIN"]

def pid_state(path, runner=None):
    try:
        pid = int(pathlib.Path(path).read_text().strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        return {"pid": None, "alive": False, "runtime_identity_match": False}
    identity_match = False
    try:
        values = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        identity_match = f"ALAYA_RUNTIME_ID={runtime_id}".encode() in values
    except Exception:
        pass
    if not identity_match and runner:
        try:
            runner_path = pathlib.Path(runner)
            argv = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            identity_match = bool(
                runner_path.is_absolute()
                and runner_path.parent == root / "run"
                and runner_path.name in {"stk.sh", "bridge.sh", "worker.sh", "preview.sh"}
                and str(runner_path).encode() in argv
            )
        except Exception:
            pass
    return {"pid": pid, "alive": alive, "runtime_identity_match": identity_match}

try:
    tmux_alive = subprocess.run(
        [tmux_bin, "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
except Exception:
    tmux_alive = False

processes = {}
for name, process in manifest["processes"].items():
    processes[name] = {
        "supervisor": pid_state(process["supervisor_pid_file"], process["runner"]),
        "child": pid_state(process["child_pid_file"]),
        "log": process["log"],
    }

documents = {}
stale_after = float(manifest["health"].get("stale_after_seconds", 5))
for name, path in manifest["health"].items():
    if not isinstance(path, str) or not name.endswith(("latest", "manifest", "status")):
        continue
    try:
        document_path = pathlib.Path(path)
        value = json.loads(document_path.read_text())
        age_seconds = max(0.0, time.time() - document_path.stat().st_mtime)
        documents[name] = {
            "ok": age_seconds <= stale_after,
            "age_seconds": round(age_seconds, 3),
            "stale_after_seconds": stale_after,
            "seq": value.get("seq"),
            "source_seq": value.get("source_seq"),
            "status": value.get("status"),
            "ready": value.get("ready"),
            "source_session_id": value.get("source_session_id"),
        }
    except Exception as error:
        documents[name] = {"ok": False, "error": str(error)[:300]}

try:
    with urllib.request.urlopen(manifest["health"]["preview_state"], timeout=2) as response:
        preview = {"ok": response.status == 200, "status": response.status}
except Exception as error:
    preview = {"ok": False, "error": str(error)[:300]}

process_healthy = all(
    value["supervisor"]["alive"]
    and value["supervisor"]["runtime_identity_match"]
    and value["child"]["alive"]
    and value["child"]["runtime_identity_match"]
    for value in processes.values()
)
healthy = bool(
    manifest.get("status") == "running"
    and tmux_alive
    and process_healthy
    and preview.get("ok")
    and all(value.get("ok") for value in documents.values())
)
payload = {
    "schema_version": 1,
    "healthy": healthy,
    "manifest_status": manifest.get("status"),
    "runtime_id": runtime_id,
    "runtime_root": str(root),
    "tmux_session": session,
    "tmux_alive": tmux_alive,
    "processes": processes,
    "documents": documents,
    "preview": preview,
    "health_endpoints": manifest["health"],
}
if os.environ.get("OUTPUT_JSON") == "1":
    print(json.dumps(payload, sort_keys=True))
else:
    print(f"healthy={str(healthy).lower()} status={manifest.get('status')} runtime={root}")
    print(f"tmux={session} alive={str(tmux_alive).lower()} preview={preview.get('ok', False)}")
    for name, value in processes.items():
        print(
            f"{name}: supervisor={value['supervisor']['pid']}:{value['supervisor']['alive']} "
            f"child={value['child']['pid']}:{value['child']['alive']}"
        )
raise SystemExit(0 if healthy else 1)
PY
}

logs_impl() {
  load_deploy_config
  local runtime_root manifest log_paths=() service
  runtime_root="$(resolve_current_runtime)"
  manifest="${runtime_root}/deployment-manifest.json"
  if [[ "${LOG_SERVICE}" == "all" ]]; then
    for service in stk bridge worker preview; do
      log_paths+=("$(manifest_value "${manifest}" "processes.${service}.log")")
    done
  else
    log_paths+=("$(manifest_value "${manifest}" "processes.${LOG_SERVICE}.log")")
  fi
  if [[ "${LOG_FOLLOW}" == "1" ]]; then
    tail -n "${LOG_LINES}" -F "${log_paths[@]}"
  else
    tail -n "${LOG_LINES}" "${log_paths[@]}"
  fi
}

stop_impl() {
  load_deploy_config
  local runtime_root pointer_value=""
  runtime_root="$(resolve_current_runtime)"
  stop_runtime_processes "${runtime_root}"
  update_manifest_status "${runtime_root}" stopped
  if [[ -r "${CURRENT_RUNTIME_POINTER}" ]]; then
    IFS= read -r pointer_value <"${CURRENT_RUNTIME_POINTER}" || true
  fi
  if [[ "$(canonical_path "${pointer_value:-/}")" == "${runtime_root}" ]]; then
    rm -f "${CURRENT_RUNTIME_POINTER}"
  fi
  # The runtime root lives in the size-capped /dev/shm and logs are on disk, so
  # free it once the processes are gone instead of leaking one per restart.
  case "${runtime_root}" in
    "$(canonical_path "${ALAYA_RUNTIME_BASE}")"/*) rm -rf -- "${runtime_root}" ;;
  esac
  printf 'stopped runtime=%s\n' "${runtime_root}"
}

case "${COMMAND}" in
  doctor) doctor_impl 0 ;;
  start) start_impl ;;
  status) status_impl ;;
  logs) logs_impl ;;
  stop) stop_impl ;;
esac
