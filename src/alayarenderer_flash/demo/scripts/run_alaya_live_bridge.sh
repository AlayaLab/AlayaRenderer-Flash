#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_DIR}/.runtime/alaya_live_control}"
CAPTURE_DIR="${CAPTURE_DIR:-/dev/shm/stk_browser_preview_live}"
WINDOW_DIR="${WINDOW_DIR:-/dev/shm/stk_alaya_windows}"
INPUT_STATE_PATH="${INPUT_STATE_PATH:-${CAPTURE_DIR}/input.json}"
EXPECTED_WIDTH="${EXPECTED_WIDTH:-1280}"
EXPECTED_HEIGHT="${EXPECTED_HEIGHT:-720}"
RESTART_DELAY="${RESTART_DELAY:-2}"
ALAYA_30FPS_SHADOW="${ALAYA_30FPS_SHADOW:-0}"
ALAYA_LOW_LATENCY_PROFILE="${ALAYA_LOW_LATENCY_PROFILE:-0}"
ALAYA_VALIDATE_ONLY="${ALAYA_VALIDATE_ONLY:-0}"
case "${ALAYA_30FPS_SHADOW}" in
  1|true|TRUE|yes|YES|on|ON) shadow_30fps=1 ;;
  0|false|FALSE|no|NO|off|OFF) shadow_30fps=0 ;;
  *)
    printf 'ALAYA_30FPS_SHADOW must be a boolean, got %s\n' "${ALAYA_30FPS_SHADOW}" >&2
    exit 2
    ;;
esac
case "${ALAYA_LOW_LATENCY_PROFILE}" in
  1|true|TRUE|yes|YES|on|ON) low_latency=1 ;;
  0|false|FALSE|no|NO|off|OFF) low_latency=0 ;;
  *)
    printf 'ALAYA_LOW_LATENCY_PROFILE must be a boolean, got %s\n' \
      "${ALAYA_LOW_LATENCY_PROFILE}" >&2
    exit 2
    ;;
esac
case "${ALAYA_VALIDATE_ONLY}" in
  1|true|TRUE|yes|YES|on|ON) validate_only=1 ;;
  0|false|FALSE|no|NO|off|OFF) validate_only=0 ;;
  *)
    printf 'ALAYA_VALIDATE_ONLY must be a boolean, got %s\n' \
      "${ALAYA_VALIDATE_ONLY}" >&2
    exit 2
    ;;
esac
if [[ "${shadow_30fps}" == 1 && "${low_latency}" == 1 ]]; then
  printf 'ALAYA_30FPS_SHADOW and ALAYA_LOW_LATENCY_PROFILE are mutually exclusive\n' >&2
  exit 2
fi
if [[ "${low_latency}" == 1 ]]; then
  profile_name=low_latency
  LATENT_WINDOW="${LATENT_WINDOW:-2}"
else
  if [[ "${shadow_30fps}" == 1 ]]; then
    profile_name=shadow_30fps
  else
    profile_name=v5
  fi
  LATENT_WINDOW="${LATENT_WINDOW:-4}"
fi
if [[ "${shadow_30fps}" == 1 ]]; then
  if [[ "${LATENT_WINDOW}" != 4 ]]; then
    printf 'ALAYA_30FPS_SHADOW requires LATENT_WINDOW=4\n' >&2
    exit 2
  fi
  WINDOW_TRANSPORT="${WINDOW_TRANSPORT:-shm_slots}"
  if [[ "${WINDOW_TRANSPORT}" != shm_slots ]]; then
    printf 'ALAYA_30FPS_SHADOW requires WINDOW_TRANSPORT=shm_slots\n' >&2
    exit 2
  fi
  TARGET_FPS="${TARGET_FPS:-30.0}"
elif [[ "${low_latency}" == 1 ]]; then
  if [[ "${LATENT_WINDOW}" != 2 ]]; then
    printf 'ALAYA_LOW_LATENCY_PROFILE requires LATENT_WINDOW=2\n' >&2
    exit 2
  fi
  WINDOW_TRANSPORT="${WINDOW_TRANSPORT:-shm_slots}"
  if [[ "${WINDOW_TRANSPORT}" != shm_slots ]]; then
    printf 'ALAYA_LOW_LATENCY_PROFILE requires WINDOW_TRANSPORT=shm_slots\n' >&2
    exit 2
  fi
  TARGET_FPS="${TARGET_FPS:-30.0}"
else
  WINDOW_TRANSPORT="${WINDOW_TRANSPORT:-npz}"
fi
WINDOW_WRITER_QUEUE_SIZE="${WINDOW_WRITER_QUEUE_SIZE:-2}"
MAX_WORKER_LAG_WINDOWS="${MAX_WORKER_LAG_WINDOWS:-2}"
case "${LATENT_WINDOW}" in
  2) TARGET_FPS="${TARGET_FPS:-22.0}" ;;
  4) TARGET_FPS="${TARGET_FPS:-22.5}" ;;
  *)
    printf 'LATENT_WINDOW must be 2 or 4, got %s\n' "${LATENT_WINDOW}" >&2
    exit 2
    ;;
esac
case "${WINDOW_TRANSPORT}" in
  npz|npy|shm_slots) ;;
  *)
    printf 'WINDOW_TRANSPORT must be npz, npy, or shm_slots, got %s\n' \
      "${WINDOW_TRANSPORT}" >&2
    exit 2
    ;;
esac
if [[ ! "${EXPECTED_WIDTH}" =~ ^[1-9][0-9]*$ ||
      ! "${EXPECTED_HEIGHT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EXPECTED_WIDTH and EXPECTED_HEIGHT must be positive integers, got %sx%s\n' \
    "${EXPECTED_WIDTH}" "${EXPECTED_HEIGHT}" >&2
  exit 2
fi

printf 'bridge temporal mode: profile=%s latent_window=%s target_fps=%s transport=%s shadow30=%s writer_queue=%s source=%sx%s input_state=%s\n' \
  "${profile_name}" \
  "${LATENT_WINDOW}" "${TARGET_FPS}" "${WINDOW_TRANSPORT}" \
  "${shadow_30fps}" "${WINDOW_WRITER_QUEUE_SIZE}" \
  "${EXPECTED_WIDTH}" "${EXPECTED_HEIGHT}" "${INPUT_STATE_PATH}"
if [[ "${validate_only}" == 1 ]]; then
  exit 0
fi

mkdir -p "${CONTROL_DIR}" "${WINDOW_DIR}"

child_pid=""
stop_child() {
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit 0
}
trap stop_child INT TERM

while true; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/../bridge.py" \
    --capture-dir "${CAPTURE_DIR}" \
    --control-dir "${CONTROL_DIR}" \
    --out-dir "${WINDOW_DIR}" \
    --input-state "${INPUT_STATE_PATH}" \
    --expected-width "${EXPECTED_WIDTH}" \
    --expected-height "${EXPECTED_HEIGHT}" \
    --target-width 832 \
    --target-height 448 \
    --target-fps "${TARGET_FPS}" \
    --latent-window "${LATENT_WINDOW}" \
    --window-transport "${WINDOW_TRANSPORT}" \
    --window-writer-queue-size "${WINDOW_WRITER_QUEUE_SIZE}" \
    --retain-windows 64 \
    --max-worker-lag-windows "${MAX_WORKER_LAG_WINDOWS}" \
    --worker-lag-observations 3 \
    --worker-lag-resync-cooldown 15 \
    --stale-after 3 \
    --log-level INFO &
  child_pid=$!
  wait "${child_pid}"
  exit_code=$?
  child_pid=""
  printf '[%s] bridge exited with code %s; restarting in %ss\n' \
    "$(date -Is)" "${exit_code}" "${RESTART_DELAY}" >&2
  sleep "${RESTART_DELAY}"
done
