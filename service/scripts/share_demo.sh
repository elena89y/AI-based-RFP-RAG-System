#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APP_FILE="${APP_FILE:-$ROOT_DIR/app.py}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8501}"
PYTHON_BIN="${PYTHON_BIN:-}"

find_streamlit_python() {
  local candidates=()
  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=("python3" "python3.12" "python3.11")

  local py
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
      if "$py" -m streamlit --version >/dev/null 2>&1; then
        echo "$py"
        return 0
      fi
    fi
  done
  return 1
}

detect_cloudflared_config() {
  local paths=(
    "$HOME/.cloudflared/config.yaml"
    "$HOME/.cloudflared/config.yml"
    "/etc/cloudflared/config.yaml"
    "/etc/cloudflared/config.yml"
    "/usr/local/etc/cloudflared/config.yaml"
    "/usr/local/etc/cloudflared/config.yml"
  )
  local p
  for p in "${paths[@]}"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[ERROR] cloudflared not found."
  echo "Install: https://developers.cloudflare.com/tunnel/setup/"
  exit 1
fi

if ! PY="$(find_streamlit_python)"; then
  echo "[ERROR] Streamlit is not installed in available Python environments."
  echo "Try: pip install -r \"$ROOT_DIR/requirements.txt\""
  exit 1
fi

if [[ ! -f "$APP_FILE" ]]; then
  echo "[ERROR] app file not found: $APP_FILE"
  exit 1
fi

if CFG="$(detect_cloudflared_config)"; then
  echo "[WARN] Quick Tunnel may fail because config file exists: $CFG"
  echo "[WARN] If tunnel start fails, temporarily rename it and retry."
fi

STREAMLIT_PID=""
cleanup() {
  if [[ -n "$STREAMLIT_PID" ]] && kill -0 "$STREAMLIT_PID" >/dev/null 2>&1; then
    kill "$STREAMLIT_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/3] Starting Streamlit"
"$PY" -m streamlit run "$APP_FILE" \
  --server.address="$APP_HOST" \
  --server.port="$APP_PORT" \
  --server.headless=true \
  >/tmp/streamlit_demo.log 2>&1 &
STREAMLIT_PID="$!"

echo "[2/3] Waiting for Streamlit health endpoint"
READY=0
for _ in $(seq 1 40); do
  if curl -fsS "http://$APP_HOST:$APP_PORT/_stcore/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[ERROR] Streamlit did not start."
  echo "----- /tmp/streamlit_demo.log -----"
  tail -n 80 /tmp/streamlit_demo.log || true
  exit 1
fi

echo "[3/3] Starting Cloudflare Quick Tunnel"
echo "Share the printed https://*.trycloudflare.com URL for demo access."
echo "Press Ctrl+C to stop both Streamlit and the tunnel."
echo

cloudflared tunnel --url "http://$APP_HOST:$APP_PORT"
