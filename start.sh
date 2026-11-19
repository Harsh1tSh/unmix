#!/usr/bin/env bash
# Starts Unmix. Ctrl+C stops everything.
#
#   ./start.sh                    # backend on this computer + frontend (http://localhost:5173)
#   ./start.sh --fake             # instant fake "models" (UI work without the weights)
#   ./start.sh --remote <url>     # frontend only, using a backend running elsewhere
#                                 # (the Colab notebook in colab/ prints the URL and key;
#                                 #  the key is asked for, or read from UNMIX_BACKEND_KEY)
#   ./start.sh --accel cpu|dml|cuda|cuda12   # override the automatic choice
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fake=0; remote=""; accel=""
while [ $# -gt 0 ]; do
  case "$1" in
    --fake) fake=1 ;;
    --remote) remote="${2:-}"; shift ;;
    --accel) accel="${2:-}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

tools="npm"; [ -z "$remote" ] && tools="uv npm ffmpeg"
for tool in $tools; do
  command -v "$tool" >/dev/null || { echo "$tool was not found on PATH. See README.md > Prerequisites." >&2; exit 1; }
done
[ -d "$root/frontend/node_modules" ] || (cd "$root/frontend" && npm install)

if [ -n "$remote" ]; then
  remote="${remote%/}"
  [[ "$remote" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]] || { echo "--remote must be an https:// address" >&2; exit 2; }
  key="${UNMIX_BACKEND_KEY:-}"
  if [ -z "$key" ]; then read -r -s -p "Access key (printed by the notebook): " key; echo; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-Unmix-Key: $key" "$remote/api/health" || true)
  [ "$code" = "401" ] && { echo "The remote backend refused the access key." >&2; exit 1; }
  [ "$code" = "200" ] || { echo "Could not reach $remote (HTTP $code). Is the notebook still running?" >&2; exit 1; }
  echo "Remote backend reachable."
  # Only the local proxy (Node) holds the key; the browser never sees it.
  export UNMIX_BACKEND_URL="$remote" UNMIX_BACKEND_KEY="$key"
  cd "$root/frontend" && exec npm run dev
fi

if [ "$fake" = 1 ]; then export UNMIX_FAKE_MODELS=1; else unset UNMIX_FAKE_MODELS; fi
unset UNMIX_BACKEND_URL UNMIX_BACKEND_KEY
[ -n "$accel" ] && export UNMIX_ACCEL="$accel"
pick="$(uv run --no-project --quiet python "$root/backend/scripts/pick_accel.py")"
group="$(printf '%s' "$pick" | sed -E 's/.*"group": "([a-z0-9]+)".*/\1/')"
echo "Accelerator build: $group ($(printf '%s' "$pick" | sed -E 's/.*"reason": "([^"]*)".*/\1/'))"

if [ -d "$root/backend/speakers/.venv" ]; then
  sgroup=cpu; [ "$group" = cuda ] && sgroup=cuda   # see backend/speakers/pyproject.toml
  (cd "$root/backend/speakers" && uv sync --quiet --no-default-groups --group "$sgroup")
fi

echo "Starting backend on http://127.0.0.1:8000 ..."
(cd "$root/backend" && exec uv run --no-default-groups --group "$group" uvicorn app.server:app --host 127.0.0.1 --port 8000) &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT INT TERM

echo "Starting frontend on http://localhost:5173 ..."
cd "$root/frontend" && npm run dev
