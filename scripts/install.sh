#!/usr/bin/env bash
# Install okxrun as two systemd services on a Debian/Ubuntu VPS.
#
#   git clone https://github.com/kirin233x/okxrun.git && cd okxrun
#   cp .env.example .env && $EDITOR .env
#   sudo ./scripts/install.sh
#
# Re-running is safe: it re-syncs dependencies and restarts the services.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
UNIT_DIR=/etc/systemd/system

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo: sudo ./scripts/install.sh" >&2
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "missing $ROOT/.env — copy .env.example and fill it in first" >&2
  exit 1
fi

UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "==> syncing dependencies"
sudo -u "$SERVICE_USER" "$UV" sync --project "$ROOT" --extra portal

echo "==> tightening .env permissions (it holds API keys)"
chown "$SERVICE_USER" "$ROOT/.env"
chmod 600 "$ROOT/.env"

mkdir -p "$ROOT/state"
chown -R "$SERVICE_USER" "$ROOT/state"

write_unit() {
  local name="$1" description="$2" exec_start="$3"
  echo "==> writing $UNIT_DIR/$name"
  cat >"$UNIT_DIR/$name" <<UNIT
[Unit]
Description=$description
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$ROOT
ExecStart=$exec_start
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# The executor holds trading keys; give it as little of the box as possible.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$ROOT/state
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
UNIT
}

write_unit "okxrun-executor.service" "okxrun trading executor" \
  "$UV run --project $ROOT python -m live run"
write_unit "okxrun-portal.service" "okxrun read-only portal" \
  "$UV run --project $ROOT --extra portal uvicorn portal.app:app --host 127.0.0.1 --port 8787"

systemctl daemon-reload
systemctl enable --now okxrun-executor.service okxrun-portal.service

cat <<DONE

installed.

  status   systemctl status okxrun-executor okxrun-portal
  logs     journalctl -u okxrun-executor -f
  portal   http://127.0.0.1:8787   (bound to localhost on purpose)
  stop     touch $ROOT/state/HALT      # flattens every position, blocks new risk
  resume   rm $ROOT/state/HALT && systemctl restart okxrun-executor

The portal listens on localhost only. To reach it from your laptop, either
tunnel it over ssh:

  ssh -N -L 8787:127.0.0.1:8787 $SERVICE_USER@<vps>

or put a reverse proxy with authentication in front of it. Do not bind it to
0.0.0.0 without auth.

OKX_DRY_RUN is 1 by default. Run a plan first and read it:

  cd $ROOT && $UV run --project . python -m live preflight
  cd $ROOT && $UV run --project . python -m live plan

DONE
