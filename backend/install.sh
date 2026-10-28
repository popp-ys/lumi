#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
if ! command -v python >/dev/null 2>&1; then
  echo 'Python 3 is required. On Arch: sudo pacman -S --needed python python-pip' >&2
  exit 1
fi
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
if [[ ! -f config.json ]]; then
  python - <<'PY'
import json, secrets
from pathlib import Path
p=Path('config.example.json')
c=json.loads(p.read_text())
c['api_token']=secrets.token_urlsafe(32)
Path('config.json').write_text(json.dumps(c, indent=2))
print('\nLUMI API TOKEN (copy this into the desktop app settings):\n' + c['api_token'] + '\n')
PY
  chmod 600 config.json
else
  echo 'Existing backend/config.json preserved.'
fi
mkdir -p data/backups
chmod 700 data data/backups
cat <<'MSG'
Install complete.
Next: install Ollama, start its service, and run `ollama pull qwen3:8b`.
Then run ./run.sh. Voice recognition is an optional separate install:
  .venv/bin/pip install -r requirements-voice.txt
MSG
