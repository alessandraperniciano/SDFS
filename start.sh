#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -d .venv ]]; then
    "${PYTHON:-python3.11}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
if [[ $# -gt 0 ]]; then
    exec .venv/bin/python -m src.run "$@"
fi
echo 'Environment ready. Run: .venv/bin/python -m src.run --config configs/mnist.json'
