#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p input output models

sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo "Setup done. Run: ./run.sh"
