#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p input output models video_input output/subs output/subs_burned

sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# demucs + resemble-enhance runtime deps (package itself installed --no-deps)
pip install demucs soundfile "numpy<2" tqdm librosa omegaconf rich resampy tabulate scipy \
    matplotlib pandas celluloid ptflops faster-whisper "coqui-tts" "transformers>=4.43,<5"

# resemble-enhance pins torch==2.1.1 — install without deps on Python 3.12
pip install --no-deps --ignore-requires-python "resemble-enhance" --pre
# deepspeed is NOT installed: inference uses a stub in app/resemble_runner.py

python -c "import torch; import demucs; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import matplotlib, pandas, resemble_enhance; print('resemble-enhance ok')"

echo "Setup done. Run: ./run.sh"
