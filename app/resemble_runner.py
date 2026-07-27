import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from config import MODEL_DIR


def _stub_deepspeed() -> None:
    if "deepspeed" in sys.modules:
        return

    deepspeed = types.ModuleType("deepspeed")

    class DeepSpeedConfig:
        def __init__(self, *args, **kwargs):
            pass

    deepspeed.DeepSpeedConfig = DeepSpeedConfig
    sys.modules["deepspeed"] = deepspeed


def _setup_cache() -> None:
    os.environ.setdefault("HF_HOME", MODEL_DIR)
    os.environ.setdefault("TORCH_HOME", MODEL_DIR)
    os.environ.setdefault("XDG_CACHE_HOME", MODEL_DIR)


@lru_cache(maxsize=1)
def _load_enhancer(device: str):
    _setup_cache()
    _stub_deepspeed()

    from resemble_enhance.enhancer.download import download
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams

    run_dir = download(None)
    hp = HParams.load(run_dir)
    enhancer = Enhancer(hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu", weights_only=False)["module"]
    enhancer.load_state_dict(state_dict)
    enhancer.eval()
    enhancer.to(device)
    return enhancer


def run_enhance(input_wav: Path, output_wav: Path, device: str = "cuda") -> None:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from resemble_enhance.inference import inference

    enhancer = _load_enhancer(device)
    enhancer.configurate_(nfe=64, solver="midpoint", lambd=0.5, tau=0.5)

    dwav, sr = torchaudio.load(str(input_wav))
    dwav = dwav.mean(0)

    hwav, sr = inference(model=enhancer, dwav=dwav, sr=sr, device=device)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_wav), hwav[None], sr)
