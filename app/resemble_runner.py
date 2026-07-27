import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from config import MODEL_DIR


def _ensure_deepspeed() -> None:
    """Use real deepspeed if installed; otherwise provide a minimal stub."""
    try:
        import deepspeed  # noqa: F401
        import deepspeed.accelerator  # noqa: F401

        return
    except Exception:
        pass

    class DeepSpeedConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _Accelerator:
        def communication_backend_name(self):
            return "nccl"

    class DeepSpeedEngine(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.DeepSpeedConfig = DeepSpeedConfig
    deepspeed.init_distributed = lambda *a, **k: None

    accelerator = types.ModuleType("deepspeed.accelerator")
    accelerator.get_accelerator = lambda: _Accelerator()

    runtime = types.ModuleType("deepspeed.runtime")
    runtime_engine = types.ModuleType("deepspeed.runtime.engine")
    runtime_engine.DeepSpeedEngine = DeepSpeedEngine
    runtime_utils = types.ModuleType("deepspeed.runtime.utils")
    runtime_utils.clip_grad_norm_ = lambda *a, **k: None

    sys.modules["deepspeed"] = deepspeed
    sys.modules["deepspeed.accelerator"] = accelerator
    sys.modules["deepspeed.runtime"] = runtime
    sys.modules["deepspeed.runtime.engine"] = runtime_engine
    sys.modules["deepspeed.runtime.utils"] = runtime_utils


def _setup_cache() -> None:
    os.environ.setdefault("HF_HOME", MODEL_DIR)
    os.environ.setdefault("TORCH_HOME", MODEL_DIR)
    os.environ.setdefault("XDG_CACHE_HOME", MODEL_DIR)


@lru_cache(maxsize=1)
def _load_enhancer(device: str):
    _setup_cache()
    _ensure_deepspeed()

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
