import torch


# Устройство:
# - CUDA (NVIDIA) — если доступно
# - MPS (Apple Silicon) — если доступно (только macOS)
# - иначе CPU
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

