from typing import Dict

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> Dict[str, str]:
    cuda_available = torch.cuda.is_available()
    info = {
        "cuda_available": str(cuda_available),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "device": "cuda" if cuda_available else "cpu",
    }

    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())
        info["gpu_capability"] = str(torch.cuda.get_device_capability(0))
    else:
        info["gpu_name"] = "None"
        info["gpu_count"] = "0"
        info["gpu_capability"] = "N/A"

    return info


if __name__ == "__main__":
    details = get_device_info()
    print(f"Is PyTorch talking to the GPU? {details['cuda_available']}")
    print(f"Torch Version: {details['torch_version']}")
    print(f"CUDA Version: {details['cuda_version']}")
    print(f"Device: {details['device']}")
    print(f"GPU Name: {details['gpu_name']}")
    print(f"GPU Count: {details['gpu_count']}")
    print(f"GPU Capability: {details['gpu_capability']}")