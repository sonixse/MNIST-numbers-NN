"""GPU helper utilities."""

import torch

print("Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Capability:", torch.cuda.get_device_capability(0))
else:
    print("GPU Name:", "N/A")
    print("GPU Capability:", "N/A")