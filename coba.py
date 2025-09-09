import torch

print(torch.__version__)              # harusnya 2.8.0+cu126
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
