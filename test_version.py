import numpy as np
import cv2
import torch
import onnxruntime as ort

print(np.__version__)
print(cv2.__version__)
print(torch.__version__)
print(ort.__version__)


print("CUDA available (PyTorch):", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(ort.get_device())