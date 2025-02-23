import numpy as np
import tensorflow as tf
import cv2
import torch
import onnxruntime as ort
from tensorflow.python.client import device_lib

print(tf.__version__)
print(np.__version__)
print(cv2.__version__)
print(torch.__version__)
print(ort.__version__)

print(device_lib.list_local_devices())
print("CUDA available (PyTorch):", torch.cuda.is_available())
tf.config.list_physical_devices('GPU')
print(ort.get_device())