import numpy as np
import cv2
import torch
import onnxruntime as ort
# import tensorflow as tf
# from tensorflow.python.client import device_lib

#print(tf.__version__)
print(np.__version__)
print(cv2.__version__)
print(torch.__version__)
print(ort.__version__)


print("CUDA available (PyTorch):", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(ort.get_device())
# tf.config.list_physical_devices('GPU')
# print(device_lib.list_local_devices())
