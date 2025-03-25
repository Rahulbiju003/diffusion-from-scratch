import torch

IMAGE_SIZE = 64
BATCH_SIZE = 128
T = 200
EPOCHS = 100
LEARNING_RATE = 0.001
DATA_DIR = "images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"