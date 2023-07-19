import torch


SAVED_MODEL_PATH = './models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
LEARNING_RATE = 0.1
BATCH_SIZE = 1024
ROOT_DIR_TRAIN = './airbus-ship-detection/train'
ROOT_DIR_TEST = './airbus-ship-detection/test_v2'
LOAD_MODEL = False
PATH_TO_MODEL = None