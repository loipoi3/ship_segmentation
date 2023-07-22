import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


SAVED_MODEL_PATH = './models/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 2
ROOT_DIR_TRAIN = './airbus-ship-detection/train_v2'
LOAD_MODEL = False
PATH_TO_MODEL = './models/model.pth'
CSV_FILE = './airbus-ship-detection/train_ship_segmentations_v2.csv'
TRANSFORM_TRAIN = A.Compose([A.RandomBrightnessContrast(p=0.5),
                             A.HorizontalFlip(p=0.5),
                             A.VerticalFlip(p=0.5),
                             A.Rotate(limit=45, p=0.5),
                             A.RandomScale(scale_limit=0.2, p=0.5),
                             A.RandomCrop(height=600, width=600, p=0.5),
                             A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0, val_shift_limit=0, p=0.5),
                             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                             A.RandomResizedCrop(height=600, width=600, scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
                             A.Resize(768, 768),
                             ToTensorV2()])
TRANSFORM_VAL_TEST = A.Compose([ToTensorV2()])