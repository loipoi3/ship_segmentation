from config import *
import os


def main():
    # if directory model exists than create this
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

    # define data transformations
    transform_train = A.Compose([A.Resize(64, 64),

                                 ToTensorV2()])
    transform_val_test = A.Compose([A.Resize(64, 64),
                                    ToTensorV2()])



if __name__ == '__main__':
    main()