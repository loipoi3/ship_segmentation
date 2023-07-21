from torch.utils.data import DataLoader
from config import (SAVED_MODEL_PATH,
                    DEVICE,
                    NUM_EPOCHS,
                    LEARNING_RATE,
                    BATCH_SIZE,
                    ROOT_DIR_TRAIN,
                    LOAD_MODEL,
                    PATH_TO_MODEL,
                    CSV_FILE,
                    TRANSFORM_TRAIN,
                    TRANSFORM_VAL_TEST)
import os
import segmentation_models_pytorch as smp
from dataset import ShipDataset
import torch
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
import torch.optim as optim
from utils import calculate_dice_coefficient


# Define a collate function for the training dataset
def train_collate_fn(batch):
    images, masks = zip(*batch)
    transformed_images = [transform_train(image=img)['image'] for img in images]
    transformed_masks = [transform_val_test(image=mask)['image'] for mask in masks]
    return torch.stack(transformed_images), torch.stack(transformed_masks)


# Define a collate function for the validation and test datasets
def val_test_collate_fn(batch):
    images, masks = zip(*batch)
    transformed_images = [transform_val_test(image=img)['image'] for img in images]
    transformed_masks = [transform_val_test(image=mask)['image'] for mask in masks]
    return torch.stack(transformed_images), torch.stack(transformed_masks)


def train_loop(model, criterion, optimizer, train_loader, scaler, losses, metrics):
    model.train()

    for idx, (img, mask) in enumerate(train_loader):
        # move data and targets to device(gpu or cpu)
        img = img.float().to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.cuda.amp.autocast():
            # making prediction
            pred = model(img)

            # calculate loss and dice coeficient and append it to losses and metrics
            Loss = criterion(pred, mask)
            losses.append(Loss.item())
            dice_coef = calculate_dice_coefficient(predicted_tensor=pred, ground_truth_tensor=mask)
            print(f"loss: {Loss.item()}")
            print(f"dice coeficient: {dice_coef}")
            metrics.append(dice_coef)

        # backward
        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()



def main():
    # if directory model exists than create this
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)


    # define model and move it to device(gpu or cpu)
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    model.to(DEVICE)

    # Create the ShipDataset instance
    ship_dataset = ShipDataset(root_dir=ROOT_DIR_TRAIN, csv_file=CSV_FILE)

    # Determine the size of the train, val, and test datasets based on the specified ratio.
    total_samples = len(ship_dataset)
    train_ratio = 0.6
    val_ratio = 0.2

    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # SSplit the dataset into train, val, and test subsets using the torch.utils.data.random_split function.
    train_dataset, remaining_dataset = torch.utils.data.random_split(ship_dataset,
                                                                     [train_size, total_samples - train_size])
    val_dataset, test_dataset = torch.utils.data.random_split(remaining_dataset, [val_size, test_size])

    # Create the DataLoader objects for each dataset to enable easy batching during training and evaluation.
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_test_collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_test_collate_fn)

    # checking whether the model needs to be retrained
    if LOAD_MODEL:
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            )
        model.to(DEVICE)
        model.load_state_dict(torch.load(PATH_TO_MODEL))

    # define loss function
    criterion = SoftBCEWithLogitsLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    # define Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}')

        losses, metrics =[], []
        train_loop(model, criterion, optimizer, train_loader, scaler, losses, metrics)


if __name__ == '__main__':
    main()