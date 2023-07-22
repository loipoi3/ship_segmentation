from torch.utils.data import DataLoader
import os
import segmentation_models_pytorch as smp
import torch
from config import (TRANSFORM_TRAIN,
                    TRANSFORM_VAL_TEST,
                    DEVICE,
                    SAVED_MODEL_PATH,
                    ROOT_DIR_TRAIN,
                    CSV_FILE,
                    BATCH_SIZE,
                    LOAD_MODEL,
                    PATH_TO_MODEL,
                    LEARNING_RATE,
                    NUM_EPOCHS)
from dataset import ShipDataset
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss


# Define a collate function for the training dataset
def train_collate_fn(batch):
    images, masks = zip(*batch)
    transformed_images = []
    transformed_masks = []
    for idx in range(len(images)):
        augmented = TRANSFORM_TRAIN(image=images[idx].numpy(),
                                    mask=masks[idx].permute(1, 2, 0).numpy())
        transformed_images.append(augmented['image'])
        transformed_masks.append(augmented['mask'].permute(2, 0, 1))
    return torch.stack(transformed_images), torch.stack(transformed_masks)


# Define a collate function for the validation and test datasets
def val_test_collate_fn(batch):
    images, masks = zip(*batch)
    transformed_images = []
    transformed_masks = []
    for idx in range(len(images)):
        augmented = TRANSFORM_VAL_TEST(image=images[idx].numpy(),
                                       mask=masks[idx].permute(1, 2, 0).numpy())
        transformed_images.append(augmented['image'])
        transformed_masks.append(augmented['mask'].permute(2, 0, 1))
    return torch.stack(transformed_images), torch.stack(transformed_masks)


def train_loop(model, criterion, optimizer, loader, scaler):
    model.train()

    for idx, (img, mask) in enumerate(loader):
        # move data and targets to device(gpu or cpu)
        img = img.float().to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.cuda.amp.autocast():
            # making prediction
            pred = model(img)
            print(pred)
            # calculate loss and dice coeficient and append it to losses and metrics
            Loss = criterion(pred, mask)
            pred = torch.sigmoid(pred)
            if idx % 10 == 0:
                tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, mask.to(torch.int64), mode='binary', num_classes=1)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
                print(f"train_loss: {Loss.item()}, iou_score: {iou_score}, f1_score: {f1_score}, f2_score: {f2_score}, "
                      f"accuracy: {accuracy}, recall:{recall}")

        # backward
        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()


def val_loop(model, criterion, loader):
    model.eval()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            # move data and targets to device(gpu or cpu)
            img = img.float().to(DEVICE)
            mask = mask.to(DEVICE)

            # making prediction
            pred = model(img)

            # calculate loss and dice coefficient and append it to losses and metrics
            Loss = criterion(pred, mask)
            pred = torch.sigmoid(pred)
            if idx % 10 == 0:
                tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, mask.to(torch.int64), mode='binary', num_classes=1)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
                print(f"val_loss: {Loss.item()}, iou_score: {iou_score}, f1_score: {f1_score}, f2_score: {f2_score}, "
                      f"accuracy: {accuracy}, recall:{recall}")


def test_loop(model, criterion, loader):
    model.eval()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            # move data and targets to device(gpu or cpu)
            img = img.float().to(DEVICE)
            mask = mask.to(DEVICE)

            # making prediction
            pred = model(img)

            # calculate loss and dice coefficient and append it to losses and metrics
            Loss = criterion(pred, mask)
            pred = torch.sigmoid(pred)
            if idx % 10 == 0:
                tp, fp, fn, tn = smp.metrics.get_stats(pred > 0.5, mask.to(torch.int64), mode='binary', num_classes=1)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
                print(f"test_loss: {Loss.item()}, iou_score: {iou_score}, f1_score: {f1_score}, f2_score: {f2_score}, "
                      f"accuracy: {accuracy}, recall:{recall}")


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
        activation=None
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

    # Create optimizer only for trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}')

        train_loop(model, criterion, optimizer, train_loader, scaler)
        val_loop(model, criterion, val_loader)
        test_loop(model, criterion, test_loader)

        # save model
        torch.save(model.state_dict(), SAVED_MODEL_PATH + f'model{epoch + 1}.pth')


if __name__ == '__main__':
    main()