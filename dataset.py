from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch


class ShipDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)


    def _rle_to_mask(self, rle, width, height):
        mask = np.zeros(width * height, dtype=np.uint8)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[::2]
        lengths = array[1::2]
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
        return mask.reshape((height, width), order='F')


    def _load_masks(self, image_name):
        image_row = self.df.loc[self.df['ImageId'] == image_name]
        if not pd.isnull(image_row.iloc[0]['EncodedPixels']):
            width, height = Image.open(os.path.join(self.root_dir, image_name)).size
            temp_masks = []
            for rle in image_row['EncodedPixels']:
                mask = self._rle_to_mask(rle, width, height)
                temp_masks.append(mask)
            masks = np.sum(temp_masks, axis=0)

            return masks / 1.0
        else:
            # Return an empty mask if there are no positive masks for the image
            return np.zeros((768, 768)) / 1.0


    def _load_images(self, idx):
        # Check if the folder path exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"The folder path '{self.root_dir}' does not exist.")

        if os.path.isfile(os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])):
            # Open the image file using PIL with the 'with' statement
            with Image.open(os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])) as image:
                # copy image
                image = image.copy()

        return image


    def __len__(self):
        return len(os.listdir(self.root_dir))


    def __getitem__(self, index):
        image = np.array(self._load_images(idx=index))
        image = image.astype(np.float32) / 255.0
        image_name = os.listdir(self.root_dir)[index]
        mask = np.array(torch.tensor(self._load_masks(image_name=image_name)).repeat(1, 1, 1).float().permute(1, 2, 0)) / 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask'].permute(2, 0, 1)
        else:
            image = torch.tensor(image).permute(2, 0, 1)
            mask = torch.tensor(mask).permute(2, 0, 1)

        return image, mask