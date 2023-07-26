from config import DEVICE, PATH_TO_MODEL
import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
    activation=None
    )
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.to(DEVICE)
model.eval()

img = Image.open('./airbus-ship-detection/train_v2/00b872d8e.jpg')
img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
pred = torch.sigmoid(model(img))

pred_mask = pred.cpu().detach().numpy()[0, 0]

# Apply a threshold to the mask to convert it into binary values (0s and 1s)

binary_mask = pred_mask.astype(np.uint8)

# Overlay the binary mask on the original image to visualize the segmentation
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

# Segmentation Mask
plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.show()