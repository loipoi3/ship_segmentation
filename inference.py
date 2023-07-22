from config import DEVICE, PATH_TO_MODEL
import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
    )
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.to(DEVICE)
model.eval()

pil_img = Image.open('./airbus-ship-detection/test_v2/00c3db267.jpg')
img = torch.tensor(np.array(pil_img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
pred = model(img)
for i in pred:
    for j in i:
        for k in j:
            print(k)