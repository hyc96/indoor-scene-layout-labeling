import sys
sys.path.append("../Bounding_box/box_layout")
import data_transforms as transforms
from model import PreTrainedResNet
import torch
from PIL import Image
import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

IM_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input/')

def layout(img_name):
    model = PreTrainedResNet(False, num_classes=4)
    path = os.path.join(scriptdir,ckpt_name)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    unorm = UnNormalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    
    img = Image.open(IM_PATH + img_name)
    img = img.resize((IM_SIZE,IM_SIZE))
    tensor = transform(img)[0]
    tensor = tensor.view(1,3,IM_SIZE,IM_SIZE)
    output = model(tensor)['out']
    _, pred = torch.max(output, 1)
    show_tensor(pred)

def show_tensor(tensor):
    tensor = tensor.squeeze().numpy()
    print(np.max(tensor))
    print(np.min(tensor))
    cmap = ListedColormap(['b', 'c', 'y', 'r'])
    plt.imshow(tensor, cmap=cmap)
    plt.show()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
