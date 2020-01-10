import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchsummaryX import summary
from torch.nn import functional as F
from collections import OrderedDict

class PreTrainedResNet(nn.Module):

  def __init__(self, feature_extracting, num_classes=4):
    super(PreTrainedResNet, self).__init__()
    self.fcn = models.segmentation.fcn_resnet101(pretrained=True)
    if feature_extracting:
      for param in self.fcn.parameters():
          param.requires_grad = False     # Fine tune whole network if requires_grad = true
                                          # or use slower learning rate for earlier layers
    #modified from models.segmentation.fcn.FCNHead, see
    #https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/fcn.py
    self.fcn.classifier = nn.Sequential(
      nn.Conv2d(2048, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    )

    self.fcn.aux_classifier = nn.Sequential(
      nn.Conv2d(1024, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    )
    # self.fcn.classifier = models.segmentation.fcn.FCNHead(2048,4)
    # self.fcn.aux_classifier = models.segmentation.fcn.FCNHead(1024,4)
    self.softmax = nn.LogSoftmax()
    # summary(self.fcn,torch.zeros((1,3,480,480)))

  def forward(self, x):
      input_shape = x.shape[-2:]
      # contract: features is a dict of tensors
      features = self.fcn.backbone(x)

      result = OrderedDict()
      x = features["out"]
      x = self.fcn.classifier(x)
#       x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
      result["out"] = x

      if self.fcn.aux_classifier is not None:
          x = features["aux"]
          x = self.fcn.aux_classifier(x)
#           x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
          result["aux"] = x
      return result

  def optim_base_parameters(self, memo=None):
    for param in self.fcn.backbone.parameters():
        yield param

  def optim_seg_parameters(self, memo=None):
    for param in self.fcn.classifier.parameters():
        yield param
    for param in self.fcn.aux_classifier.parameters():
        yield param