import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import logging
import cv2
import time
import sys
sys.path.append("../Bounding_box/box_layout")
import data_transforms as transforms
from model import PreTrainedResNet
import dataset
from params import *

logging.basicConfig(format = '[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def load_data():
    train_transform = transforms.Compose(
        [transforms.RandomCrop(IM_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose(
        [transforms.RandomCrop(IM_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    trainset = dataset.LSUN(DATADIR, 'train', train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    print("Train set size: "+str(len(trainset)))

    valset = dataset.LSUN(DATADIR, 'val', val_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    print("Val set size: "+str(len(valset)))
    return trainloader, valloader

def train():
    start = time.time()
    trainloader, valloader = load_data()
    # test(trainloader)
    model = PreTrainedResNet(FEATURE_EXTRACTING, num_classes=4)
    # show_network(model)
    if GPU: model = model.cuda()
    criterion = nn.NLLLoss(ignore_index=255)
    if GPU: criterion.cuda()
    if FEATURE_EXTRACTING:
        optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE*10, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD([{'params':model.optim_base_parameters(), 'lr':LEARNING_RATE},
                                {'params':model.optim_seg_parameters(),'lr':LEARNING_RATE*10}],
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    #learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(NUM_EPOCHS * 1), eta_min = 0.0001)

    #to resume learning:
    start_ep = 0
    for epoch in range(start_ep, NUM_EPOCHS):
        base_lr, lr = scheduler.get_lr()
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train_helper(trainloader, optimizer, model, criterion)
        # scheduler.step()
        validate(valloader, optimizer, model, criterion)

        filename = str(epoch) + '.pth.tar'
        save_path = os.path.join(OUTPUT_DIR,filename)
        torch.save(model.state_dict(), save_path)
    end = time.time()
    print("time: ", end - start)

def train_helper(trainloader, optimizer, model, criterion):
    model.train()
    for i, (img, target) in enumerate(trainloader):
        # transform target into more pixelized layout image
        # not needed atm, res101 interpolates output


        small_target = torch.zeros(int(target.size(0)) , int(target.size(1)/8) , int(target.size(2)/8))
        for index in range(target.size(0)):
            temp = target[index,:,:]
            temp = cv2.resize(temp.numpy(),(int(target.size(1)/8) , int(target.size(2)/8)), interpolation=cv2.INTER_NEAREST)
            temp = torch.Tensor(temp)
            small_target[index,:,:] = temp
        target = small_target
        target = target.long()
        if GPU:
            target = target.cuda(async=True)
            img = img.cuda()
        input_var = Variable(img)
        target_var = Variable(target)
        # print("here")
        # print(img.size())
        outputs = model(input_var)
        output = outputs['out']
        aux_output = outputs['aux']
        # print(output.size())
        # print(aux_output.size())       
        loss1 = criterion(output,target_var)
        loss2 = criterion(aux_output,target_var)
        loss = loss1 + loss2
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Loss: ", loss.item())

def validate(valloader, optimizer, model, criterion):
    epoch_loss = 0.0
    model.eval()
    for i, (img, target) in enumerate(valloader):
        # transform target into more pixelized layout image
        # not needed atm, res101 interpolates output

        small_target = torch.zeros(int(target.size(0)) , int(target.size(1)/8) , int(target.size(2)/8))
        for index in range(target.size(0)):
            temp = target[index,:,:]
            temp = cv2.resize(temp.numpy(),(int(target.size(1)/8) , int(target.size(2)/8)), interpolation=cv2.INTER_NEAREST)
            temp = torch.Tensor(temp)
            small_target[index,:,:] = temp
        target = small_target
        target = target.long()
        with torch.no_grad():
            if GPU:
                target = target.cuda(async=True)
                img = img.cuda()
            input_var = Variable(img)
            target_var = Variable(target)
            
            outputs = model(input_var)
            output = outputs['out']
            aux_output = outputs['aux']

            loss1 = criterion(output,target_var)
            loss2 = criterion(aux_output,target_var)
            loss = loss1 + loss2
            epoch_loss += loss.item()
    print(epoch_loss)

def show_network(model):
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is:")
        print(child)
        child_counter += 1
    # print(model.fcn.backbone)

def test(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(labels.size())
    showtensor(labels)

def showtensor(tensor):
    x = torch.narrow(tensor,0,0,1)
    plt.figure()
    plt.imshow(x.squeeze().numpy())
    plt.show()

def save_checkpoint(self, state):
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.get_model_name() + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
