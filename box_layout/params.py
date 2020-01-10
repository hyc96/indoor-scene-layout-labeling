import os

NUM_EPOCHS = 30
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
IM_SIZE = 480
CROP_SIZE = 244
FEATURE_EXTRACTING = False
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GPU = False
OUTPUT_DIR = './output'
scriptdir = os.path.dirname(__file__)
DATADIR = os.path.join(scriptdir,'data/lsun/')
ckpt_name = "ckpt.pth.tar"
