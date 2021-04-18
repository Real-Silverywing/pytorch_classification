# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
import os
home = os.path.expanduser('~')
##数据集的类别
NUM_CLASSES = 2

#训练时batch的大小
BATCH_SIZE = 5

#网络默认输入图像的大小---过大裁剪的！！！
INPUT_SIZE = 224
#训练最多的epoch
MAX_EPOCH = 30
# 使用gpu的数目
GPUS = 0
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3


# 采用的模型名称
model_name = 'resnet101'

from models import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d
MODEL_NAMES = {
    'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'moblienetv2': Mobilenetv2,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet
}


BASE = r'F:\Programming\pytorch_classification\data'


# 训练好模型的保存位置
SAVE_FOLDER = BASE + r'\weights'

#数据集的存放位置
TRAIN_LABEL_DIR =BASE + r'\train.txt'
VAL_LABEL_DIR = BASE + r'\val.txt'
TEST_LABEL_DIR = BASE + r'\test.txt'


##训练完成，要使用predict.py验证时，权重文件的保存路径,默认保存在trained_model下
PREDICT_MODEL_NAME = 'resnet101'
PREDICT_EPOCH = '60'
TRAINED_MODEL = BASE + '\weights\{}\epoch_{}.pth'.format(PREDICT_MODEL_NAME,PREDICT_EPOCH)




