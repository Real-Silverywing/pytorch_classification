# -*- coding:utf-8 -*-
# @time :2019.03.15
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
import matplotlib.pyplot as plt
from data import tta_test_transform, get_test_transform
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id ,true_id = [], [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        if _id[i].startswith('n'):
            true_id.append(cfg.Norm_label)
        elif _id[i].startswith('p'):
            true_id.append(cfg.Polyp_label)
        elif _id[i].startswith('s'):
            true_id.append(cfg.Swell_label)
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list,true_id


def tta_predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img1 = Image.open(img_path).convert('RGB')
        # print(type(img))
        pred = []
        for i in range(8):
            img = tta_test_transform(size=cfg.INPUT_SIZE)(img1).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred.append(prediction)
        res = Counter(pred).most_common(1)[0][0]
        pred_list.append(res)
    return _id, pred_list


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    #plt.show()
    plt.close()


if __name__ == "__main__":
    classes = ['normal', 'polyp', 'cysts']
    kappa_list=[]
    PREDICT_MODEL_NAME = 'resnet50'

    #循环跑每个epoch结果range(最大）
    for zongshu in range(50):
        #顺序0 1 2

        PREDICT_EPOCH= zongshu+1
        TRAINED_MODEL = cfg.BASE + '\weights\{}\epoch_{}.pth'.format(PREDICT_MODEL_NAME, PREDICT_EPOCH)

        ##训练完成，要使用predict.py验证时，权重文件的保存路径,默认保存在trained_model下


        trained_model = TRAINED_MODEL
        model_name = cfg.model_name
        with open(cfg.TEST_LABEL_DIR,  'r')as f:
            imgs = f.readlines()

        # _id, pred_list = tta_predict(trained_model)
        print(PREDICT_EPOCH)
        _id, pred_list,true_label = predict(trained_model)
        cm = confusion_matrix(true_label,pred_list)
        kappa= cohen_kappa_score(true_label,pred_list)
        kappa_list.append(kappa)
        plot_confusion_matrix(cm, os.path.join(cfg.SAVE_FOLDER , PREDICT_MODEL_NAME , 'Confusion Matrix_{}-{}epoch-{}.png'
                          .format(PREDICT_MODEL_NAME,PREDICT_EPOCH,cfg.INPUT_SIZE)), title='Confusion Matrix \n Kappa={}'.format(kappa))

        # submission = pd.DataFrame({"ID": _id, "Label": pred_list,"True Label":true_label})
        # submission.to_csv(os.path.join(cfg.SAVE_FOLDER , PREDICT_MODEL_NAME , '测试结果_{}-{}epoch-{}.csv'
        #                   .format(PREDICT_MODEL_NAME,PREDICT_EPOCH,cfg.INPUT_SIZE)), index=True, header=True)

    #range+1才对，不然少一个
    xaxis = range(1,len(kappa_list)+1)
    yaxis = kappa_list
    plt.figure(figsize=(16, 9), dpi=100)

    plt.ylabel('Kappa')
    plt.xlabel('epoch')
    plt.title('Kappa vs Epoch')
    plt.plot(xaxis, yaxis, '.-')
    plt.savefig(os.path.join(cfg.SAVE_FOLDER , PREDICT_MODEL_NAME , 'Kappa系数_{}-{}epoch-{}.png'
                          .format(PREDICT_MODEL_NAME,PREDICT_EPOCH,cfg.INPUT_SIZE)))