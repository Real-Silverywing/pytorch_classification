# -*- coding:utf-8 -*-
# @time :2020/8/13
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


import torch
import argparse
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from data import train_dataloader,train_datasets, val_datasets, val_dataloader
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
from sklearn.metrics import cohen_kappa_score

##创建训练模型参数保存的文件夹
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)

#保存训练过程中的数据
Loss_list = []
Accuracy_list = []
Kappa_list=[]
train_acc_list=[]
train_loss_list=[]
train_iter=[]
train_epoch=[]

#将训练参数的cfg.py拷贝到结果里
cfg_path = os.path.join(os.getcwd(),'cfg.py')
cfg_copy_path = os.path.join(save_folder, 'param.py')
shutil.copyfile(cfg_path, cfg_copy_path)

#绘制准确度acc和loss
def plot_accloss(acclist,losslist):
    x1 = range(1, len(acclist) + 1)
    x2 = range(1, len(losslist) + 1)
    y1 = acclist
    y2 = losslist

    plt.figure(figsize=(16, 9), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('Test accuracy vs. iteration')
    plt.ylabel('Test accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. iteration')
    plt.ylabel('Test loss')

    plt.savefig(os.path.join(save_folder, 'accuracy_loss_{}-{}epoch-{}.jpg'
                             .format(cfg.model_name, cfg.MAX_EPOCH-1, cfg.INPUT_SIZE)))
    plt.close()

#绘制结果
#列表,横轴名,纵轴名,标题
def plot_result(list,xlabel,ylabel,title):
    x1 = range(1, len(list) + 1)
    y1 = list
    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(x1, y1, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(save_folder, xlabel+' vs '+ylabel+'_{}-{}epoch-{}.jpg'
                             .format(cfg.model_name, cfg.MAX_EPOCH-1, cfg.INPUT_SIZE)))
    plt.close()

#
def val_in_train():
    model.eval()
    total_correct = 0
    total_val_loss = 0
    total_kappa = 0
    #算kappa有问题,会出现nan现象,会影响平均值计算
    nan_exist=0
    kappa_count=0
    val_iter = iter(val_dataloader)
    max_iter = len(val_dataloader)
    for iteration in range(max_iter):
        # try:
        images, labels = next(val_iter)
        # except:
        #     continue
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
            out = model(images)
            prediction = torch.max(out, 1)[1]

            correct = (prediction == labels).sum()
            total_correct += correct
            val_loss = criterion(out, labels.long())
            total_val_loss+= val_loss.item()
            predict_list=prediction.cpu().numpy().tolist()
            #predict_list.append(prediction.tolist())
            label_list=labels.cpu().numpy().tolist()
            #label_list.append(labels[0].tolist())
            kappa_in_batch = cohen_kappa_score(label_list, predict_list)

            if not kappa_in_batch == kappa_in_batch:#nan类型的特性是自身不等于自身
                print('发现一个nan')
                nan_exist += 1
            else:
                total_kappa += kappa_in_batch
                kappa_count +=1

            print('VALIDATION: Iteration: {}/{}'.format(iteration, max_iter),
                  'ACC: %.6f' %(correct.float()/batch_size),'Loss: %.6f' % (val_loss.item()),'Kappa:%.5f'%kappa_in_batch)
    Accuracy_list.append((total_correct.float()/(len(val_dataloader)* batch_size)))
    Loss_list.append((total_val_loss/(len(val_dataloader)* batch_size)))
    Kappa_list.append((total_kappa/(kappa_count-nan_exist)))
    print('VALIDATION SET: ACC: %.6f'%(total_correct.float()/(len(val_dataloader)* batch_size)),
          'loss: %.3f' %(total_val_loss/(len(val_dataloader)* batch_size)),
          'Kappa:%.4f'%(total_kappa/(kappa_count-nan_exist)))

    #训练完以后绘制结果,存储结果到csv
    if epoch == cfg.MAX_EPOCH:
        plot_accloss(Accuracy_list,Loss_list)
        plot_result(Kappa_list,'epoch','Kappa','Validation Kappa')

        savedata1 = pd.DataFrame({"Val_acc": Accuracy_list, "Val_loss": Loss_list,"Val_kappa": Kappa_list})
        savedata1.to_csv(os.path.join(save_folder, 'val验证变量_{}-{}epoch.csv'
                                      .format(cfg.model_name, cfg.MAX_EPOCH)), index=True, header=True)
        savedata2 = pd.DataFrame(
            {"epoch": train_epoch, "iter": train_iter, "acc": train_acc_list, "loss": train_loss_list})
        savedata2.to_csv(os.path.join(save_folder, 'Train训练变量_{}-{}epoch.csv'
                                      .format(cfg.model_name, cfg.MAX_EPOCH)), index=True, header=True)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

#####build the network model
if not cfg.RESUME_EPOCH:
    print('****** Training {} ****** '.format(cfg.model_name))
    print('****** loading the Imagenet pretrained weights ****** ')
    if not cfg.model_name.startswith('efficientnet'):
        model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)
        # #冻结前边一部分层不训练
        ct = 0
        for child in model.children():
            ct += 1
            # print(child)
            if ct < 7:#以前是7,在resnet上好用,densenet冻结1或者2
                print(child)
                for param in child.parameters():
                    param.requires_grad = False
    else:
        model = cfg.MODEL_NAMES[cfg.model_name](cfg.model_name,num_classes=cfg.NUM_CLASSES)
        # print(model)
        c = 0
        for name, p in model.named_parameters():
            c += 1
            # print(name)
            if c >=700:
                break
            p.requires_grad = False

    # print(model)
if cfg.RESUME_EPOCH:
    print(' ******* Resume training from {}  epoch {} *********'.format(cfg.model_name, cfg.RESUME_EPOCH))
    model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))



##进行多gpu的并行计算
if cfg.GPUS>1:
    print('****** using multiple gpus to training ********')
    model = nn.DataParallel(model,device_ids=list(range(cfg.GPUS)))
else:
    print('****** using single gpu to training ********')
print("...... Initialize the network done!!! .......")

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()


##定义优化器与损失函数
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
# optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                      momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()


lr = cfg.LR

batch_size = cfg.BATCH_SIZE

#每一个epoch含有多少个batch
max_batch = len(train_datasets)//batch_size
epoch_size = len(train_datasets) // batch_size
## 训练max_epoch个epoch
max_iter = cfg.MAX_EPOCH * epoch_size

start_iter = cfg.RESUME_EPOCH * epoch_size


max_iter_end=math.floor(max_iter/10)*10


epoch = cfg.RESUME_EPOCH

# cosine学习率调整
warmup_epoch=5
warmup_steps = warmup_epoch * epoch_size
global_step = 0

# step 学习率调整参数
stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
step_index = 0


for iteration in range(start_iter, max_iter):

    global_step += 1

    ##更新迭代器
    if iteration % epoch_size == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        if epoch > 1:
            val_in_train()


        ###保存模型
        model.train()
        if epoch % cfg.CHECKPOINT_EPOCH == 0 and epoch > 0:
            if cfg.GPUS > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))

    if iteration in stepvalues:
        step_index += 1
    lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)

    ## 调整学习率
    # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
    #                           learning_rate_base=cfg.LR,
    #                           total_steps=max_iter,
    #                           warmup_steps=warmup_steps)


    ## 获取image 和 label
    # try:
    images, labels = next(batch_iterator)
    # except:
    #     continue

    ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels.long())

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()  # loss反向传播
    optimizer.step()  ##梯度更新

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / batch_size

    #训练过程中显示
    if iteration % 10 == 0:
        train_acc_list.append(train_acc)
        train_loss_list.append(loss.item())
        train_iter.append(repr(iteration))
        train_epoch.append(repr(epoch))
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) + 'in batch ACC: %.4f ||' %train_acc + 'LR: %.8f' % (lr))