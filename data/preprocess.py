import os
import glob



import sys 
sys.path.append("..") 
import cfg
import random



if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = cfg.BASE + 'test'
    ##写train.txt文件
    txtpath = cfg.BASE
    print(labels)

    if os.path.exists(txtpath + 'train.txt'): os.remove(txtpath + 'train.txt')
    if os.path.exists(txtpath + 'val.txt'): os.remove(txtpath + 'val.txt')
    if os.path.exists(txtpath + 'test.txt'): os.remove(txtpath + 'test.txt')

    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path,label, '*.jpg'))
        #print(imglist)
        random.shuffle(imglist)
        print('类别：{}--->【{}】   读取图像数量：{}'.format(label,str(index),len(imglist)))
        trainlist = imglist[:int(0.8*len(imglist))]
        vallist = imglist[(int(0.8*len(imglist))):]
        print(' Train: {}  Validation: {} '.format(len(trainlist), len(vallist)))
        #可能会出现浪费样本，但是保证val和test不冲突


        with open(txtpath + 'train.txt', 'a')as f:
            for img in trainlist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')


        with open(txtpath + 'val.txt', 'a')as f:
            for img in vallist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')


    imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))

    with open(txtpath + 'test.txt', 'a')as f:
        for img in imglist:
            f.write(img)
            f.write('\n')