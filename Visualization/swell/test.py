

import torch
import torchvision.models as models

# pretrained=True就可以使用预训练的模型
net = models.resnet101(pretrained=False)
pthfile = r'F:\Programming\pytorch_classification-master\data\weights\resnet101\epoch_60.pth'
net.load_state_dict(torch.load(pthfile))
print(net)