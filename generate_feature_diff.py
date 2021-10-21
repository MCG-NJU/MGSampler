# Code for paper:
# [Title]  - "PAN: Towards Fast Action Recognition via Learning Persistence of Appearance"
# [Author] - Can Zhang, Yuexian Zou, Guang Chen, Lei Gan
# [Github] - https://github.com/zhang-can/PAN-PyTorch

import torch
from torch import nn
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json

device = torch.device('cuda:0')


img_diff = dict()
state_dict = torch.load('/home/zhiyuan/PAN/PAN_PA_something_resnet50_shift8_blockres_avg_segment8_e80.pth.tar')
for k, v in state_dict.items():
    conv_weight = v['module.PA.shallow_conv.weight']
    conv_bias = v['module.PA.shallow_conv.bias']



def read_lines(file):
    arr = []
    with open(file, 'r') as f:
        arr = f.readlines()
    return arr


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.shallow_conv.weight = torch.nn.Parameter(conv_weight)
        self.shallow_conv.bias = torch.nn.Parameter(conv_bias)
        self.n_length = n_length

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        PA = d.view(-1, 1*(self.n_length-1), h, w)
        return PA

class VAP(nn.Module):
    def __init__(self, n_segment, feature_dim, num_class, dropout_ratio):
        super(VAP, self).__init__()
        VAP_level = int(math.log(n_segment, 2))
        print("=> Using {}-level VAP".format(VAP_level))
        self.n_segment = n_segment
        self.VAP_level = VAP_level
        total_timescale = 0
        for i in range(VAP_level):
           timescale = 2**i
           total_timescale += timescale
           setattr(self, "VAP_{}".format(timescale), nn.MaxPool3d((n_segment//timescale,1,1),1,0,(timescale,1,1)))
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.TES = nn.Sequential(
            nn.Linear(total_timescale, total_timescale*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_timescale*4, total_timescale, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.pred = nn.Linear(feature_dim, num_class)
        
        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, d = x.size()
        x = x.view(-1, self.n_segment, d, 1, 1).permute(0,2,1,3,4)
        x = torch.cat(tuple([getattr(self, "VAP_{}".format(2**i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).squeeze(3).permute(0,2,1)
        w = self.GAP(x).squeeze(2)
        w = self.softmax(self.TES(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1)
        x = self.dropout(x)
        x = self.pred(x.view(-1,d))
        return x

"""
data_root = '/data0/zhiyuan/20bn-something-something-v2-frames'
data_root_val = '/data0/zhiyuan/20bn-something-something-v2-frames'
ann_file_train = '/home/zhiyuan/mmaction2/data/sthv2/sthv2_train_list_rawframes.txt'
"""

data_root = '/data0/zhiyuan/20bn-something-something-v1'
data_root_val = '/data0/zhiyuan/20bn-something-something-v1'
ann_file_train = '/data0/zhiyuan/annotations/sthv1_train_list_rawframes.txt'
num = 0
video_list = read_lines(ann_file_train)[40001:44000]
for item in video_list:
    video_name = item.split(" ")[0]
   # video_name = str(6685)
    video_length = int(item.split(" ")[1])
   # video_length =67
    path = data_root + '/' + video_name + '/'
    tmpl = '{:05}.jpg'
    pic1 = path + "00001.jpg"
    tmp = cv2.imread(pic1)
    tmp = tmp.transpose(2, 0, 1)
    tmp = torch.from_numpy(tmp)
    tmp = tmp.unsqueeze(0)
    for i in range(video_length-1):
        name = path + tmpl.format(i + 2)
        pic = cv2.imread(name)
        pic = pic.transpose(2, 0, 1)
        pic = torch.from_numpy(pic)
        pic = pic.unsqueeze(0)
        tmp = torch.cat((tmp, pic), 0)

    PA_module = PA(n_length=video_length)  # adjacent '4' frames are sampled for computing PA
    # shape of x: [N*T*m, 3, H, W]
    tmp = tmp.float()
    tmp = tmp.to(device)
    # shape of PA_out: [N*T, m-1, H, W]
    PA_out = PA_module(tmp)  # torch.Size([40, 3, 224, 224])
    PA_out = PA_out.squeeze(0)  # [42,100,180]
    motion = list()
    for i in range(video_length-1):
        img = PA_out[i, :, :]
       # img_name = "/home/zhiyuan/PAN/img_diff_6685/"+tmpl.format(i+1)
       # plt.imsave(img_name,img.detach().numpy(),cmap='gray')
        plt.imshow(img.cpu().detach().numpy(), cmap='gray')
        motion.append((torch.sum(img)).item())
      # plt.show()
    # 归一化到[0,255]

    motion = np.array(motion)

    motion = np.power(motion, 0.5)
    sum_num = np.sum(motion)
    diff_score = motion / sum_num
    count = 0

    img_diff[video_name] = list()
    for i in range(len(diff_score)):
        count = count + diff_score[i]
        img_diff[video_name].append(count)
    num +=1
    print(num)








fileObject = open('/home/zhiyuan/PAN/conv_diff_sthv1_train_11.json', 'a+')
jsonData = json.dumps(img_diff)
fileObject.write(jsonData)
fileObject.close()





