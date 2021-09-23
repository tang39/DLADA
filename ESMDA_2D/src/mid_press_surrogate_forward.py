from __future__ import print_function, division
from WideResNet import WideResNet
import torch
import torch.nn as nn
import numpy as np
import cv2


device = torch.device('cpu')


class Dataset():
    def __init__(self, data, labels, size=64, transform=None):
        self.data, self.labels = data, labels
        self.height = size
        self.width = size
        self.transform = transform

    def __getitem__(self, index):
        pixels = cv2.resize(self.data[index], dsize=(self.height, self.width))
        pixels = (torch.from_numpy(pixels).view(-1, self.height, self.width)).float()
        return pixels, self.labels[index]

    def __len__(self):
        return len(self.labels)


def load_resnet(file_name, use_gpu=0):
  print('Loading saved ResNet ...')
  resnet = WideResNet(depth=16, num_classes=1, widen_factor=1, drop_rate=0.4, input_chanel=2)
  if use_gpu == 1:
    print('use gpu ...')
    resnet.load_state_dict(torch.load(file_name))
  else:
    resnet.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
  resnet.to(device)
  for aa in resnet.modules():
    if isinstance(aa, nn.BatchNorm2d):
        aa.momentum = 1e-2
  return resnet


def mid_pressure_prediction(perm, ti, use_gpu=0):
  print('load model')
  resnet = load_resnet('./input_model/mid_pressure.mdl', use_gpu)
  n_total_params = sum(p.numel() for p in resnet.parameters())
  print('resnet total parameters: ', n_total_params)

  print('evaluate mid p')
  nr, nx, ny = perm.shape
  ti = ti/10
  nt = len(ti)
  maplength = 211
  cropstart = 0
  k_t = np.log10(perm)
  max_k = -9.958663123413906
  min_k = -16.685844772309014
  max_p = 16.718127393830926
  min_p = 12.894143541141064

  k_t_scale = (k_t - min_k) / (max_k - min_k)
  ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
  ktscale = np.repeat(ktscale[:, None, :, :], nt, axis=1)
  ktscale = np.reshape(ktscale, (-1, maplength, maplength))
  del k_t, k_t_scale

  time = np.zeros((nr, nt, nx, ny), dtype=np.float32)
  for i in range(nt):
    time[:, i, :, :] = ti[i]

  time_scale = time[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
  time_scale = np.reshape(time_scale, (-1, maplength, maplength))
  del time

  data = np.zeros((nr*nt, maplength, maplength, 2), dtype=np.float32)
  data[:, :, :, 0] = ktscale
  del ktscale
  data[:, :, :, 1] = time_scale
  del time_scale

  labels = np.zeros((nr*nt, 1))
  dataset = Dataset(data, labels)
  test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=64)
  output_t = []
  resnet.eval()
  for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # forward
    with torch.no_grad():
        outputs = resnet(inputs)
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()
        output_t = np.append(output_t, outputs)

  outputs = np.array(output_t)
  mid_p = outputs.reshape((nr, nt))
  mid_p = mid_p * (max_p - min_p) + min_p

  return mid_p
