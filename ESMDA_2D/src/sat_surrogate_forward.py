from __future__ import print_function, division
from tensorflow import keras
from WideResNet import WideResNet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import cv2


use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
  device = torch.device('cuda')
  cudnn.benchmark = True
else:
  device = torch.device('cpu')
torch.cuda.empty_cache()

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


def load_unet(file_name):
  print('Loading saved Unet ...')
  unet = keras.models.load_model(file_name, compile=False)

  return unet


def load_resnet(file_name, use_gpu=0):
  print('Loading saved ResNet ...')
  resnet = WideResNet(depth=16, num_classes=4, widen_factor=2, drop_rate=0.4, input_chanel=1)
  if use_gpu == 1:
    resnet.load_state_dict(torch.load(file_name))
  else:
    resnet.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
  resnet.to(device)
  for aa in resnet.modules():
    if isinstance(aa, nn.BatchNorm2d):
        aa.momentum = 1e-2
  return resnet


def saturation_prediction(perm, poro_scale_factor, ti, use_gpu=0):
  print('load model')
  resnet = load_resnet('./input_model/allocation_ratio_model.mdl', use_gpu)
  n_total_params = sum(p.numel() for p in resnet.parameters())
  print('resnet total parameters: ', n_total_params)
  unet = load_unet('./input_model/sat_plume.h5')
  # unet.summary()

  print('evaluate rate distribution')
  nr, nx, ny = perm.shape
  maplength = 140
  cropstart = 36
  k_t = np.log10(perm)
  max_k = -9.958663123413906
  min_k = -16.685844772309014
  k_t_scale = (k_t - min_k) / (max_k - min_k)
  ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
  ktscale[ktscale < 0] = 0
  ktscale[ktscale > 1] = 1
  del k_t

  resnet.eval()
  data = ktscale[..., None]
  labels = np.zeros((nr, 4))
  dataset = Dataset(data, labels)
  test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=64)
  output_t = []
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
  ratio_pred = outputs.reshape((nr, 4))

  print('evaluate plume shape')
  maplength = 200
  cropstart = 6
  ti = ti/10
  nt = len(ti)
  ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
  ktscale = np.repeat(ktscale[:, None, :, :], nt, axis=1)
  ktscale = np.reshape(ktscale, (-1, maplength, maplength))

  time = np.zeros((nr, nt), dtype=np.float32)
  for i in range(nt):
    time[:, i] = ti[i]

  ratio_map = np.zeros((nr, nt, nx, ny), dtype=np.float32)
  ratio_map[:, :, 71, 71] = ratio_pred[:, 0, np.newaxis]/poro_scale_factor[:, np.newaxis] * time
  ratio_map[:, :, 71, 141] = ratio_pred[:, 1, np.newaxis]/poro_scale_factor[:, np.newaxis] * time
  ratio_map[:, :, 141, 71] = ratio_pred[:, 2, np.newaxis]/poro_scale_factor[:, np.newaxis] * time
  ratio_map[:, :, 141, 141] = ratio_pred[:, 3, np.newaxis]/poro_scale_factor[:, np.newaxis] * time
  ratio_map_scale = ratio_map[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
  ratio_map_scale = np.reshape(ratio_map_scale, (-1, maplength, maplength))

  X = np.zeros((nr*nt, maplength, maplength, 2))
  X[:, :, :, 0] = ktscale
  del ktscale
  X[:, :, :, 1] = ratio_map_scale
  del ratio_map_scale

  y_pred = unet.predict(X)
  sat_t_pred = np.zeros((nr, nt, nx, ny))
  sat_t_pred[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength] = y_pred[..., 0].reshape((nr, nt, maplength, maplength))

  return sat_t_pred, ratio_pred
