from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from npy_data_loader import *
from models import *
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

use_cuda = True if torch.cuda.is_available() else False 
if use_cuda:
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

print('Loading SEM data...')


image_size = 64

SEM_test_dataset = create_SEM_test_dataset(size=image_size)
batch_size = 64
test_loader = torch.utils.data.DataLoader(SEM_test_dataset, shuffle=False, batch_size=batch_size)

epochs = 300
droprate = 0.4
widenfactor = 1
depth = 16
num_classes = 1

# Load my model
file_name = str(image_size)+'_deep{0}_epochs{1}_wide{2}_drop{3}.mdl'.format(depth, epochs, widenfactor, droprate)

print('Loading saved ResNet ...')
resnet = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widenfactor, drop_rate=droprate)
print(resnet)
resnet.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
resnet.to(device)

for aa in resnet.modules():
    if isinstance(aa, nn.BatchNorm2d):
        aa.momentum = 1e-2

# Evaluation
print('Start evaluating...')
output_t = []
label_t = []
resnet.eval()   # Set model to evaluate mode
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # forward
    with torch.no_grad():
        outputs = resnet(inputs)
        labels = labels.float()
        labels = labels.cpu()
        outputs = outputs.cpu()
        labels = labels.detach().numpy()
        label_t = np.append(label_t, labels)
        outputs = outputs.detach().numpy()
        output_t = np.append(output_t, outputs)

outputs = np.array(output_t)
labels = np.array(label_t)
fig = plt.figure()
plt.plot([0, 1], [0, 1], 'r')
plt.scatter(labels, outputs)
r_square = r2_score(labels, outputs)
rmse = np.sqrt(mean_squared_error(labels, outputs))
plt.title('$R^2$= %.4f, RMSE= %.4f' % (r_square, rmse), fontsize=12)
fig.savefig('test'+str(image_size)+'_deep{0}_epochs{1}_wide{2}_drop{3}.png'.format(depth, epochs, widenfactor, droprate), dpi=100)
plt.close()

