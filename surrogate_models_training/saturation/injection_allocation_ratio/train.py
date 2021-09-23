from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
import time
import copy
from npy_data_loader import *
from models import *


use_cuda = True if torch.cuda.is_available() else False 


def train_model(model, data_loaders,  device, optimizer, out_file, num_epochs, file_name):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000
    # criterion = nn.MSELoss() # crossentropy # mae
    criterion = nn.L1Loss()

    for epoch in range(num_epochs + 1):
        print('Epoch {0}/{1}'.format(epoch, num_epochs))
        out_file.write('Epoch {}/{}\n'.format(epoch, num_epochs))
        out_file.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = []
               
            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss.append(loss.item())
            print(labels)
            print(outputs)
            loss = np.mean(running_loss)
            out_file.write('\n{0} Loss: {1}'.format(phase, loss))
            if phase == 'val':
                out_file.write('\n')
                out_file.flush()
                if loss < best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = loss
                    torch.save(best_model_wts, file_name)
             
    time_elapsed = time.time() - since
    out_file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


def main():
    
    image_size = 64
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:0')
    print('Loading SEM data...')
    SEM_train_dataset = create_SEM_train_dataset(image_size)
    SEM_val_dataset = create_SEM_val_dataset(image_size)
    SEM_test_dataset = create_SEM_test_dataset(image_size)
    epochs = 300
    depth = 16
    batch_size = 64
    droprate = 0.4
    widenfactor = 2
    num_classes = 4

    train_loader = torch.utils.data.DataLoader(SEM_train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(SEM_val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(SEM_test_dataset, shuffle=False, batch_size=batch_size)
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    out_file = open(str(image_size)+'_deep{0}_epochs{1}_wide{2}_drop{3}.out'.format(depth, epochs, widenfactor, droprate), 'a+')
    out_file.write('Epochs: {0}\n'.format(epochs))
    file_name = str(image_size)+'_deep{0}_epochs{1}_wide{2}_drop{3}.mdl'.format(depth, epochs, widenfactor, droprate)
    if use_cuda:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print('Loading new ResNet')
    resnet = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widenfactor, drop_rate=droprate)
    resnet.to(device)

    for aa in resnet.modules():
        if isinstance(aa, nn.BatchNorm2d):
            aa.momentum = 1e-2

    optimizer = optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=0)  # lr base is 0.001, wd 0.01

    print('Start training...') 
    train_model(resnet, data_loaders, device, optimizer, out_file, epochs, file_name)
    out_file.close()


main()

