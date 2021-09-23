'''
Created on May 27, 2020

@author: tang39
'''
seed_value = 1024
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
from r_u_net import reset_random_seeds, para_RUNet
from evaluation_plots import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import h5py
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split

print(tf.__version__)
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

output_dir = os.getcwd()
os.makedirs('%s/figures' % (output_dir), exist_ok=True)


#################  Load Data  ########################
print('loading data...')

data = np.load('Datatemp.npz', allow_pickle=True)

test_x = data['test_x']
test_y = data['test_y']

print('test_x shape is ', test_x.shape)

#################   Build Auto tuner ########################

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#################   Build the Model ########################


load_model = True

if load_model:

    model = keras.models.load_model('final_model.h5', compile=False)

else:
    input_shape = test_x.shape[-3:]

    model = para_RUNet(input_shape, seed_value, res_num=6)
    reset_random_seeds(seed_value)
    opt = optimizers.Adam(lr=1e-3)
    model.compile(loss='mae', optimizer=opt)

model.summary()
model.load_weights('%s/tmp/000080.hdf5' % (output_dir))


####################  Evaluate Model   ########################
eval_model = True
test_nr = 400
nt = 10
nx = 211
ny = 211
maplength = 200
cropstart = 6

if eval_model:
    y_pred = model.predict(test_x)
    sat_t_pred = np.zeros((test_nr, nt, nx, ny))
    sat_t_pred[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength] = y_pred[..., 0].reshape((test_nr, nt, maplength, maplength))

    sat_t = np.zeros((test_nr, nt, nx, ny))
    sat_t[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength] = test_y[..., 0].reshape((test_nr, nt, maplength, maplength))

    mse_pool = np.zeros(test_nr)
    for i in range(test_nr):
        mse_pool[i] = np.sqrt(
            np.mean((sat_t[i, 4, ...] - sat_t_pred[i, 4, ...])**2))

    import pandas as pd

    ind_temp = np.zeros([test_nr, 1])
    ind_temp[:, 0] = mse_pool
    df = pd.DataFrame(data=ind_temp, columns=['MSE'])
    df_sort_by_MSE_D = df.sort_values(by='MSE', ascending=False)

    ImagesPerRow = 8
    numRow = 4
    step = 5
    num_case = ImagesPerRow * numRow

    SaveFig_MSE(sat_t[:, 4, ...], num_case, step, df_sort_by_MSE_D, vmin=0, vmax=1, cmap='cool')
    filepath = os.getcwd() + '/figures/' + 'True_sat_MSE_descending.png'
    CreatFig(num_case, ImagesPerRow, numRow, filepath)

    SaveFig_MSE(sat_t_pred[:, 4, ...], num_case, step, df_sort_by_MSE_D,
                vmin=0, vmax=1, cmap='cool')
    filepath = os.getcwd() + '/figures/' + 'Pred_sat_MSE_descending.png'
    CreatFig(num_case, ImagesPerRow, numRow, filepath)

    diff = sat_t_pred[:, 4, ...] - sat_t[:, 4, ...]
    SaveFig_MSE(diff, num_case, step, df_sort_by_MSE_D,
                vmin=np.min(diff), vmax=np.max(diff), cmap='cool')
    filepath = os.getcwd() + '/figures/' + 'Diff_sat_MSE_descending.png'
    CreatFig(num_case, ImagesPerRow, numRow, filepath)

    ImagesPerRow = nt
    numRow = 3
    num_case = ImagesPerRow * numRow

    rank_list = [0, 108]
    time = np.arange(1, 11, 1)

    for i in range(len(rank_list)):
        rank = rank_list[i]
        SaveFig_time(sat_t, sat_t_pred, time, rank, nt, df_sort_by_MSE_D, cmap='cool')
        filepath = os.getcwd() + '/figures/' + 'Test_rank_' + \
            str(rank) + '_descend_MSE' + '.png'
        CreatFig(num_case, ImagesPerRow, numRow, filepath)

#    df.to_csv(os.getcwd() + '/figures/' + 'pred.csv')
