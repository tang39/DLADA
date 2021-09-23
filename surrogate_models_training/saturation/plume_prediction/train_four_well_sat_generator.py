'''
Created on May 27, 2020

@author: tang39
'''
seed_value = 1024
import math
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
os.makedirs('%s/tmp/checkpoint' % (output_dir), exist_ok=True)

#################  Load Data  ########################
print('loading data...')

data = np.load('Datatemp.npz', allow_pickle=True)
train_x = data['train_x']
train_y = data['train_y']

val_x = data['val_x']
val_y = data['val_y']

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('val_x shape is ', val_x.shape)


class DataGenerator(keras.utils.Sequence):
  def __init__(self, X, Y, batch_size, shuffle=True):
    self.x, self.y = X, Y
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return math.ceil(len(self.x)/self.batch_size)

  def __getitem__(self, idx):
    indexes = self.indexes[idx * self.batch_size:(idx+1) * self.batch_size]
    batch_x = self.x[indexes]
    batch_y = self.y[indexes]
    return np.array(batch_x), np.array(batch_y)

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.x))
    if self.shuffle is True:
      np.random.shuffle(self.indexes)


#################   Build Auto tuner ########################

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#################   Build the Model ########################


load_model = False

if load_model:

    model = keras.models.load_model('/p/lustre2/tang39/SMART/model/sat_plume.h5', compile=False)

else:
    input_shape = train_x.shape[-3:]

    model = para_RUNet(input_shape, seed_value, res_num=6)
    reset_random_seeds(seed_value)

opt = optimizers.Adam(lr=1e-3)
model.compile(loss='mae', optimizer=opt)

model.summary()


####################  Train  Model   ########################

train_model = True
epochs = 100
batch_size = 64
checkpoint_filepath = '%s/tmp/{epoch:06d}.hdf5' % (output_dir)

if train_model:
    output_dir = os.getcwd()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_weights_only=True,
        mode='min',
        period=10)

    lrScheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, cooldown=1, verbose=1, min_lr=1e-6)
    Logger = keras.callbacks.CSVLogger('%s/loss_history.csv' % (output_dir))
    early_stopping = EarlyStopping(
        monitor='val_loss', verbose=1, patience=10, mode='max', restore_best_weights=False)
    # checkpoints = [lrScheduler, Logger, early_stopping]]
    checkpoints = [lrScheduler, Logger, model_checkpoint]
    train_generator = DataGenerator(train_x, train_y, batch_size=batch_size, shuffle=True)
    history = model.fit(x=train_generator, epochs=epochs, validation_data=(val_x, val_y), verbose=2, callbacks=checkpoints)

    print('plot training history...')
    train_hisotry(history, output_dir)


####################  Save Model   ########################

save_model = True

if save_model:
    output_dir = os.getcwd()
    model.save('%s/final_model.h5' % (output_dir))

