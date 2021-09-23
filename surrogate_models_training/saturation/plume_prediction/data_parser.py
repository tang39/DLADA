import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize


#################  Load Data  ########################
print('loading data...')

datafilename = '10800_K_Press_Sat.npz'
dirName = '/p/lustre2/tang39/SMART/data_generator/saturation_data_generator/'
filepath = dirName + datafilename

data = np.load(filepath, allow_pickle=True)
kHydro = data['kHydro']
saturation = data['Saturation']
poro_scale_factor = data['poro_scale_factor']

print('shape of k: ', kHydro.shape)
print('shape of saturation: ', saturation.shape)
print('shape of poro_scale_factor:', poro_scale_factor.shape)

#################  Data Preprocess ########################
nr, nt, nx, ny = saturation.shape
train_nr = 1000
val_nr = 400
test_nr = 400
maplength = 200
cropstart = 6

time = np.arange(1, 11, 1)/10
time = np.repeat(time[None, :], nr, axis=0)
print('time shape: ', time.shape)


porosity = (kHydro / (1e-15) / 0.0009)**(1 / 4.0001) / 100 * \
    poro_scale_factor[:, np.newaxis, np.newaxis]
gas_volume = saturation[:, 4, :, :] * porosity
well_1 = np.sum(gas_volume[:, 0:105, 0:105], axis=(1, 2))
well_2 = np.sum(gas_volume[:, 0:105, 105:], axis=(1, 2))
well_3 = np.sum(gas_volume[:, 105:, 0:105], axis=(1, 2))
well_4 = np.sum(gas_volume[:, 105:, 105:], axis=(1, 2))
well_sum = well_1 + well_2 + well_3 + well_4
del porosity, gas_volume

ratio_map = np.zeros(saturation.shape, dtype=np.float32)
ratio_map[:, :, 71, 71] = well_1[:, None]/well_sum[:, None]/poro_scale_factor[:, None] * time
ratio_map[:, :, 71, 141] = well_2[:, None]/well_sum[:, None]/poro_scale_factor[:, None] * time
ratio_map[:, :, 141, 71] = well_3[:, None]/well_sum[:, None]/poro_scale_factor[:, None] * time
ratio_map[:, :, 141, 141] = well_4[:, None]/well_sum[:, None]/poro_scale_factor[:, None] * time

ratio_map_scale = ratio_map[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
ratio_map_scale = np.reshape(ratio_map_scale, (-1, maplength, maplength))
print('injection_ratio_shape:', ratio_map_scale.shape)
del ratio_map

k_t = np.log10(kHydro)
max_k = np.max(k_t[:train_nr + val_nr, ...])
min_k = np.min(k_t[:train_nr + val_nr, ...])
k_t_scale = (k_t - min_k) / (max_k - min_k)
print('max_k, min_k ', max_k, min_k)

ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
ktscale = np.repeat(ktscale[:, None, :, :], nt, axis=1)
ktscale = np.reshape(ktscale, (-1, maplength, maplength))
print('k_input_shape:', ktscale.shape)
del k_t_scale, k_t

saturation[saturation <= 0.1] = 0
saturation[saturation > 0.1] = 1
sat_t_reshape = saturation[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
sat_t_reshape = np.reshape(sat_t_reshape, (-1, maplength, maplength))
print('saturation_shape', sat_t_reshape.shape)

sat_t_reshape[sat_t_reshape <= 0.1] = 0
sat_t_reshape[sat_t_reshape > 0.1] = 1
del saturation

X = np.zeros((nr*nt, maplength, maplength, 2), dtype=np.float32)
X[:, :, :, 0] = ktscale
del ktscale
X[:, :, :, 1] = ratio_map_scale
del ratio_map_scale
Y = sat_t_reshape[..., None]

train_x = X[:train_nr*nt, ...]
train_y = Y[:train_nr*nt, ...]

val_x = X[train_nr*nt:(train_nr + val_nr)*nt, ...]
val_y = Y[train_nr*nt:(train_nr + val_nr)*nt, ...]

test_x = X[-test_nr*nt:, ...]
test_y = Y[-test_nr*nt:, ...]

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('val_x shape is ', val_x.shape)
print('test_x shape is ', test_x.shape)

np.savez(r'Datatemp.npz', train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y)
