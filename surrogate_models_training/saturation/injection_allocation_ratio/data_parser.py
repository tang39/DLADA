import os
import h5py
import numpy as np

#################  Load Data  ########################
print('loading data...')

datafilename = '10800_K_Press_Sat.npz'
dirName = '/p/lustre2/tang39/SMART/data_generator/saturation_data_generator/'
filepath = dirName + datafilename

data = np.load(filepath, allow_pickle=True)
kHydro = data['kHydro']
saturation = data['Saturation']
poro_scale_factor = data['poro_scale_factor']

ind = 4
saturation = saturation[:, ind, ...]

print('shape of k: ', kHydro.shape)  
print('shape of saturation: ', saturation.shape)  
print('shape of poro_scale_factor:', poro_scale_factor.shape)


#################  Data Preprocess ########################
nr, nx, ny = kHydro.shape
train_nr = 1000
val_nr = 400
test_nr = 400
maplength = 140
cropstart = 36
tunit = 60 * 60 * 24

k_t = np.log10(kHydro)
max_k = np.max(k_t[:train_nr + val_nr, ...])
min_k = np.min(k_t[:train_nr + val_nr, ...])
k_t_scale = (k_t - min_k) / (max_k - min_k)
print('max_k, min_k ', max_k, min_k)

ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
print('k_input_shape:', ktscale.shape)

porosity = (kHydro / (1e-15) / 0.0009)**(1 / 4.0001) / 100 * \
    poro_scale_factor[:, np.newaxis, np.newaxis]

gas_volume = saturation * porosity
well_1 = np.sum(gas_volume[:, 0:105, 0:105], axis=(1, 2))
well_2 = np.sum(gas_volume[:, 0:105, 105:], axis=(1, 2))
well_3 = np.sum(gas_volume[:, 105:, 0:105], axis=(1, 2))
well_4 = np.sum(gas_volume[:, 105:, 105:], axis=(1, 2))
well_sum = well_1 + well_2 + well_3 + well_4
ratio = [well_1 / well_sum, well_2 / well_sum,
         well_3 / well_sum, well_4 / well_sum]
ratio = np.array(ratio).T
print(ratio.shape)

X = ktscale[..., None]
train_x = X[:train_nr, ...]
train_y = ratio[:train_nr, ...]

val_x = X[train_nr:train_nr + val_nr, ...]
val_y = ratio[train_nr:train_nr + val_nr, ...]

test_x = X[-test_nr:, ...]
test_y = ratio[-test_nr:, ...]

print('train_x shape is ', train_x.shape)
print('train_y shape is ', train_y.shape)
print('val_x shape is ', val_x.shape)
print('test_x shape is ', test_x.shape)

np.savez(r'Datatemp.npz', train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y)
