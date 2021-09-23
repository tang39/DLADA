import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

#################  Load Data  ########################
print('loading data...')

datafilename = '10800_K_Press_dynamic.hdf5'
dirName = '/p/lustre2/tang39/SMART/data_generator/pressure_data_generator/'
filepath = dirName + datafilename

with h5py.File(filepath, 'r') as hdf:
    kHydro = hdf.get('kHydro')
    kHydro = np.array(kHydro)
    pressure = hdf.get('Pressure')
    pressure = np.array(pressure)

print('shape of k: ', kHydro.shape)
print('shape of pressure: ', pressure.shape)

#################  Data Preprocess ########################
nr, nt, nx, ny = pressure.shape
train_nr = 1000
val_nr = 400
test_nr = 400
maplength = 211
cropstart = 0


k_t = np.log10(kHydro)
max_k = np.max(k_t[:train_nr + val_nr, ...])
min_k = np.min(k_t[:train_nr + val_nr, ...])
k_t_scale = (k_t - min_k) / (max_k - min_k)
print('max_k, min_k ', max_k, min_k)

ktscale = k_t_scale[:, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
ktscale = np.repeat(ktscale[:, None, :, :], nt, axis=1)
ktscale = np.reshape(ktscale, (-1, maplength, maplength))
print('k_input_shape:', ktscale.shape)
del k_t_scale

p_t = pressure[:, :, 105, 105]/1e6
max_p = np.max(p_t[:train_nr + val_nr, ...])
min_p = np.min(p_t[:train_nr + val_nr, ...])
p_t_scale = (p_t - min_p) / (max_p - min_p)
print('max_p, min_p ', max_p, min_p)
del pressure

p_t_scale = np.reshape(p_t_scale, (nr*nt, ))

time = np.zeros((nr, nt, nx, ny), dtype=np.float32)
for i in range(nt):
  time[:, i, :, :] = (i+1) * 0.1

time_scale = time[:, :, cropstart:cropstart+maplength, cropstart:cropstart+maplength]
time_scale = np.reshape(time_scale, (-1, maplength, maplength))
print(time_scale.shape)
del time

X = np.zeros((nr*nt, maplength, maplength, 2), dtype=np.float32)
X[:, :, :, 0] = ktscale
del ktscale
X[:, :, :, 1] = time_scale
del time_scale
Y = p_t_scale[..., None]

# train_x, val_test_x, train_y, val_test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
# val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=42)

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
