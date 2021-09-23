#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tang39
"""

import time
import numpy as np
import h5py
import os
import sys
sys.path.append("./src/")
from pca import PCA
from utils import *
from sat_surrogate_forward import saturation_prediction
from mid_press_surrogate_forward import mid_pressure_prediction

use_gpu = 0
random_seed = 80385
output_dir = os.getcwd()
os.makedirs('%s/data' % (output_dir), exist_ok=True)
os.makedirs('%s/figures' % (output_dir), exist_ok=True)
start = time.perf_counter()
color_map = plt.cm.get_cmap('Reds')
reversed_color_map = color_map.reversed()
rng = np.random.RandomState(random_seed)


# ###############  Functions for large dataset evaluation #######################
def pressure_prediction_block(xi_rec, pca_model, time_list, use_gpu=0, block_size=1000, nx=211, ny=211):
    nmap = xi_rec.shape[1]
    dim = xi_rec.shape[0]
    n_blocks, remainder = divmod(nmap, block_size)

    for i in range(0, n_blocks):
        time_a = time.perf_counter()
        m_pca_rec = pca_model.generate_pca_realization(xi_rec[:, i*block_size:(i+1)*block_size], dim=dim).T
        m_pca_rec = m_pca_rec.reshape((block_size, nx, ny))
        perm = 10**m_pca_rec
        p = mid_pressure_prediction(perm, time_list, use_gpu=use_gpu)
        time_b = time.perf_counter()
        print('{0}: It takes {1} s to evaluate pressure for {2} maps'.format(i, (time_b-time_a), block_size))
        if i == 0:
            pressure = p
        else:
            pressure = np.append(pressure, p, axis=0)

    if remainder > 0:
        time_a = time.perf_counter()
        m_pca_rec = pca_model.generate_pca_realization(xi_rec[:, n_blocks*block_size:], dim=dim).T
        m_pca_rec = m_pca_rec.reshape((remainder, nx, ny))
        perm = 10**m_pca_rec
        p = mid_pressure_prediction(perm, time_list, use_gpu=use_gpu)
        time_b = time.perf_counter()
        print('{0}: It takes {1} s to evaluate pressure for {2} maps'.format(i, (time_b-time_a), remainder))
        pressure = np.append(pressure, p, axis=0)

    print('pressure shape: ', pressure.shape)
    return pressure


def saturation_prediction_block(xi_rec, pca_model, poro_scale_factor, time_list, use_gpu=0, block_size=1000, nx=211, ny=211):
    nmap = xi_rec.shape[1]
    dim = xi_rec.shape[0]
    n_blocks, remainder = divmod(nmap, block_size)

    for i in range(0, n_blocks):
        time_a = time.perf_counter()
        m_pca_rec = pca_model.generate_pca_realization(xi_rec[:, i*block_size:(i+1)*block_size], dim=dim).T
        m_pca_rec = m_pca_rec.reshape((block_size, nx, ny))
        perm = 10**m_pca_rec
        sat, _, = saturation_prediction(perm, poro_scale_factor[i*block_size:(i+1)*block_size, ...], time_list, use_gpu=use_gpu)
        time_b = time.perf_counter()
        print('{0}: It takes {1} s to evaluate saturation for {2} maps'.format(i, (time_b-time_a), block_size))
        if i == 0:
            saturation = sat
        else:
            saturation = np.append(saturation, sat, axis=0)

    if remainder > 0:
        time_a = time.perf_counter()
        m_pca_rec = pca_model.generate_pca_realization(xi_rec[:, n_blocks*block_size:], dim=dim).T
        m_pca_rec = m_pca_rec.reshape((remainder, nx, ny))
        perm = 10**m_pca_rec
        sat, _, = saturation_prediction(perm, poro_scale_factor[n_blocks*block_size:, ...], time_list, use_gpu=use_gpu)
        time_b = time.perf_counter()
        print('{0}: It takes {1} s to evaluate pressure for {2} maps'.format(i, (time_b-time_a), remainder))
        saturation = np.append(saturation, sat, axis=0)

    print('saturation shape: ', saturation.shape)
    return saturation


# ###############   True model  #######################
datafilename = 'p50-1.out_tabular_co2_plume.csv'
dirName = './input_data/'
filepath = dirName + datafilename
file = open(filepath, 'r')
file.readline()
plume = []
for line in file:
    data = line[:-1].split(',')
    p = int(data[-1])
    plume.append(p)
plume = np.array(plume)
plume = np.reshape(plume, (33, 211 * 211))

true_sat = np.reshape(plume[-6, :], (211, 211), order='F')
fig = plot_single_map(true_sat, vmin=0, vmax=1, cmap=reversed_color_map)
fig.savefig('%s/figures/true_sat_field.png' % (output_dir))

true_sat_last = np.reshape(plume[-1, :], (211, 211), order='F')
fig = plot_single_map(true_sat_last, vmin=0, vmax=1, cmap=reversed_color_map)
fig.savefig('%s/figures/true_sat_field_pred_step.png' % (output_dir))


true_idx = [12, -9, -8, -7, -6]   # observation pressure 1-5year
pred_idx = [12, -9, -8, -7, -6, -5, -4, -3, -2, -1]  # prediction pressure 1-10year

obs_time = np.arange(5) + 1
pred_time = np.arange(10) + 1

datafilename = 'p50-1pressure_mean_MPa.h5'
filepath = dirName + datafilename
with h5py.File(filepath, 'r') as hdf:
    press = hdf.get('pressure')
    press = np.array(press)

max_p = 18
min_p = 12
obs_well_x = 105
obs_well_y = 105
true_p = press[true_idx, obs_well_x, obs_well_y]
true_p_last = press[pred_idx, obs_well_x, obs_well_y]

# ##################  Prior Assemble   #########################
ensemble_size = 5000
tol = 0.9
nx = 211
ny = 211

with h5py.File('./input_data/fractal_permeability.hdf5', 'r') as hdf:
    kHydro = hdf.get('kHydro')
    kHydro = np.array(kHydro)

Input_data = np.log10(kHydro)
nmap, nx, ny = Input_data.shape
pca_model = PCA(nc=nx * ny, nr=nmap, l=nmap)
pca_model.construct_pca(Input_data.reshape((nmap, nx * ny)).T)
del kHydro

truncate_point, rel_energy = pca_model.princ_component(tol=tol)
dim = truncate_point
xi_rec = rng.normal(0, 1, (dim, ensemble_size))
kmin = -17
kmax = -10
nr = ensemble_size

# For prior permeability fields evaluation
# m_pca_rec = pca_model.generate_pca_realization(xi_rec, dim=dim).T
# m_pca_rec = m_pca_rec.reshape((nr, nx, ny))
# kmin = np.min(m_pca_rec)
# kmax = np.max(m_pca_rec)
# multi_maps_evaluation(m_pca_rec, 'prior_perm', vmin=kmin, vmax=kmax, step=5, numRow=2)
# save_data(10**m_pca_rec, '%s/data/prior_perm.hdf5' % (output_dir), 'permeability')

# ##################  Prior Prediction  #########################

#  Pressure Prediction 
time_list = np.arange(1, 11, 1)
pressure = pressure_prediction_block(xi_rec, pca_model, time_list, use_gpu=use_gpu, block_size=2000)
obs_idx = [0, 1, 2, 3, 4]
obs_time = np.arange(5) + 1
p_pred = pressure[:, obs_idx].T
# filepath = '%s/figures/prior_pressure_pred.png' % (output_dir)
# pressure_uncertainty_plot(pressure.T, true_p_last, time_list, filepath, y_min=min_p, y_max=max_p)
# save_data(pressure, '%s/data/prior_press.hdf5' % (output_dir), 'Pressure')

#  Saturation Prediction 
poro_scale_factor = 0.05 + 0.4*rng.rand(nr)
saturation = saturation_prediction_block(xi_rec, pca_model, poro_scale_factor, np.array([5]), use_gpu=use_gpu, block_size=2000)
saturation[saturation <= 0.1] = 0
saturation[saturation > 0.1] = 1
sat_pred = saturation[:, 0, :, :]
del saturation
# save_data(sat_pred, '%s/data/prior_sat_obs.hdf5' % (output_dir), 'Saturation')

# #######  Parameterization of saturation field #####
resolution = 100
ntheta = 30
obs_dim = ntheta*4
maplength = 35

true_radius = polar_transform(true_sat, resolution=resolution, obs_dim=ntheta, maplength=maplength)
true_radius_pred = polar_transform(true_sat_last, resolution=resolution, obs_dim=ntheta, maplength=maplength)
radius_pool = get_radius(sat_pred, resolution=resolution, obs_dim=ntheta, maplength=maplength)

# #############  Parameterization of saturation field #############
obs_sigma = 0.1 * np.ones((obs_dim, ))
press_sigma = 0.02 * np.ones((len(obs_idx), ))
obs_sigma = np.append(obs_sigma, [press_sigma])
obsVariance = obs_sigma ** 2
obsVariance = np.diag(obsVariance)

# ##################  ES-MDA   #########################
n_xi = xi_rec.shape[0]
mPrior = np.concatenate((xi_rec, np.log(poro_scale_factor[None, ...])), axis=0)
dPrior = np.concatenate((radius_pool, p_pred), axis=0)
print('dPrior shape:', dPrior.shape)

obs_xi = np.expand_dims(np.append(true_radius, [true_p]), axis=1)
obs_dim = obs_xi.shape[0]
print('obs_xi shape:', obs_xi.shape)

scale_factor_pool = []
scale_factor_pool.append(poro_scale_factor)
xi_pool = []
xi_pool.append(xi_rec)

Na = 4
alpha = [Na for i in range(Na)]

for i in range(Na):
    print('step ', i + 1)
    obs_pert = np.repeat(obs_xi, nr, axis=1) + \
        np.sqrt(alpha[i]) * np.multiply(obs_sigma[..., None], rng.randn(obs_dim, nr))
    mu_m = np.mean(mPrior, axis=1, keepdims=True)
    mu_d = np.mean(dPrior, axis=1, keepdims=True)
    Cmd = np.matmul(mPrior - mu_m, (dPrior - mu_d).T) / (1 - nr)
    Cdd = np.matmul(dPrior - mu_d, (dPrior - mu_d).T) / (1 - nr)
    Cd = obsVariance
    K = np.matmul(Cmd, np.linalg.inv(Cdd + alpha[i] * Cd))
    mPred = mPrior + np.matmul(K, obs_pert - dPrior)

    # updata dPrior, mPrior
    latent = mPred[:n_xi, :]
    xi_pool.append(latent)
    scale_factor = np.exp(mPred[-1, :])
    scale_factor_pool.append(scale_factor)

    saturation = saturation_prediction_block(latent, pca_model, scale_factor, np.array([5]), use_gpu=use_gpu, block_size=2000)
    sat_pred = saturation[:, 0, :, :]
    del saturation
    sat_pred[sat_pred <= 0.1] = 0
    sat_pred[sat_pred > 0.1] = 1

    p_pred = pressure_prediction_block(latent, pca_model, obs_time, use_gpu=use_gpu, block_size=2000)
    p_pred = p_pred.T
    radius_pool = get_radius(sat_pred, resolution=resolution, obs_dim=ntheta, maplength=maplength)

    dPrior = np.concatenate((radius_pool, p_pred), axis=0)
    mPrior = mPred

filepath = '%s/data/sat_fields_%d_iter.hdf5' % (output_dir, i + 1)
save_data(sat_pred, filepath, 'Saturation')
save_data(scale_factor_pool, '%s/data/porosity_scale_factor.h5' % (output_dir), 'scale_factor')
save_data(xi_pool, '%s/data/latent_variable.h5' % (output_dir), 'latent_variable')
radius_uncertainty_plot(radius_pool, true_radius, '%s/figures/obs_radius_compare.png' % (output_dir))

time_list = np.arange(6, 11, 1)
press_pred = pressure_prediction_block(latent, pca_model, time_list, use_gpu=use_gpu, block_size=2000)
pressure = np.concatenate((p_pred.T, press_pred), axis=1)
print('post pressure shape: ', pressure.shape)
save_data(pressure, '%s/data/post_press.hdf5' % (output_dir), 'Pressure')
filepath = '%s/figures/post_pressure_pred.png' % (output_dir)
pressure_uncertainty_plot(pressure.T, true_p_last, pred_time, filepath, y_min=min_p, y_max=max_p)


saturation = saturation_prediction_block(latent, pca_model, scale_factor, np.array([10]), use_gpu=use_gpu, block_size=2000)
saturation[saturation <= 0.1] = 0
saturation[saturation > 0.1] = 1
save_data(saturation[:, -1, ...], '%s/data/post_sat_pred.hdf5' % (output_dir), 'Saturation')

pred_radius_pool = get_radius(saturation[:, -1, ...], resolution=resolution, obs_dim=ntheta, maplength=maplength)
radius_uncertainty_plot(pred_radius_pool, true_radius_pred, '%s/figures/pred_radius_compare.png' % (output_dir))

time3 = time.perf_counter()
print(f'Finished in {(time3-start)/60} (min)')


