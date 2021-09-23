#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:07:28 2020

@author: tang39
"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import polarTransform


def SaveFig(data, num_case, step, vmin=0, vmax=1, cmap='jet'):
    for i in range(num_case):
        export_path = os.getcwd() + '/figures/' + str(i) + '.png'
        rank = i * step
        fig = plt.figure(figsize=(2, 2))
        # im = plt.imshow(data[rank, :, :], interpolation='bicubic',
        #                 cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        im = plt.imshow(data[rank, :, :], cmap=cmap,
                        origin='lower', vmin=vmin, vmax=vmax)
        plt.scatter(71, 71, s=20, facecolors='none', edgecolors='k')
        plt.scatter(71, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 71, s=20, facecolors='none', edgecolors='k')
        plt.axis('off')
        if i == 0:
            fig.colorbar(im, orientation="horizontal",
                         pad=0.03, fraction=0.045)

        fig.savefig(export_path, dpi=100)
        plt.close()


def CreatFig(ImagesPerRow, numRow, filepath):
    num_case = ImagesPerRow * numRow
    images = [Image.open(os.getcwd() + '/figures/' + str(i) + '.png')
              for i in range(num_case)]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_width = int(max_width * 0.9)
    max_heights = max(heights)
    max_heights = int(max_heights * 0.9)
    total_width = max_width * ImagesPerRow
    total_height = max_heights * numRow
    new_im = Image.new('RGB', (total_width, total_height),
                       color=(255, 255, 255))
    x_offset = 0
    y_offset = 0
    for i in range(num_case):
        new_im.paste(images[i], (x_offset, y_offset))
        x_offset += max_width
        if (i + 1) % ImagesPerRow == 0:
            x_offset = 0
            y_offset += max_heights

    new_im.save(filepath)


def plot_single_map(data, vmin, vmax, cmap='jet'):
    fig = plt.figure(figsize=(4, 4))
    # plt.imshow(data, interpolation='bicubic', cmap=cmap,
    #            origin='lower', vmin=vmin, vmax=vmax)
    plt.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(pad=0.03, fraction=0.045)
    plt.scatter(71, 71, s=20, facecolors='none', edgecolors='k')
    plt.scatter(71, 141, s=20, facecolors='none', edgecolors='k')
    plt.scatter(141, 141, s=20, facecolors='none', edgecolors='k')
    plt.scatter(141, 71, s=20, facecolors='none', edgecolors='k')
    plt.axis('off')
    plt.tight_layout()
    return fig


def scatter_plot(data, filepath):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(data, '*')
    plt.xlabel('Porosity Scale Factor', fontsize=12)
    fig.savefig(filepath)
    plt.close()


def contour_plot(data, vmin, vmax, cmap='RdGy'):
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    levels = np.array([0.5, 1, 1.5, 2.5, 5])
    fig = plt.figure(figsize=(4, 4))
    contours = plt.contour(X, Y, data, levels=levels, colors='black')
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f', colors='black')
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap='RdGy')
    plt.axis('off')
    plt.colorbar(pad=0.03, fraction=0.045)
    return fig


def plot_multi_maps(filepath, data, step=1, ImagesPerRow=9, numRow=5, vmin=0, vmax=1, cmap='jet'):
    num_case = ImagesPerRow * numRow
    SaveFig(data, num_case, step, vmin=vmin, vmax=vmax, cmap=cmap)
    CreatFig(ImagesPerRow, numRow, filepath)


def save_perm(data):
    with h5py.File('permeability.hdf5', 'w') as hdf:
        hdf.create_dataset('permeability', data=data)


def save_data(data, filepath, dataname):
    with h5py.File(filepath, 'w') as hdf:
        hdf.create_dataset(dataname, data=data)


def multi_maps_evaluation(data, title, vmin=0, vmax=1, step=5, numRow=2, cmap='jet'):
    output_dir = os.getcwd()
    filepath = output_dir + '/figures/' + title + '_fields.png'
    plot_multi_maps(filepath, data, step=step, ImagesPerRow=9, numRow=numRow,
                    vmin=vmin, vmax=vmax, cmap=cmap)

    std_prior = np.std(data, axis=0)
    # std_min = np.min(std_prior)
    # std_max = np.max(std_prior)
    std_min = 0
    std_max = 1
    fig = plot_single_map(std_prior, vmin=std_min,
                          vmax=std_max, cmap='rainbow')
    filepath = output_dir + '/figures/' + title + '_std.png'
    fig.savefig(filepath)

    mean_prior = np.mean(data, axis=0)
    fig = plot_single_map(mean_prior, vmin=vmin, vmax=vmax, cmap=cmap)
    filepath = output_dir + '/figures/' + title + '_mean.png'
    fig.savefig(filepath)


def polar_transform(sat, resolution=100, obs_dim=30, maplength=35):
    sat_1 = sat[71 - maplength:71 + maplength, 71 - maplength:71 + maplength]
    sat_2 = sat[71 - maplength:71 + maplength, 141 - maplength:141 + maplength]
    sat_3 = sat[141 - maplength:141 + maplength,
                141 - maplength:141 + maplength]
    sat_4 = sat[141 - maplength:141 + maplength, 71 - maplength:71 + maplength]
    center = (maplength, maplength)
    sat_pool = [sat_1, sat_2, sat_3, sat_4]
    true_radius = []
    for i in range(0, 4):
        polar_sat, media = polarTransform.convertToPolarImage(
            sat_pool[i], center=center, order=0, radiusSize=resolution, angleSize=obs_dim)
        radius = [np.amax(np.nonzero(polar_sat[i, :]), initial=0.1)
                  for i in range(0, polar_sat.shape[0])]
        true_radius.append(radius)
    true_radius = np.array(true_radius)
    true_radius = true_radius.flatten()

    return true_radius


def get_radius(sat_sim, resolution=100, obs_dim=30, maplength=35):
    nmap, nx, ny = sat_sim.shape
    radius_pool = []
    for i in range(0, nmap):
        radius = polar_transform(
            sat_sim[i, ...], resolution=100, obs_dim=obs_dim, maplength=35)
        radius_pool.append(radius)
    radius_pool = np.array(radius_pool)
    radius_pool = radius_pool.T
    return radius_pool


def box_plot(scale_factor, n, true_value=0.134):
    output_dir = os.getcwd()
    fig = plt.figure(figsize=(5, 4))
    plt.boxplot(scale_factor, widths=0.5)
    plt.plot([0, n], [true_value, true_value], 'r--')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Porosity scale factor', fontsize=14)
    plt.ylim([0, 0.6])
    plt.xlim([0, n])
    plt.tight_layout()
    fig.savefig('%s/figures/sat_scale_factor_box_plot.png' % (output_dir))
    plt.close()


def p_box_plot(scale_factor, n):
    output_dir = os.getcwd()
    fig = plt.figure(figsize=(5, 4))
    plt.boxplot(scale_factor, widths=0.5)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Porosity scale factor', fontsize=14)
    # plt.ylim([0, np.percentile(scale_factor, 99)])
    plt.ylim([0, 2])
    plt.xlim([0, n])
    plt.tight_layout()
    fig.savefig('%s/figures/p_scale_factor_box_plot.png' % (output_dir))
    plt.close()


def latent_box_plot(scale_factor, n, iter):
    output_dir = os.getcwd()
    fig = plt.figure(figsize=(5, 4))
    plt.boxplot(scale_factor, widths=0.5)
    plt.xlabel('Dimension Rank', fontsize=12)
    plt.ylabel('Latent Variable', fontsize=12)
    plt.ylim([-6, 6])
    plt.xlim([0, n])
    plt.tight_layout()
    fig.savefig('%s/figures/latent_%d_iter.png' % (output_dir, iter))
    plt.close()


def pressure_uncertainty_plot(p_pred, true_p, time, filepath, y_min=12, y_max=18):
    fig = plt.figure()
    p_pred_5 = np.percentile(p_pred, 5, axis=1)
    p_pred_95 = np.percentile(p_pred, 95, axis=1)
    p_pred_mean = np.mean(p_pred, axis=1)

    fig, ax = plt.subplots(1)
    ax.fill_between(time, p_pred_95, p_pred_5, facecolor='green',
                    alpha=0.25, label='90%_credible_interval')
    ax.plot(time, p_pred_mean, color='green', label='pred_mean')
    ax.plot(time, true_p, 'r*', markersize=6, label='true')
    ax.legend()
    ax.set_xlabel('Time(year)', fontsize=14)
    ax.set_ylabel('Pressure(MPa)', fontsize=14)
    ax.set_ylim((y_min, y_max))
    fig.savefig(filepath, dpi=100)
    plt.close()


def prior_post_pressure_uncertainty(prior_press, post_press, obs_time, obs_p, time_list, true_p, filepath, y_min=12, y_max=18):
    fig = plt.figure(figsize=(5, 4))
    prior_pred_1 = np.percentile(prior_press, 5, axis=1)
    prior_pred_99 = np.percentile(prior_press, 95, axis=1)
    post_pred_1 = np.percentile(post_press, 5, axis=1)
    post_pred_99 = np.percentile(post_press, 95, axis=1)
    fig, ax = plt.subplots(1)
    ax.fill_between(time_list, prior_pred_99, prior_pred_1, facecolor='dimgray', alpha=0.25, label='Prior')
    ax.fill_between(time_list, post_pred_99, post_pred_1, facecolor='green', alpha=0.25, label='Posterior')
    ax.plot(obs_time, obs_p, 'ko', label='Observed')
    ax.plot(time_list, true_p, 'r-', label='True')
    ax.set_ylim((y_min, y_max))
    ax.legend(fontsize=12, loc='upper left')
    plt.ylabel('Pressure (MPa)', fontsize=14)
    plt.xlabel('Time (year)', fontsize=14)
    fig.savefig(filepath, dpi=100)
    plt.close()


def pdf_plot(data, xlabel, file_path):
    fig = plt.figure()
    plt.hist(data.flatten(), bins=21, color='b',
             edgecolor='black', linewidth=1, density='True')
    plt.xlabel(xlabel, fontsize=14)
    fig.savefig(file_path, dpi=100, bbox_inches='tight')


def multi_pdf_plot(prior, post, xlabel, file_path, num=50, mean_perm=-13.07, plot_lim=4):
    fig = plt.figure()
    low = min(np.min(prior), np.min(post))
    high = max(np.max(prior), np.max(post))
    bins = np.linspace(low, high, num=50)
    plt.hist(prior.flatten(), bins=bins, color='b',
             density='True', label='prior')
    plt.hist(post.flatten(), bins=bins, color='y',
             alpha=0.5, density='True', label='posterior')
    plt.plot([mean_perm, mean_perm], [0, plot_lim], 'r--', label='3D mean')
    plt.legend()
    plt.xlabel(xlabel, fontsize=14)
    fig.savefig(file_path, dpi=100, bbox_inches='tight')


def plot_ensemble(timestep, pred_p, true_time, true_p, obs_time, obs_p, file_path):
    ensemble_size = pred_p.shape[0]
    fig = plt.figure(figsize=(4, 4))
    for i in range(0, ensemble_size):
        plt.plot(timestep, pred_p[i, :], linewidth=0.1, c='0.8', label='Pred')

    plt.plot(true_time, true_p, 'r-', linewidth=1, label='True')
    plt.plot(obs_time, obs_p, 'ko', label='Obs')
    plt.ylabel('Pressure (MPa)', fontsize=14)
    plt.xlabel('Time (year)', fontsize=14)
    plt.tight_layout()
    fig.savefig(file_path)


def plot_true_pressure(true_time, true_p, obs_time, obs_p, file_path, y_min=12, y_max=18):
    fig = plt.figure(figsize=(4, 3))
    plt.plot(true_time, true_p, 'r-', linewidth=2, label='True')
    plt.plot(obs_time, obs_p, 'ko', label='Observed')
    plt.legend()
    plt.ylabel('Pressure (MPa)', fontsize=14)
    plt.xlabel('Time (year)', fontsize=14)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    fig.savefig(file_path)


def compute_sd(d, d_obs, Cd):
    sd = 0
    ne = d.shape[1]
    for i in range(ne):
        diff = np.expand_dims(d[:, i] - d_obs[:, 0], axis=1)
        temp = np.matmul(np.matmul(diff.T, np.linalg.inv(Cd)), diff)
        sd = sd + temp[0, 0]

    return sd / ne


def radius_uncertainty_plot(radius_pool, true_radius, filepath, y_down=0, y_up=3000, resolution=100, maplength=35, ndim=30):
    x = np.linspace(0, 360, ndim)
    true_radius = true_radius / resolution * maplength * 2**0.5 * 152.4
    radius_pool = radius_pool / resolution * maplength * 2**0.5 * 152.4
    radius_mean = np.mean(radius_pool, axis=1)
    # radius_99 = np.percentile(radius_pool, 99, axis=1)
    # radius_1 = np.percentile(radius_pool, 1, axis=1)
    radius_99 = np.percentile(radius_pool, 95, axis=1)
    radius_1 = np.percentile(radius_pool, 5, axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[1, 0].plot(x, true_radius[:ndim, ], 'r+', markersize=6)
    ax[1, 0].plot(x, radius_mean[:ndim, ], color='green')
    ax[1, 0].fill_between(x, radius_99[:ndim, ],
                          radius_1[:ndim, ], facecolor='green', alpha=0.25)

    ax[1, 1].plot(x, true_radius[ndim:ndim * 2, ],
                  'r+', markersize=6, label='True')
    ax[1, 1].plot(x, radius_mean[ndim:ndim * 2, ],
                  color='green', label='Pred_mean')
    # ax[1, 1].fill_between(x, radius_99[ndim:ndim * 2, ], radius_1[ndim:ndim * 2, ],
    #                       facecolor='green', alpha=0.25, label='98%_credible_interval')
    ax[1, 1].fill_between(x, radius_99[ndim:ndim * 2, ], radius_1[ndim:ndim * 2, ],
                          facecolor='green', alpha=0.25, label='90%_credible_interval')

    ax[0, 1].plot(x, true_radius[ndim * 2:ndim * 3, ], 'r+', markersize=6)
    ax[0, 1].plot(x, radius_mean[ndim * 2:ndim * 3, ], color='green')
    ax[0, 1].fill_between(x, radius_99[ndim * 2:ndim * 3, ],
                          radius_1[ndim * 2:ndim * 3, ], facecolor='green', alpha=0.25)

    ax[0, 0].plot(x, true_radius[ndim * 3:ndim * 4, ], 'r+', markersize=6)
    ax[0, 0].plot(x, radius_mean[ndim * 3:ndim * 4, ], color='green')
    ax[0, 0].fill_between(x, radius_99[ndim * 3:ndim * 4, ],
                          radius_1[ndim * 3:ndim * 4, ], facecolor='green', alpha=0.25)

    for ax in fig.get_axes():
        ax.set_ylim((y_down, y_up))
        ax.set_xlabel('Theta', fontsize=14)
        ax.set_ylabel('Radius(m)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.label_outer()

    fig.legend(loc='upper center', ncol=3, borderaxespad=0.8, fontsize=14)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
