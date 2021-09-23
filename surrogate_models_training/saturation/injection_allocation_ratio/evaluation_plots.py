# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:14:55 2020

@author: thw_1
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.metrics import r2_score

def train_hisotry(history, output_dir):

    fig = plt.figure()
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    fig.savefig('%s/loss_history.png' % (output_dir))
    plt.close()


def ssimplot(ssim, nr, output_dir):
    ind = np.arange(0, nr, 1)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(ind, ssim)
    plt.ylabel('SSIM', fontsize=14)
    plt.xlabel('Sample number', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.close()
    fig.savefig('%s/ssim_plot.png' % (output_dir))


def rmseplot(rmse, nr, output_dir):
    ind = np.arange(0, nr, 1)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(ind, rmse)
    plt.ylabel('RMSE', fontsize=14)
    plt.xlabel('Sample number', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.close()
    fig.savefig('%s/rmse_plot.png' % (output_dir))


def CreatFig(num_case, ImagesPerRow, numRow, filepath):
    images = [Image.open(os.getcwd() + '/figures/' + str(i) + '.png')
              for i in range(num_case)]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_heights = max(heights)
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


def SaveFig_MSE(data, num_case, step, df, vmin=0, vmax=1, cmap='jet'):
    for i in range(num_case):
        export_path = os.getcwd() + '/figures/' + str(i) + '.png'
        rank = i * step
        fig = plt.figure(figsize=(2, 2))
        im = plt.imshow(data[df.index[rank], :, :],
                        cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(str(rank) + 'RMSE_%.2f' % df['MSE'][df.index[rank]], fontsize=8.5, fontweight='bold')
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


def SaveFig_time(true_data, pred_data, time, rank, depth, df, vmin=0, vmax=1, cmap='jet'):

    diff = pred_data - true_data
    # vmin = np.min(diff)
    # vmax = np.max(diff)
    for i in range(depth):
        export_path_1 = os.getcwd() + '/figures/' + str(i) + '.png'
        fig = plt.figure(figsize=(2, 2))
        im = plt.imshow(true_data[df.index[rank], i, :, :],
                        interpolation='bicubic', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(str(int(time[i])) + 'year', fontsize=9, fontweight='bold')
        plt.scatter(71, 71, s=20, facecolors='none', edgecolors='k')
        plt.scatter(71, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 71, s=20, facecolors='none', edgecolors='k')
        plt.axis('off')
        if i == 0:
            fig.colorbar(im, orientation="horizontal",
                         pad=0.03, fraction=0.045)
        fig.savefig(export_path_1, dpi=100, bbox_inches='tight')
        plt.close()

        export_path_2 = os.getcwd() + '/figures/' + str(i + depth) + '.png'
        fig = plt.figure(figsize=(2, 2))
        im = plt.imshow(pred_data[df.index[rank], i, :, :],
                        interpolation='bicubic', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.scatter(71, 71, s=20, facecolors='none', edgecolors='k')
        plt.scatter(71, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 71, s=20, facecolors='none', edgecolors='k')
        plt.axis('off')
        fig.savefig(export_path_2, dpi=100, bbox_inches='tight')
        plt.close()

        export_path_3 = os.getcwd() + '/figures/' + str(i + depth * 2) + '.png'

        fig = plt.figure(figsize=(2, 2))
        im = plt.imshow(diff[df.index[rank], i, :, :], interpolation='bicubic',
                        cmap=cmap, origin='lower', vmin=np.min(diff), vmax=np.max(diff))
        plt.scatter(71, 71, s=20, facecolors='none', edgecolors='k')
        plt.scatter(71, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 141, s=20, facecolors='none', edgecolors='k')
        plt.scatter(141, 71, s=20, facecolors='none', edgecolors='k')
        plt.axis('off')
        if i == 0:
            fig.colorbar(im, orientation="horizontal",
                         pad=0.03, fraction=0.045)
        fig.savefig(export_path_3, dpi=100, bbox_inches='tight')
        plt.close()


def SaveFig_p(t_true, p_true, t_pred, p_pred, num_case, step):
    for i in range(num_case):
        export_path = os.getcwd() + '/figures/'+str(i) + '.png'
        rank = i*step
        fig = plt.figure(figsize=(2, 2))
        plt.plot(t_true, p_true[rank, :], 'ro')
        plt.plot(t_pred, p_pred[rank, :], 'b-', linewidth=4)
        plt.xticks(np.linspace(0, 10, 2))
        plt.ylim([np.min(p_pred), np.max(p_pred)])
        if i == 0:
            plt.xlabel('Time(year)', fontsize=10)
            plt.ylabel('Pressure(MPa)', fontsize=10)
        else:
            plt.tick_params(axis='x', labelbottom=False)
            plt.tick_params(axis='y', labelleft=False)

        plt.tight_layout()
        fig.savefig(export_path, dpi=100)
        plt.close()


def well_crossplot(labels, outputs, savepath):
    fig = plt.figure()
    up = np.max(outputs) + 0.1
    low = np.min(outputs) - 0.1
    plt.plot([low, up], [low, up], 'r')
    plt.scatter(labels, outputs)
    r_square = r2_score(labels, outputs)
    # plt.text(0.7*up, 0.2*up, '$R^2$= %.3f' % r_square, fontsize=14, color='darkred')
    plt.title('$R^2$= %.3f' % r_square, fontsize=14)
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Pred', fontsize=14)
    fig.savefig(savepath)
    plt.close()


def Save_crossplot(true, pred, num_case, step, df, vmin=0, vmax=1):
    for i in range(num_case):
        export_path = os.getcwd() + '/figures/' + str(i) + '.png'
        rank = i * step
        fig = plt.figure(figsize=(2, 2))
        plt.plot([0, 1], [0, 1], 'r')
        plt.plot(true[df.index[rank], :], pred[df.index[rank], :], '*')
        # r_square = r2_score(true[df.index[rank], :], pred[df.index[rank], :])
        plt.title(str(rank) + 'RMSE_%.2f' % df['MSE'][df.index[rank]],
                  fontsize=8.5, fontweight='bold')
        plt.tight_layout()
        fig.savefig(export_path, dpi=100)
        plt.close()

