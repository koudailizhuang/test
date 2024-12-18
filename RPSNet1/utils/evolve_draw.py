import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from pathlib import Path
from utils.metrics import fitness

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def plot_evolve(evolve_csv):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    #evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), dpi=500, tight_layout=True)
    # 最前方才有效
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内`
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[-29:]):
        v = x[:, 12 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        #plt.subplots_adjust(0.0,0.0,0.01,0.01,1,1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=12.5)
        plt.title(f'{k} = {mu:.3g}', fontdict={'family': 'Times New Roman', 'size': 13})  # limit to 40 characters
        plt.yticks(fontproperties='Times New Roman', size=12.5)
        plt.xticks(fontproperties='Times New Roman', size=12.5)

        ax = plt.gca();  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(1);  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);  ####设置左边坐标轴的粗细

        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    #f = evolve_csv.with_suffix('.png')  # filename
    f = 'D:/algorithm/yolov5-v7.0-2/runs/train-seg/exp169/evolve.jpg'
    plt.savefig(f, dpi=500)
    plt.close()
    print(f'Saved {f}')

def plot_evolve1(evolve_csv):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    #evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), dpi=500, tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[-29:]):
        v = x[:, 12 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    #f = evolve_csv.with_suffix('.png')  # filename
    f = 'D:/algorithm/yolov5-v7.0-2/runs/train-seg/exp169/evolve.jpg'
    plt.savefig(f, dpi=500)
    plt.close()
    print(f'Saved {f}')

if __name__ == "__main__":
    evolve_csv = 'D:/algorithm/yolov5-v7.0-2/runs/train-seg/exp169/evolve.csv'
    plot_evolve(evolve_csv)
