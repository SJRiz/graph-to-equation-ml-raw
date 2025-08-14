import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import EquationDataset
matplotlib.use('Agg')

def inspect_dataset(dataset, n_samples=1000, x_range=(-5,5), n_plot=24):
    # sample params
    params = []
    reroll_counts = []
    for i in range(min(len(dataset), n_samples)):
        item = dataset[i]
        p = item['parameters']
        params.append(p.cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p))
        # optional: if your dataset can return reroll count, do: reroll_counts.append(item.get('rerolls', 0))
    params = np.stack(params)  # (S, D)
    print("params shape:", params.shape)
    means = params.mean(axis=0)
    stds  = params.std(axis=0)
    mins  = params.min(axis=0)
    maxs  = params.max(axis=0)
    for i,(mn,st,mn_i,mx_i) in enumerate(zip(means,stds,mins,maxs)):
        print(f" coeff a{params.shape[1]-1-i}: mean={mn:.4g}, std={st:.4g}, min={mn_i:.4g}, max={mx_i:.4g}")

    # quick diversity check: number of unique parameter rows
    unique = np.unique(np.round(params, 6), axis=0).shape[0]
    print(f" Unique parameter vectors (rounded 1e-6): {unique}/{params.shape[0]}")

    # sample and plot curves
    N = 400
    xs = np.linspace(x_range[0], x_range[1], N)
    fig, axs = plt.subplots(2, n_plot//2, figsize=(14,6))
    axs = axs.flatten()
    S = min(n_plot, params.shape[0])
    for i in range(S):
        coeffs = params[i]  # assume highest->lowest
        # evaluate poly: coeffs dot [x^deg ... x^0]
        deg = coeffs.size - 1
        xpowers = np.vstack([xs**(deg-j) for j in range(deg+1)])  # (D,N)
        ys = coeffs.dot(xpowers)
        axs[i].plot(xs, ys)
        axs[i].set_xlim(x_range)
        axs[i].set_ylim(-10, 10)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.suptitle("Sample curves (trimmed y-range -10..10)")
    plt.show()

    # quick statistic of max|y|
    maxy = np.max(np.abs(params.dot(np.vstack([xs**(params.shape[1]-1-j) for j in range(params.shape[1])]))), axis=1)
    print("max|y| stats: mean=", maxy.mean(), " median=", np.median(maxy), " 90pct=", np.percentile(maxy,90))

# usage:
train_dataset = EquationDataset(num_samples=2000, image_size=64, split='train')
val_dataset   = EquationDataset(num_samples=500, image_size=64, split='val')
inspect_dataset(train_dataset, n_samples=500, x_range=(-5,5), n_plot=24)
inspect_dataset(val_dataset, n_samples=200, x_range=(-5,5), n_plot=12)
