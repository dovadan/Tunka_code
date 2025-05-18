import os
import uuid
from datetime import datetime
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from IPython.display import clear_output
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from multiprocessing import Pool, Process, Manager
from joblib import Parallel, delayed

import tunka_data_prep
import importlib
import tunka_nn

def train_model(model_class,
                optim_class, optim_kwargs,
                crit_class, crit_kwargs,
                sched_class, sched_kwargs,
                epochs, seed, results_path, gpu_id=0, fold=0):

    """
    Реализует пайплайн обучения отдельной модели, нужно для параллельного обучения нескольких моделей
    Сохраняет в текущий каталог график лосс функции на трейне и тесте и значени лосса на трейне и тесте после обучения

    """

    # если не используем k-fold
    if fold == 0:
        file_path = 'train_test.h5'
    else:
        file_path = 'train_test_'+str(fold)+'.h5'

    with open('structure.txt', 'r') as file_struct:
        struct = file_struct.readlines()
    
    columns = struct[0].split(sep = ',')
    columns = [column.strip() for column in columns] # len(columns) = 19

    features_indeces = [3, 4, 6, 13, 14, 15, 16, 17, 18]

    train_dataset_fit = tunka_data_prep.CustomDataset(file_path=file_path, structure_name = 'structure.txt',
                                        group_name='train', features_indeces=features_indeces,
                                        normalize = False,
                                        interp = False,
                                        skip = False,
                                        mean = [], std = [], channel_mins = [],
                                        bias = [],
                                        mean_feat = [], std_feat = [],
                                        bias_feat = [])
    
    train_dataset_fit._init_file()
    
    mean, std, channel_mins = tunka_data_prep.normalize_fit(train_dataset_fit, skip = False, skip_value = -1.0)
    mean_feat, std_feat = tunka_data_prep.normalize_fit_features(dataset=train_dataset_fit, skip = False,
                                                                 features_indeces = features_indeces,
                                                                skip_ind=[16], skip_value=-10.0)
    
    bias = [0.5, 0.5, 0.5, 0.5]
    bias_feat = [9.5]

    train_dataset = tunka_data_prep.CustomDataset(
        file_path=file_path,
        structure_name='structure.txt',
        group_name='train',
        features_indeces=features_indeces,
        normalize=True,
        interp=False,
        skip=True,
        mean=mean,
        std=std,
        channel_mins=channel_mins,
        bias=bias,
        skip_feat=True,
        skip_ind=[16],
        mean_feat=mean_feat,
        std_feat=std_feat,
        bias_feat=bias_feat
    )
    
    test_dataset = tunka_data_prep.CustomDataset(
        file_path=file_path,
        structure_name='structure.txt',
        group_name='test',
        features_indeces=features_indeces,
        normalize=True,
        interp=False,
        skip=True,
        mean=mean,
        std=std,
        channel_mins=channel_mins,
        bias=bias,
        skip_feat=True,
        skip_ind=[16],
        mean_feat=mean_feat,
        std_feat=std_feat,
        bias_feat=bias_feat
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    model = model_class().to(device)

    criterion = crit_class(**crit_kwargs)

    optimizer = optim_class(model.parameters(), **optim_kwargs)

    if sched_class:
        scheduler = sched_class(optimizer, **sched_kwargs)
    else:
        scheduler = None


    train_loss_list, test_loss_list = tunka_nn.train_evaluate_par(
        model, criterion, optimizer, epochs=epochs,
        train_loader=train_loader, test_loader=test_loader,
        scheduler=scheduler
    )

    
    """
    Отрисовка лоссов
    """
    model_name = model_class.__name__

    fig, (ax_plot, ax_info) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 4),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.4},
        constrained_layout=True
    )

    
    ax_plot.plot(train_loss_list, label='train')
    ax_plot.plot(test_loss_list, label='test')
    ax_plot.set_xlabel('Epoch')
    ax_plot.set_ylabel('Loss')
    ax_plot.legend()
    ax_plot.set_title(f'Model {model_name}, seed={seed}')
    
 
    ax_info.axis('off')
    final_train = train_loss_list[-1]
    final_test  = test_loss_list[-1]
    
    optim_name = optim_class.__name__
    crit_name  = crit_class.__name__
    sched_name = sched_class.__name__ if sched_class else "None"
    def fmt(d): return '\n'.join(f'{k}={v}' for k,v in d.items())
    
    text = (
        f'Final train: {final_train:.4f}\n'
        f'Final test:  {final_test:.4f}\n\n'
        f'Opt:   {optim_name}\n{fmt(optim_kwargs)}\n\n'
        f'Crit:  {crit_name}\n{fmt(crit_kwargs)}\n\n'
        f'Sched: {sched_name}\n{fmt(sched_kwargs)}\n\n'
        f'Epochs: {epochs}\n'
        f'Seed:   {seed}'
    )
    ax_info.text(0, 1, text, va='top', ha='left', fontfamily='monospace')

    base_dir = os.path.dirname(results_path)
    pics_dir = os.path.join(base_dir, "pics")
    os.makedirs(pics_dir, exist_ok=True)

    ts_fig = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid_fig = uuid.uuid4().hex[:8]
    filename = f"loss_{model_name}_{seed}_{ts_fig}_{uid_fig}.png"
    filepath = os.path.join(pics_dir, filename)

    plt.savefig(filepath)
    plt.close(fig)


    """
    Отрисовка распределений предсказаний модели
    """
    model_name = model_class.__name__

    fig, (ax_plot, ax_info) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 4),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.4},
        constrained_layout=True
    )


    preds_class_0 = []
    preds_class_1 = []
    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
    
            logits = model(images, features)                      
            probs = F.softmax(logits, dim=1).cpu().numpy()        
    
            labels_np = labels.cpu().numpy()      
    
            preds_class_0.extend(probs[labels_np == 0, 0])
            preds_class_1.extend(probs[labels_np == 1, 0])
    
    
    bins = np.linspace(0, 1, 30)
    
    ax_plot.hist(preds_class_0, bins=bins, alpha=0.6, label='Class 0', density=True, log=True, color='blue', edgecolor='black')
    ax_plot.hist(preds_class_1, bins=bins, alpha=0.6, label='Class 1', density=True, log=True, color='orange', edgecolor='black')
    
    ax_plot.set_xlabel('Predicted probability of class 0 (0 - photon)')
    ax_plot.set_ylabel('Log Count (normalized)')
    ax_plot.legend()
    ax_plot.grid(True, which='both', ls='--', lw=0.5)
    
 
    ax_info.axis('off')
    final_train = train_loss_list[-1]
    final_test  = test_loss_list[-1]
    
    optim_name = optim_class.__name__
    crit_name  = crit_class.__name__
    sched_name = sched_class.__name__ if sched_class else "None"
    def fmt(d): return '\n'.join(f'{k}={v}' for k,v in d.items())
    
    text = (
        f'Final train: {final_train:.4f}\n'
        f'Final test:  {final_test:.4f}\n\n'
        f'Opt:   {optim_name}\n{fmt(optim_kwargs)}\n\n'
        f'Crit:  {crit_name}\n{fmt(crit_kwargs)}\n\n'
        f'Sched: {sched_name}\n{fmt(sched_kwargs)}\n\n'
        f'Epochs: {epochs}\n'
        f'Seed:   {seed}'
    )
    ax_info.text(0, 1, text, va='top', ha='left', fontfamily='monospace')

    base_dir = os.path.dirname(results_path)
    preds_distr = os.path.join(base_dir, "preds_distr")
    os.makedirs(preds_distr, exist_ok=True)

    ts_fig = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid_fig = uuid.uuid4().hex[:8]
    filename = f"preds_distr_{model_name}_{seed}_{ts_fig}_{uid_fig}.png"
    filepath = os.path.join(preds_distr, filename)

    plt.savefig(filepath)
    plt.close(fig)

    """
    Сохранение предсказаний в csv файл (может не подойти для больших данных)
    """
    preds_class_0.sort()
    preds_class_1.sort()
    csv_path = os.path.join(base_dir, "preds.csv")

    row = {
        "Seed": seed,
        "Fold": fold,
        "Class0_Preds": ",".join(f"{x:.6f}" for x in preds_class_0),
        "Class1_Preds": ",".join(f"{x:.6f}" for x in preds_class_1)
    }


    if not os.path.isfile(csv_path):
        df_init = pd.DataFrame([row], columns=["Seed", "Fold", "Class0_Preds", "Class1_Preds"])
        df_init.to_csv(csv_path, index=False)
    else:
        df_new = pd.DataFrame([row], columns=["Seed", "Fold", "Class0_Preds", "Class1_Preds"])
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    

    """
    Подсчет метрик для обученной модели
    """
    metric_loss = tunka_nn.evaluate(model, test_loader, criterion = criterion)
    ksi_opt,  metric_n = tunka_nn.evaluate_n(model, test_loader)

    return model, metric_loss, metric_n, ksi_opt


def train_parallel(
    model_class,
    opt_cfg,
    crit_cfg,
    sched_cfg,
    n_models: int,
    folds: int,
    seeds: list,
    epochs: int
):
    """
    Параллельное обучение по k‑fold. 
    - model_class: класс модели (наследник nn.Module)
    - opt_cfg: tuple (OptimizerClass, optimizer_kwargs)
    - crit_cfg: tuple (CriterionClass, criterion_kwargs)
    - sched_cfg: tuple (SchedulerClass or None, scheduler_kwargs)
    - n_models: число параллельно обучаемых моделей (число сидов) на каждом фолде
    - folds: количество фолдов (целое число)
    - seeds: список из n_models целых значений random seed
    - epochs: число эпох обучения

    Внутри создаётся папка train_logs/<timestamp>/, в неё пишется train_results.txt,
    графики лоссов сохраняются в pics, сохраняется картинка с метриками для каждого разбиения.
    Для каждого разбиения используется n_models сидов.
    
    """

    def train_and_evaluate(seed, results_path, epochs, fold=0):
        model, metric_loss, metric_n, ksi_opt = train_model(
            model_class,
            opt_cfg[0], opt_cfg[1],
            crit_cfg[0], crit_cfg[1],
            sched_cfg[0], sched_cfg[1],
            epochs=epochs,
            seed=seed,
            fold=fold,
            results_path=results_path
        )
        return metric_loss, metric_n, ksi_opt

    ts_main = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("train_logs", ts_main)
    os.makedirs(base_dir, exist_ok=True)

    results_path = os.path.join(base_dir, "train_results.txt")

    mean_loss_folds = []
    std_loss_folds = []

    mean_n_folds = []
    std_n_folds = []

    ksi_mean_folds = []
    ksi_std_folds = []

    with open(results_path, "w") as f:
        for fold_idx in range(1, folds + 1):
            results = Parallel(n_jobs=n_models)(
                delayed(train_and_evaluate)(
                    seed=seeds[i],
                    results_path=results_path,
                    epochs=epochs,
                    fold=fold_idx
                ) for i in range(n_models)
            )

            metrics = np.array(results, dtype=object)  # shape = (n_models, 3)
            losses = metrics[:, 0].astype(float)
            ns = metrics[:, 1].astype(float)
            ksi_opts = metrics[:, 2]

            mean_loss = losses.mean()
            std_loss = losses.std(ddof=1)
            mean_n = ns.mean()
            std_n = ns.std(ddof=1)
            mean_ksi = np.mean(ksi_opts)
            std_ksi = np.std(ksi_opts)

            mean_loss_folds.append(mean_loss)
            std_loss_folds.append(std_loss)

            mean_n_folds.append(mean_n)
            std_n_folds.append(std_n)

            ksi_mean_folds.append(mean_ksi)
            ksi_std_folds.append(std_ksi)

            f.write(f"Fold: {fold_idx}\n")
            f.write(f"Лоссы:              {losses}\n")
            f.write(f"Средний лосс:       {mean_loss:.4f} ± {std_loss:.4f}\n")
            f.write(f"n=sigma_95/s:       {ns}\n")
            f.write(f"Средняя метрика n:  {mean_n:.4f} ± {std_n:.4f}\n")
            f.write(f"ksi_opts:           {ksi_opts}\n\n")


    model_name = model_class.__name__

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[3, 1], wspace=0.4)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[1, 0])
    ax_ksi = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    num_folds = np.arange(1, folds + 1)

    ax_loss.errorbar(
        num_folds,
        mean_loss_folds,
        yerr=std_loss_folds,
        fmt='o',
        capsize=5,
        label='Loss'
    )
    ax_loss.set_ylabel('Средний лосс')
    ax_loss.set_title('Loss по фолдам')
    ax_loss.grid(True)
    ax_loss.legend()

    ax_n.errorbar(
        num_folds,
        mean_n_folds,
        yerr=std_n_folds,
        fmt='o',
        capsize=5,
        color='orange',
        label='n = σ₉₅ / s'
    )
    ax_n.set_ylabel('Среднее n')
    ax_n.set_title('n = σ₉₅ / s по фолдам')
    ax_n.grid(True)
    ax_n.legend()

    ax_ksi.errorbar(
        num_folds,
        ksi_mean_folds,
        yerr=ksi_std_folds,
        fmt='o',
        capsize=5,
        color='green',
        label='ξₒₚₜ'
    )
    ax_ksi.set_xlabel('Фолд')
    ax_ksi.set_ylabel('ξₒₚₜ')
    ax_ksi.set_title('ξₒₚₜ по фолдам')
    ax_ksi.grid(True)
    ax_ksi.legend()

    ax_info.axis('off')

    def fmt(d):
        return "\n".join(f"{k}={v}" for k, v in d.items())

    text = (
        f"Model: {model_name}\n\n"
        f"Opt:   {opt_cfg[0].__name__}\n"
        f"{fmt(opt_cfg[1])}\n\n"
        f"Crit:  {crit_cfg[0].__name__}\n"
        f"{fmt(crit_cfg[1])}\n\n"
        f"Sched: {sched_cfg[0].__name__ if sched_cfg[0] else 'None'}\n"
        f"{fmt(sched_cfg[1]) if sched_cfg[0] else ''}\n\n"
        f"Epochs: {epochs}\n"
        f"n_models: {n_models}"
    )

    ax_info.text(0, 1, text, va="top", ha="left", fontfamily="monospace")

    ts_fig = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid_fig = uuid.uuid4().hex[:8]
    filename_fig = f"fold_metrics_{model_name}_{ts_fig}_{uid_fig}.png"
    filepath_fig = os.path.join(base_dir, filename_fig)

    plt.savefig(filepath_fig)
    plt.close(fig)


    print(f"Done. Все файлы сохранены в папке: {base_dir}")

