import torch.multiprocessing as mp
import h5py
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from IPython.display import clear_output
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple
from datetime import datetime
import uuid
import os

import tunka_data_prep
import importlib
import tunka_nn

importlib.reload(tunka_nn)
importlib.reload(tunka_data_prep)

def train_model(model_class,
                optim_class, optim_kwargs,
                crit_class, crit_kwargs,
                sched_class, sched_kwargs,
                epochs, seed, gpu_id=0, fold=0):

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

    os.makedirs("Pics", exist_ok=True)

    # plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid = uuid.uuid4().hex[:8]
    filename = f"loss_{model_name}_{seed}_{ts}_{uid}.png"
    filepath = os.path.join("Pics", filename)
    plt.savefig(filepath)
    plt.close()

    return model

