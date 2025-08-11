import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import bisect
import csv
import os
import random
from tqdm import tqdm
import pandas as pd
from itertools import cycle

def smooth(values, window=10):
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def get_upper_poisson_95(num):
    fc_coeff = [3.09,5.14,6.72,8.25,9.76,11.26,12.75,13.81,15.29,16.77,17.82,19.29,20.34,21.80,22.94,24.31,25.40,26.84,27.84,29.31,30.33]
    if num<21:
        return fc_coeff[num]
    elif num>21:
        return num+2*np.sqrt(num)
    else:
        return 30.855

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = None, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # веса для классов
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size] - индексы классов
        logpt = F.log_softmax(inputs, dim=1) # [batch_size, num_classes]
        pt = torch.exp(logpt)

        # переводим target из формы [batch_size] в форму [batch_size, 1]
        # берем только логарифмы и вероятности нужного класса
        # затем squeeze(1) переводит тензоры из формы [batch_size, 1] в форму [batch_size]
        logpt = logpt.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = pt.gather(1, targets.view(-1, 1)).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)): # если для каждого класса свое alpha
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)[targets]
            else: # если alpha одно для всех
                alpha_t = self.alpha
            logpt = logpt * alpha_t

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def find_n_ksi(probs, labels, fix_threshold=False, threshold=-1.0, graph=False, plt_range=(0,1)):
    """
    Args:
    probs - np.ndarray of predicted prob-s with shape (N, )
    labels - np.ndarray of labels with shape (N, )
    fix_threshold - fix threshold or not
    threshold - fixed threshold
    
    Return:
    ksi_opt =  argmin_{ksi} sigma_95(n(ksi)) / s(ksi)
    s(ksi) = n^{ksi}_{gamma} / n^{0}_{gamma}
    n_ksi = min_{ksi} sigma_95(n(ksi)) / s(ksi)
    
    """
    assert isinstance(probs, np.ndarray), "probs must be np.ndarray"
    assert isinstance(labels, np.ndarray), "labels must be np.ndarray"
    assert len(probs) == len(labels), "probs and labels must be the same size"
    
    preds_class_0 = probs[labels == 0]
    assert len(preds_class_0) != 0 
    
    preds_class_1 = probs[labels == 1]
    assert len(preds_class_1) != 0 

    # сортируем по возрастанию, чтобы можно было применять бин. поиск
    preds_class_gamma = sorted(preds_class_1)
    preds_class_proton = sorted(preds_class_0)

    # если ksi больше всех элементов в preds_class_gamma, то bisect_left выдает len(preds_class_gamma)
    # может получиться так, что найдется протонное событие, которому классификатор даст вероятность принадлежности к фотонам больше,
    # чем для истинных фотонов. Тогда может произойти деление на ноль, и выдаст ошибку
    if fix_threshold == True:
        assert threshold != -1.0, "If fix_threshold==True you must specify the threshold"
        thresholds = [threshold]
    else:
        thresholds = sorted(preds_class_gamma + preds_class_proton)

        max_pred_gamma = preds_class_gamma[-1]
        ind_max_gamma = thresholds.index(max_pred_gamma)
        thresholds = thresholds[0:ind_max_gamma+1]

    # ищем ksi_opt, сложность по времени o(len(test)* log(len(test))), по памяти o(len(test))
    ksi_opt = -1
    s_opt = -1
    n_ksi_min = 10**9

    if graph:
        n_ksi_arr=[]
        thrs=[]
    for ksi in thresholds:
        # ищем бин. поиском индекс элемента, начиная с которого все значения >= ksi
        ind_left_gamma = bisect.bisect_left(preds_class_gamma, ksi)
        n_gamma_0 = len(preds_class_gamma)
        n_gamma_ksi = len(preds_class_gamma) - ind_left_gamma
        s = n_gamma_ksi / n_gamma_0
        
        ind_left_proton = bisect.bisect_left(preds_class_proton, ksi)
        n_gamma_cand_mk = len(preds_class_proton) - ind_left_proton
        
        sigma_95 = get_upper_poisson_95(n_gamma_cand_mk)
        n_ksi = sigma_95 / s

        if graph:
            n_ksi_arr.append(n_ksi)
            thrs.append(ksi)
        
        if n_ksi < n_ksi_min:
            n_ksi_min = n_ksi
            s_opt = s
            ksi_opt = ksi

    if graph:
        print(thrs[n_ksi_arr.index(min(n_ksi_arr))])
        n_ksi_arr = np.array(n_ksi_arr)
        thrs= np.array(thrs)
        plt.plot(thrs, n_ksi_arr)
        plt.xlim(*plt_range)
        plt.show()

    return ksi_opt, s_opt, n_ksi_min


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # при необходимости:
    # torch.use_deterministic_algorithms(True)

def dann_lambda(step, total_steps, lmax=1.0):
    # функция из оригинальной статьи про unsupervised da: 2/(1+exp(-10p)) - 1
    p = step / float(total_steps)
    return lmax * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0).item()

def train_eval(model, train_loader, loss_fn, optimizer, epochs, device, val_loader, graph_every, eval_every, window, save_path,
              fix_threshold=False, threshold=-1.0, scheduler=None, 
              domain_adapt=False, expr_loader=None, loss_fn_dom=None):
    step = 0
    total_steps = len(train_loader) * epochs

    train_cls_losses = [] # по шагам (1 батч - 1 шаг)
    eval_cls_steps = []
    eval_cls_losses = []
    if domain_adapt:
        train_dom_losses = []
        assert expr_loader is not None, "Select expr_loader"
        assert loss_fn_dom is not None, "Select loss_fn_dom"
        expr_iter = cycle(expr_loader)

    assert window % 2 != 0, "Select an odd window size"

    log_path = save_path+"/logs.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "train_loss","test_loss", "ksi_opt", "s_opt", "n_ksi_min"
        ])
        writer.writeheader()
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        for image, features, labels in train_loader:
            model.train()
            step += 1
            optimizer.zero_grad()

            image = image.to(device)
            features = features.to(device)
            labels = labels.to(device).long().squeeze(-1)

            if not domain_adapt:
                logits = model(image, features)
                loss = loss_fn(logits, labels)
                train_cls_losses.append(loss.item())
                
            else:
                image_expr, features_expr, labels_expr = next(expr_iter)
                image_expr = image_expr.to(device)
                features_expr = features_expr.to(device)
    
                # собираем общий батч
                image_all = torch.cat([image, image_expr], dim=0)
                features_all = torch.cat([features, features_expr], dim=0)
        
                # доменные метки: 0=mk, 1=expr
                dom_labels = torch.cat([
                    torch.zeros(image.size(0), dtype=torch.long),
                    torch.ones(image_expr.size(0), dtype=torch.long)
                ], dim=0).to(device)
    
                alpha = dann_lambda(step, total_steps)
    
                cls_logits, dom_logits = model(image_all, features_all, alpha=alpha)
    
                # лосс по меткам только на source части
                loss_cls = loss_fn(cls_logits[:image.size(0)], labels)
                train_cls_losses.append(loss_cls.item())
        
                # лосс по доменам на всем батче
                loss_dom = loss_fn_dom(dom_logits, dom_labels)
                train_dom_losses.append(loss_dom.item())
    
                # общий лосс
                loss = loss_cls + loss_dom
                
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            
            if (step % graph_every) == 0:
                plt.figure(figsize=(10, 4))

                train_steps = np.arange(1,len(train_cls_losses)+1)
                plt.plot(train_steps, train_cls_losses, color='blue', label='train', alpha=0.3)
                
                if domain_adapt:
                    assert len(train_cls_losses) == len(train_dom_losses), "len(train_cls_losses) != len(train_dom_losses)"
                    plt.plot(train_steps, train_dom_losses, color='grey', label='dom', alpha=0.3)

                if len(train_cls_losses) >= window:
                    smoothed_steps = np.arange(window//2+1, len(train_cls_losses)-window//2 + 1)
                    smoothed = smooth(train_cls_losses, window)
                    plt.plot(smoothed_steps, smoothed, color='orange', label='avg_train')

                    if domain_adapt:
                        smoothed_dom = smooth(train_dom_losses, window)
                        plt.plot(smoothed_steps, smoothed_dom, color='green', label='avg_dom')
                
                plt.plot(eval_cls_steps, eval_cls_losses, 'x', color='red', label='valid')
                plt.xlabel("Training steps")
                plt.ylabel("Loss")
                plt.title(f"Loss: step {step}, epoch {epoch+1}")
                plt.legend()
                plt.grid(True)
                plt.savefig(save_path+'/loss.png')
                plt.close()

            if (step % eval_every) == 0:
                model.eval()
                running_loss = 0.0
                preds_all=[]
                labels_all=[]
                with torch.no_grad():
                    for images, features, labels in val_loader:
                        images = images.to(device)
                        features = features.to(device)

                        if not domain_adapt:
                            logits = model(images, features)
                        else:
                            logits = model(images, features)[0] # cls логиты (batch, num_classes)
                        probs = F.softmax(logits, dim=1)[:, 1] # вероятности второго класса
                        preds_all.append(probs.cpu().numpy())
                            
                        labels = labels.to(device).long().squeeze(-1)
                        labels_all.append(labels.cpu().numpy())
                        
                        loss = loss_fn(logits, labels)
            
                        running_loss += loss.item() * labels.size(0)
                        
                preds_all=np.concatenate(preds_all)
                labels_all=np.concatenate(labels_all)

                df_preds_labels = pd.DataFrame({
                    "preds": preds_all,
                    "labels": labels_all
                })
                df_preds_labels.to_csv(os.path.join(save_path, "preds_labels.csv"), index=False)

                if fix_threshold == False:
                    ksi_opt, s_opt, n_ksi_min = find_n_ksi(preds_all, labels_all)
                else:
                    ksi_opt, s_opt, n_ksi_min = find_n_ksi(preds_all, labels_all, fix_threshold, threshold)

                eval_cls_steps.append(step)
                eval_loss = running_loss / len(val_loader.dataset)
                eval_cls_losses.append(eval_loss)

                with open(log_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "train_loss","test_loss", "ksi_opt", "s_opt", "n_ksi_min"
                    ])
                    writer.writerow({
                        "train_loss": f"{train_cls_losses[-1]:.4f}",
                        "test_loss": f"{eval_loss:.4f}",
                        "ksi_opt": f"{ksi_opt:.3f}",
                        "s_opt": f"{s_opt:.3f}",
                        "n_ksi_min": f"{n_ksi_min:.3f}"
                    })

                model.train()

    return model
