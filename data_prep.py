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
from IPython.display import clear_output
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple, Dict, Optional
import os


def create_particles_h5(file_names: List[str], hdf5_filename: str, warns: bool) -> List[List[int]]:
    """
    Переводит батчами данные МК-моделирования из txt в h5df.
    Возвращает номера пропущенных строк в каждом файле, нужно для проверки корректности работы функции.
    Никаких нормировок здесь не производится.

    Args:
        file_names (list[str]) - названия исходных файлов, каждый файл должен соответствовать определенному типу частицы
        hdf5_filename (str) - название выходного файла
        warns (bool) - вывод предупреждений об ошибке в формате данных и незаписи в итоговый в файл

    Returns:
        list[list[int]] - номера пропущенных строк в каждом файле

    """
    mis_lines_all = []
    batch_size = 100
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        for file_name in file_names:
            with open(file_name, 'r') as data_file:
                group_name = os.path.splitext(os.path.basename(file_name))[0]
                group = hdf5_file.create_group(group_name)

                header_dset = group.create_dataset(
                    "headers", shape=(0, 19), maxshape=(None, 19), dtype="float32", chunks=(batch_size, 19)
                )
                data_dset = group.create_dataset(
                    "data_blocks", shape=(0, 19, 5), maxshape=(None, 19, 5), dtype="float32", chunks=(batch_size, 19, 5)
                )

                header_list = []
                data_list = []

                line_num = 0 # на какой строке находимся
                mis_lines = [] # номера строк незаписанных событий для каждого вида частицы
                while True:
                    valid = True

                    line_num += 1
                    header_line = data_file.readline().strip().split()
                    if not header_line: # если файл кончился
                        break

                    if len(header_line) != 19:
                        print(f'Error "len(header_line) != 19" in file {file_name} in line {line_num}: incorrect format')
                        print(header_line)
                        header_line = np.array([-10.0] * 19)
                        valid = False
                        break

                    try:
                        header = np.array(list(map(float, header_line)))
                    except:
                        if warns:
                            print(f'Error "cannot covert to float" in file {file_name} in line {line_num}: incorrect format')
                            print(header_line)
                        valid = False
                        mis_lines.append(line_num)
                        header_line = np.array([-10.0] * 19)


                    data_block = []
                    for _ in range(19):
                        line_num += 1
                        data_line = data_file.readline().strip().split()
                        if len(data_line) != 5:
                            print(f'Error "len(data_line) != 5" in file {file_name} in line {line_num}: incorrect format')
                            print(data_line)
                            data_line = np.array([-10.0] * 5)
                            valid = False
                            break

                        try:
                            data_line = np.array(list(map(float, data_line)))
                        except:
                            if warns:
                                print(f'Error "cannot covert to float" in file {file_name} in line {line_num}: incorrect format')
                                print(data_line)
                            valid = False
                            mis_lines.append(line_num)
                            data_line = np.array([-10.0] * 5)

                        data_block.append(data_line)

                    if valid:
                        header_list.append(header)
                        data_list.append(data_block)


                    if len(header_list) >= batch_size:
                        header_array = np.array(header_list, dtype="float32")
                        data_array = np.array(data_list, dtype="float32")

                        header_dset.resize(header_dset.shape[0] + header_array.shape[0], axis=0)
                        header_dset[-header_array.shape[0]:] = header_array

                        data_dset.resize(data_dset.shape[0] + data_array.shape[0], axis=0)
                        data_dset[-data_array.shape[0]:] = data_array

                        header_list = []
                        data_list = []

                if header_list: # записываем оставшиеся данные
                    header_array = np.array(header_list, dtype="float32")
                    data_array = np.array(data_list, dtype="float32")

                    header_dset.resize(header_dset.shape[0] + header_array.shape[0], axis=0)
                    header_dset[-header_array.shape[0]:] = header_array

                    data_dset.resize(data_dset.shape[0] + data_array.shape[0], axis=0)
                    data_dset[-data_array.shape[0]:] = data_array

                mis_lines_all.append(mis_lines)

    return mis_lines_all


def get_points(coords_filename: str) -> np.ndarray:
    """
    Считывает координаты левой нижней и правой верхней точек прямоугольных детекторов из файла coords_filename.txt
    и переводит их в точки на плоскости XY.
    Расстояние между детекторами много больше их размеров, поэтому в качестве координат детекторов
    приняты геометрические центры пряоугольников

    Args:
        coords_filename (str) - название файла с координатами.

    Returns:
        np.ndarray - координаты геометрических центров прямоугольников на плоскости XY

    """
    with open(coords_filename, 'r') as file_coords:
        lines = file_coords.readlines()
        rectangles = [line.strip().split() for line in lines]

    for i in range(len(rectangles)):
        rectangles[i][0] = int( rectangles[i][0])
        rectangles[i][1: ] = map(float, rectangles[i][1: ])

    points = [ [rectangles[i][0],
    (rectangles[i][1] + rectangles[i][3]) / 2,
    (rectangles[i][2] + rectangles[i][4]) / 2] for i in range(len(rectangles))
    ]

    points = np.array(points)

    return points

def vals_to_pic(points: np.ndarray, table_vals: np.ndarray, vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Преобразует значения с каждого детектора в картинку, пригодную для обработки сверточной сетью.

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        vals (np.ndarray) - массив со значениями измеренных величин для каждого детектора.
            -10.0 означает несрабатывание выбранного детектора.
            Размер 19 x 5 (номер детектора, lg(расстояние до оси ШАЛ для наземного детектора),
            lg(ro_el), lg(ro_mu), lg(расстояние до оси ШАЛ для подземного детектора)
        table_vals (np.ndarray) - табличные значения (нужны для восстановления расстояний от детекторов до оси ШАЛ). Размер 19 x 1
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов


    Returns:
        np.ndarray - тензор размера 3 x 5 x 5
        По каналам: 1 - lg_r от i-го детектора до оси ШАЛ, 2 - lg_ro_200_el, 3 - lg_ro_200_mu

    """
    coords = points[:, 1 : 3]

    # номер детектора : y, x координаты на сетке
    num_to_ind = {
        1 : (2,2),
        2 : (1,1),
        3 : (2,1),
        4 : (3,2),
        5 : (3,3),
        6 : (2,3),
        7 : (1,2),
        8 : (1,0),
        9 : (2,0),
        10 : (3,1),
        11 : (4,2),
        12 : (4,3),
        13 : (4,4),
        14 : (3,4),
        15 : (2,4),
        16 : (1,3),
        17 : (0,2),
        18 : (0,1),
        19 : (0,0),
    }

    tensor = np.full((3, 5, 5), fill_value)

    # найдем расстояние от j-го детектора до оси ШАЛ
    vals_r_layer = np.zeros(19)
    theta = np.deg2rad(table_vals[0]) # восстановленные углы
    phi =  np.deg2rad(table_vals[1])
    last_x = table_vals[3]
    last_y = table_vals[4]

    n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    for j in range(19):
        x_i = coords[j, 1]
        y_i = coords[j, 0]

        a_j = np.array([x_i - last_x, y_i - last_y, 0])
        l_j = np.linalg.norm(a_j - np.dot(a_j, n) * n)
        lg_l_j = np.log10(l_j)

        vals_r_layer[j] = round(lg_l_j, 2)

    for i in [1]:
        vals_layer = vals_r_layer

        for j in range(19):
            x, y = num_to_ind[j + 1]
            tensor[i - 1][y][x] = vals_layer[j]

        # код ниже для отладки
        # for j in range(19):
        #     if vals[j, 1] != -10.0:
        #         print(vals_layer[j], vals[j, 1])

    pics_idxs = [2, 4]
    tens_idxs = [1, 2]
    for p, t in zip(pics_idxs, tens_idxs):
        vals_layer = vals[:, p]
        for j in range(19):
            x, y = num_to_ind[j + 1]

            if vals_layer[j] == -10.0:
                tensor[t][y][x] = fill_value
            else:
                tensor[t][y][x] = vals_layer[j]

    return tensor


def batch_vals_to_pic(points: np.ndarray, batch_table_vals,
                             batch_vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Берет батч из событий с детекторов n x 19 x 5 (в 5 один индекс соответствует номеру детектора, его убираем)
    Возвращает батч из тензоров  для сверточной нейросети, полученных с помощью griddata из scipy
    Размер n x 3 x 5 x 5, где n - число элементов в батче

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        batch_table_vals (np.ndarray) - батч из восстановленных значений. Размер n x 19
        batch_vals (np.ndarray) - батч из значений детекторов. Размер n x 19 x 5
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов

    Returns:
        np.ndarray - батч из тензоров размера n x 4 x 5 x 5

    """

    batch_pics = [vals_to_pic(points, table_vals, vals, fill_value) for table_vals, vals in zip(batch_table_vals, batch_vals)]
    return np.stack(batch_pics)


# Изначально код писался для работы с большими данными, где k-fold бы не потребовался,
# но для экономии времени и чтобы ничего не поломать в предыдущем коде, я решил написать функцию, которая просто создает
# h5 файл с некоторым выбранным фолдом в качестве тестового
# учесть угол тета в пересчете N_E, N_mu, Ro_200 !!!
def get_train_test_kfold(init_file: str, new_file: str, fold: int, fill_value: float = -1.0, 
                         batch_size: int = 1000, 
                         cuts: Optional[Dict[str, float]] = None) -> None:
    """
    Создает new_file.h5 из данных с файла init_file с train и test датасетами, где тестовым является некоторый выбранный фолд.
    В new_file добавляются новые табличные признаки lg_ro_e_sum и lg_ro_mu_sum - суммы по показаниям детекторов соответствующих величин.
    Перед перемешиванием фиксируем random_seed, так что при разных значениях fold трейн и тест не смешиваются.

    Args:
        init_file (str) - имя входного файла
        new_file (str) - имя выходного файла
        fill_value (float) - чем заполнять значения несработавших детекторов
        batch_size (int) - размер батча для считывания из h5-файла
        cuts (dict) - каты для отбора событий

    Returns:

    """
    assert fold <= 10

    np.random.seed(42)

    if cuts is None:
        cuts = {"N_cut": 2, "theta_cut": 35, "dist_cut": 350, "lg_ro200_cut": 0.75}

    points = get_points(coords_filename='data/raw_data/mk/Koordinaty_oldGeantGrande.txt')

    with h5py.File(init_file, 'r') as hdf5_file, h5py.File(new_file, 'w') as new_hdf5:
        table_features = [3, 4, 6, 13, 14, 15, 16, 17, 18]

        gamma_size = hdf5_file['gamma/headers'].shape[0]
        proton_size = hdf5_file['proton/headers'].shape[0]

        min_size = min(gamma_size, proton_size) # в данных примерно поровну протонных и фотонных событий, так что почти ничего не теряется

        train_group = new_hdf5.create_group('train')
        test_group = new_hdf5.create_group('test')

        num_tb_feats = len(table_features) + 2 # 9 исходных + 2 новых признака
        train_headers_dset = train_group.create_dataset('headers', shape=(0, num_tb_feats), maxshape=(None, num_tb_feats), 
                                                        dtype='float32', chunks=(batch_size, num_tb_feats))

        train_pics_dset = train_group.create_dataset('pics', shape=(0, 3, 5, 5),
                                                           maxshape=(None, 3, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 3, 5, 5))
        
        train_labels_dset = train_group.create_dataset('labels', shape=(0, 1), maxshape=(None, 1), 
                                                        dtype='int8', chunks=(batch_size, 1))


        test_headers_dset = test_group.create_dataset('headers', shape=(0, num_tb_feats), maxshape=(None, num_tb_feats), 
                                                      dtype='float32', chunks=(batch_size, num_tb_feats))

        test_pics_dset = test_group.create_dataset('pics', shape=(0, 3, 5, 5),
                                                           maxshape=(None, 3, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 3, 5, 5))
        test_labels_dset = test_group.create_dataset('labels', shape=(0, 1), maxshape=(None, 1), 
                                                        dtype='int8', chunks=(batch_size, 1))

        # создаем массивы из перемешанных индексов для считывания батчей
        gamma_indeces = np.arange(min_size)
        np.random.shuffle(gamma_indeces)

        proton_indeces = np.arange(min_size)
        np.random.shuffle(proton_indeces)

        k = 10
        min_size = min_size//k * k
        fold_size = min_size//k

        for i in range(10):
            start = i * fold_size
            end   = start + fold_size

            # здесь используется sorted, т.к. hdf5_file['gamma/headers'] требует отсортированный массив
            # далее батч будет перемешан
            gamma_indeces_batch = sorted(gamma_indeces[start:end])
            proton_indeces_batch = sorted(proton_indeces[start:end])

            # загружаем батч данных
            gamma_headers = hdf5_file['gamma/headers'][gamma_indeces_batch, :] # (batch_size_gamma, num_table_features)
            gamma_headers = gamma_headers[:, table_features] 
            gamma_data_blocks = hdf5_file['gamma/data_blocks'][gamma_indeces_batch]
            gamma_labels = np.ones((len(gamma_indeces_batch),1))
    
            proton_headers = hdf5_file['proton/headers'][proton_indeces_batch, :]
            proton_headers = proton_headers[:, table_features] 
            proton_data_blocks = hdf5_file['proton/data_blocks'][proton_indeces_batch]
            proton_labels = np.zeros((len(proton_indeces_batch),1))

            # объединяем gamma и proton
            headers = np.vstack([gamma_headers, proton_headers])
            data_blocks = np.vstack([gamma_data_blocks, proton_data_blocks])
            labels = np.vstack([gamma_labels, proton_labels])

            # перемешиваем батч
            indices = np.random.permutation(len(headers))
            headers = headers[indices] # (batch_size, num_table_features=9+2)
            data_blocks = data_blocks[indices] # (batch_size, num_dets=19, num_chanel_features=5)
            labels = labels[indices] # (batch_size, 1)

            # добавляем новые признаки lg_ro_e_sum и lg_ro_mu_sum
            ######
            lg_ro_e_dets = data_blocks[:, :, 2].copy() # (batch_size, num_dets=19)
            lg_ro_mu_dets = data_blocks[:, :, 4].copy() # (batch_size, num_dets=19)

            # электроны
            mask_ro_e = lg_ro_e_dets == -10
            ro_e_dets = np.zeros_like(lg_ro_e_dets)
            ro_e_dets[~mask_ro_e] = np.power(10.0, lg_ro_e_dets[~mask_ro_e])
            ro_e_sum = np.sum(ro_e_dets, axis=1)
            lg_ro_e_sum = np.where(ro_e_sum > 0, np.log10(ro_e_sum), -10.0)

            # мюоны
            mask_ro_mu = lg_ro_mu_dets == -10
            ro_mu_dets = np.zeros_like(lg_ro_mu_dets)
            ro_mu_dets[~mask_ro_mu] = np.power(10.0, lg_ro_mu_dets[~mask_ro_mu])
            ro_mu_sum = np.sum(ro_mu_dets, axis=1)
            lg_ro_mu_sum = np.where(ro_mu_sum > 0, np.log10(ro_mu_sum), -10.0)

            lg_ro_e_sum = lg_ro_e_sum[:, np.newaxis]
            lg_ro_mu_sum = lg_ro_mu_sum[:, np.newaxis]
            headers = np.concatenate([headers, lg_ro_e_sum, lg_ro_mu_sum], axis=1)  # теперь (batch_size, 21)
            ######

            # учитываем каты
            ######
            N_cut = cuts["N_cut"]
            theta_cut = cuts["theta_cut"]
            dist_cut = cuts["dist_cut"]
            lg_ro200_cut = cuts["lg_ro200_cut"]
            
            n_triggered = headers[:, 2] # (batch_size, )
            theta = headers[:, 0] # (batch_size, )
            x = headers[:, 3] # (batch_size, )
            y = headers[:, 4] # (batch_size, )
            lg_ro_200 = headers[:, 7] # (batch_size, )

            mask = (
            (theta < theta_cut) &
            (n_triggered > N_cut) &
            (np.sqrt(x**2 + y**2) < dist_cut) &
            (lg_ro_200 > lg_ro200_cut)
            )

            if not mask.any():
                continue

            headers = headers[mask]
            data_blocks = data_blocks[mask]
            labels = labels[mask]
            ######

            # преобразуем батч из data_blocks в пригодные для сверточной сети тензоры
            pics = batch_vals_to_pic(points, headers, data_blocks, fill_value)
       
            if (i+1) != fold:
                # записываем в hdf5 с изменением размера
                train_headers_dset.resize(train_headers_dset.shape[0] + headers.shape[0], axis=0)
                train_headers_dset[-headers.shape[0]:] = headers
    
                train_pics_dset.resize(train_pics_dset.shape[0] + pics.shape[0], axis=0)
                train_pics_dset[-pics.shape[0]:] = pics

                train_labels_dset.resize(train_labels_dset.shape[0] + labels.shape[0], axis=0)
                train_labels_dset[-labels.shape[0]:] = labels

            else:
                test_headers_dset.resize(test_headers_dset.shape[0] + headers.shape[0], axis=0)
                test_headers_dset[-headers.shape[0]:] = headers
    
                test_pics_dset.resize(test_pics_dset.shape[0] + pics.shape[0], axis=0)
                test_pics_dset[-pics.shape[0]:] = pics

                test_labels_dset.resize(test_labels_dset.shape[0] + labels.shape[0], axis=0)
                test_labels_dset[-labels.shape[0]:] = labels

        print(f"{new_file} was created")


def get_expr(input_path: str, output_path: str, fill_value: float = -1.0, batch_size: int = 10_000,
            cuts: Optional[Dict[str, float]] = None) -> None:
    """
    Создает h5 файл с экспериментальными данными, пригодными для подачи в нейросеть.
    В new_file добавляются новые табличные признаки lg_ro_e_sum и lg_ro_mu_sum - суммы по показаниям детекторов соответствующих величин.

    Args:
        input_path (str) - имя входного файла
        output_path (str) - имя выходного файла
        fill_value (float) - чем заполнять значения несработавших детекторов
        batch_size (int) - размер батча для считывания из h5-файла
        cuts (dict) - каты для отбора событий

    Returns:

    """

    if cuts is None:
        cuts = {"N_cut": 2, "theta_cut": 35, "dist_cut": 350, "lg_ro200_cut": 0.75}
    
    points = get_points('data/raw_data/mk/Koordinaty_oldGeantGrande.txt')
    coords = points[:, 1 : 3]
    num_to_ind = {
        1 : (2,2),
        2 : (1,1),
        3 : (2,1),
        4 : (3,2),
        5 : (3,3),
        6 : (2,3),
        7 : (1,2),
        8 : (1,0),
        9 : (2,0),
        10 : (3,1),
        11 : (4,2),
        12 : (4,3),
        13 : (4,4),
        14 : (3,4),
        15 : (2,4),
        16 : (1,3),
        17 : (0,2),
        18 : (0,1),
        19 : (0,0),
    }
    
    xy_idx = np.array([num_to_ind[i + 1] for i in range(19)]).T   # shape (2,19)
    x_pos, y_pos = xy_idx[0], xy_idx[1]
    
    # те же, что и в МК
    table_features = [
        'tetta',
        'fi',
        'num_dets_triggered',
        'x',
        'y',
        'Ne',
        'Nmu',
        'Ro200',
        'EfromRoLdf'
    ]
    
    pic_features = [
        'Ro_el',
        'Ro_mu'
    ]

    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        n_total = f_in['EHs'].shape[0]
        n_features = len(table_features) + 2 # 9 исходных + 2 новые
        n_pic_features = len(pic_features)  # 2
        n_detectors = 19  # размерность второго измерения у Ro_el и Ro_mu
    
        # создаём две группы train и test
        grp_train = f_out.create_group('train')
        grp_test = f_out.create_group('test')
    
        # в группе train создаём пустые датасеты
        grp_train.create_dataset(
            'headers',
            shape=(0, n_features),
            maxshape=(None, n_features),
            dtype='float32',
            chunks=True
        )
        grp_train.create_dataset(
            'pics',
            shape=(0, 3, 5, 5),
            maxshape=(None, 3, 5, 5),
            dtype='float32',
            chunks=True
        )
    
        # В группе test создаём датасеты для записи
        dset_out = grp_test.create_dataset(
            'headers',
            shape=(0, n_features),
            maxshape=(None, n_features),
            dtype='float32',
            chunks=True
        )
        dset_out_pics = grp_test.create_dataset(
            'pics',
            shape=(0, 3, 5, 5),
            maxshape=(None, 3, 5, 5),
            dtype='float32',
            chunks=True
        )
    
        total_written = 0
    
        # батчами перекидываем данные
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_slice = slice(start, end)
        
            # табличные признаки
            feature_arrays = []
            for key in table_features:
                data = f_in[key][batch_slice].reshape(-1, 1)
                feature_arrays.append(data)
            batch_headers = np.hstack(feature_arrays).astype('float32')
            
            # переводим эксп-е данные к формату, который был при обучении нейронок на МК-данных
            # перевод тета и фи из радиан в градусы (индексы 0 и 1)
            batch_headers[:, 0] = np.degrees(batch_headers[:, 0])  # тета
            batch_headers[:, 1] = np.degrees(batch_headers[:, 1])  # фи
    
            # учитываем зенитный угол для Ne, Nmu, Ro200
            # в эксп-х данных Ne, Nmu, ro200 без логарифма 
            batch_headers[:, 5] = np.exp( 960/260 * (1/np.cos(np.deg2rad(batch_headers[:, 0])) -1) ) * batch_headers[:, 5] 
            batch_headers[:, 6] = np.exp( 960/1050 * (1/np.cos(np.deg2rad(batch_headers[:, 0])) -1) ) * batch_headers[:, 6]
            batch_headers[:, 7] = np.exp(960/260*(1/np.cos(np.deg2rad(batch_headers[:, 0]))-1)) * batch_headers[:, 7]
    
            # логарифмируем Ne, Nmu, Ro200 (индексы 5, 6, 7)
            # пропущенным значениям соответствует 0.0 в исходных данных
            # в логарифме поставим сопоставим ему -10
            for idx in [5, 6, 7]:
                values = batch_headers[:, idx]
                mask = values != 0.0
                lg_values = np.full_like(values, -10.0)
                lg_values[mask] = np.log10(values[mask]) 
                batch_headers[:, idx] = lg_values
    
            # пересчитываем энергию
            alpha, beta = 0.84, 15.99
            batch_headers[:, 8] = alpha * batch_headers[:, 7] + beta

            # добавляем новые признаки lg_ro_e_sum и lg_ro_mu_sum
            ######
            ro_e_dets = f_in['Ro_el'][batch_slice].copy() # (batch_size, num_dets=19)
            ro_mu_dets = f_in['Ro_mu'][batch_slice].copy() # (batch_size, num_dets=19)

            # электроны
            ro_e_sum = np.sum(ro_e_dets, axis=1)
            lg_ro_e_sum = np.where(ro_e_sum > 0, np.log10(ro_e_sum), -10.0)

            # мюоны
            ro_mu_sum = np.sum(ro_mu_dets, axis=1)
            lg_ro_mu_sum = np.where(ro_mu_sum > 0, np.log10(ro_mu_sum), -10.0)

            lg_ro_e_sum = lg_ro_e_sum[:, np.newaxis]
            lg_ro_mu_sum = lg_ro_mu_sum[:, np.newaxis]
            batch_headers = np.concatenate([batch_headers, lg_ro_e_sum, lg_ro_mu_sum], axis=1)  # теперь (batch_size, 21)
            ######
            
            # создаем батч из картинок
            vals_arrays = []
            for key in pic_features:
                data = f_in[key][batch_slice]  # (b_s, 19)
                vals_arrays.append(data)
            batch_vals = np.stack(vals_arrays, axis=-1).astype('float32')  # (b_s, 19, 2)
    
            bs = batch_vals.shape[0]
                
            # найдем расстояние от каждого детектора до оси ШАЛ
            theta = np.deg2rad(batch_headers[:, 0]) # (bs, 1)
            phi   = np.deg2rad(batch_headers[:, 1]) # (bs, 1)
    
            # (bs, 3)
            n_vec = np.stack([np.sin(theta)*np.cos(phi),
                              np.sin(theta)*np.sin(phi),
                              np.cos(theta)], axis=1)
    
            # (bs, 19, 3)
            a = np.stack([
                coords[:, 1][None, :] - batch_headers[:, 3][:, None],
                coords[:, 0][None, :] - batch_headers[:, 4][:, None],
                np.zeros((bs, 19), dtype=np.float32)
            ], axis=-1)
    
            proj = (np.sum(a * n_vec[:, None, :], axis=-1, keepdims=True) * n_vec[:, None, :])
    
            l = np.linalg.norm(a - proj, axis=-1) # (bs,19)
            lg_l = np.round(np.log10(l), 3)
    
            # формируем итоговый 3×5×5
            batch_pics = np.full((bs, 3, 5, 5), fill_value, dtype=np.float32)
            batch_pics[:, 0, y_pos, x_pos] = lg_l
    
            rho_layers = np.where(batch_vals > 0, np.log10(batch_vals), -1)                         # (bs,19,2)
            batch_pics[:, 1, y_pos, x_pos] = rho_layers[:, :, 0]
            batch_pics[:, 2, y_pos, x_pos] = rho_layers[:, :, 1]

            
            # учитываем каты
            ######
            N_cut = cuts["N_cut"]
            theta_cut = cuts["theta_cut"]
            dist_cut = cuts["dist_cut"]
            lg_ro200_cut = cuts["lg_ro200_cut"]
            
            n_triggered = batch_headers[:, 2] # (batch_size, )
            theta = batch_headers[:, 0] # (batch_size, )
            phi = batch_headers[:, 1] # (batch_size, )
            x = batch_headers[:, 3] # (batch_size, )
            y = batch_headers[:, 4] # (batch_size, )
            lg_ro_200 = batch_headers[:, 7] # (batch_size, )

            mask = (
            (theta < theta_cut) &
            (n_triggered > N_cut) &
            (np.sqrt(x**2 + y**2) < dist_cut) &
            (lg_ro_200 > lg_ro200_cut) &
            (x != 0.0) &
            (y != 0.0) &
            (theta != 0.0) &
            (phi != 0.0) 
            )

            if not mask.any():
                continue

            batch_headers = batch_headers[mask]
            batch_pics = batch_pics[mask]
            ######
            
            # добавляем в датасет новые объекты
            selected = mask.sum()
            new_size = total_written + selected
            dset_out.resize((new_size, n_features))
            dset_out[total_written:new_size, :] = batch_headers
    
            dset_out_pics.resize((new_size, 3, 5, 5))
            dset_out_pics[total_written:new_size] = batch_pics
    
            total_written = new_size
    
        print(f"Создан файл {output_path} с headers формой: {dset_out.shape} и pics формой: {dset_out_pics.shape}")




def get_mean_std_tb(hdf5_filename: str, skip_ind: Dict[int, float], test=False) -> List[Tuple[float, float]]:
    """
    Батчами вычисляем среднее и стандартное отклонение на трейне для всех табличных признаков без учета пропущенных значений

    Args:
    hdf5_filename - название h5-файла с train и test датасетами
    skip_ind - индексы табличных признаков (из headers), где есть пропущенные значения (номер индекса в headers: значение)
    test - считать на тестовой выборке (нужно для проверки сдвига распределений трейна и теста)

    Return:
    List[Tuple[float, float]] - массив размером (len(features_indeces), 2), (среднее, стандартное отклонение)

    """

    with h5py.File(hdf5_filename, "r") as hdf5_file:
        batch_size = 10000

        if test:
            train_len = hdf5_file['test']['headers'].shape[0]
            num_feats = hdf5_file['test']['headers'].shape[1]
        else:
            train_len = hdf5_file['train']['headers'].shape[0]
            num_feats = hdf5_file['train']['headers'].shape[1]

        mean_std = []
        for feature_index in range(num_feats):
            sum_x = 0.0
            sum_x2 = 0.0
            count = 0
            
            if feature_index in skip_ind.keys():
                for i in range(0, train_len, batch_size):
                    if test:
                        batch = hdf5_file['test']['headers'][i:i+batch_size, feature_index]
                    else:
                        batch = hdf5_file['train']['headers'][i:i+batch_size, feature_index]
                    mask = batch != skip_ind[feature_index]
                    valid = batch[mask]
    
                    sum_x += np.sum(valid)
                    sum_x2 += np.sum(valid ** 2)
                    count += len(valid)
    
                if count != 0:
                    mean = sum_x / count
                    variance = (sum_x2 / count) - mean ** 2
                else:
                    mean = skip_ind[feature_index]
                    variance = 0
    
                mean_std.append((mean, np.sqrt(variance)))
                
            else:
                for i in range(0, train_len, batch_size):
                    if test:
                        batch = hdf5_file['test']['headers'][i:i+batch_size, feature_index]
                    else:
                        batch = hdf5_file['train']['headers'][i:i+batch_size, feature_index]
                    sum_x += np.sum(batch)
                    sum_x2 += np.sum(batch ** 2)
                    count += len(batch)
                
                mean = sum_x / count
                variance = (sum_x2 / count) - mean ** 2

                mean_std.append((mean, np.sqrt(variance)))
                
    return mean_std


def get_mean_std_ch(hdf5_filename: str, skip_ind: Dict[int, float], test=False) -> List[Tuple[float, float]]:
    """
    Батчами вычисляем среднее и стандартное отклонение на трейне для всех 2d-признаков без учета пропущенных значений

    Args:
    hdf5_filename - название h5-файла с train и test датасетами
    skip_ind - индексы 2d-признаков (из pics), где есть пропущенные значения (номер канала: значение)
    test - считать на тестовой выборке (нужно для проверки сдвига распределений трейна и теста)

    Return:
    List[Tuple[float, float]] - массив размером (len(features_indeces), 2), (среднее, стандартное отклонение)

    """

    with h5py.File(hdf5_filename, "r") as hdf5_file:
        batch_size = 10000

        if test:
            train_len = hdf5_file['test']['pics'].shape[0]
            num_feats = hdf5_file['test']['pics'].shape[1]
            h = hdf5_file['test']['pics'].shape[2]
            w = hdf5_file['test']['pics'].shape[3]
        else:
            train_len = hdf5_file['train']['pics'].shape[0]
            num_feats = hdf5_file['train']['pics'].shape[1]
            h = hdf5_file['train']['pics'].shape[2]
            w = hdf5_file['train']['pics'].shape[3]
            

        mean_std = []
        for feature_index in range(num_feats):
            sum_x = 0.0
            sum_x2 = 0.0
            count = 0
            
            if feature_index in skip_ind.keys():
                for i in range(0, train_len, batch_size):
                    if test:
                        batch = hdf5_file['test']['pics'][i:i+batch_size, feature_index] # (batch_size, 5, 5)
                    else:
                        batch = hdf5_file['train']['pics'][i:i+batch_size, feature_index] # (batch_size, 5, 5)
                    mask = batch != skip_ind[feature_index]
                    valid = batch[mask]
    
                    sum_x += np.sum(valid)
                    sum_x2 += np.sum(valid ** 2)
                    count += mask.sum()
    
                if count != 0:
                    mean = sum_x / count
                    variance = (sum_x2 / count) - mean ** 2
                else:
                    mean = skip_ind[feature_index]
                    variance = 0
    
                mean_std.append((mean, np.sqrt(variance)))
                
            else:
                for i in range(0, train_len, batch_size):
                    if test:
                        batch = hdf5_file['test']['pics'][i:i+batch_size, feature_index] # (batch_size, 5, 5)
                    else:
                        batch = hdf5_file['train']['pics'][i:i+batch_size, feature_index] # (batch_size, 5, 5)
                    sum_x += np.sum(batch)
                    sum_x2 += np.sum(batch ** 2)
                    count += batch.shape[0]*h*w
                
                mean = sum_x / count
                variance = (sum_x2 / count) - mean ** 2

                mean_std.append((mean, np.sqrt(variance)))
                
    return mean_std


def get_target(particle_type):
    """
    Функция для преобразования particle type в метку (0 - gamma, 1 - proton)

    """
    if particle_type == 14.0:
        return 1  # proton
    elif particle_type == 1.0:
        return 0  # gamma
    else:
        return -1

# процедура нормировки следующая:
# если у каких-то объектов может отсутствовать некоторый признак (вместо него подставляется некоторое фикс-е число),
# то для такого признака существующие значения нормируются на среднее 0 и дисперсию 1, затем двигаются вправо
# на фиксированное расстояние. Значения, которые сопоставляются пропущенным двигаются на то же расстояние влево.
# Это нужно поскольку в нейронке в FC-слоях используются симметричные функции активации tanh.
# При таком подходе признаки из той и другой группы будут равнозначными для нейронки.
# Для признаков без отсутствующих значений выполняется обычная нормировка на среднее 0 и дисперсию 1.
# 2d-признаки нормируем так: существующие на 0 и 1, пропущенные влево на фиксированное значение
class CustomDataset(Dataset):
    """
    Датасет, пригодный для подачи в нейросеть

    file_path - путь к h5-файлу с данными
    structure_name - путь к txt файлу со структрурой признаков в h5 файле
    group_name - название группы ('train'/'test')
    features_indeces - индексы признаков, которые берем из файла
    normalize - нормировать или нет
    expr - флаг для эксп-х данных (неразмеченных)

    По отдельным детекторам (2d-признаки):
    mean_ch - среднее по каналам
    std_ch - стандартное отклонение по каналам
    skip_vals_ch - значения, которые приписываются пропущенным признакам по каналам (номер канала : чем заменяется)
    
    Для табличных признаков:
    skip_ind - индексы табличных признаков (из headers), где есть пропущенные значения
    mean_tb, std_tb, skip_vals_tb - аналогично параметрам для каналов
    
    """
    def __init__(self, file_path: str, structure_name: str, group_name: str, features_indeces: List[int],
                       normalize: bool, expr: bool,
                       mean_ch: List[float], std_ch: List[float], skip_vals_ch: Dict[int, float], # игнорируется, если normalize == False

                       mean_tb: List[float] = [], std_tb: List[float] = [], skip_vals_tb: Dict[int, float] = {},
                      ):
        self.file_path = file_path
        self.group_name = group_name
        self.structure_name = structure_name
        self.features_indeces = features_indeces
        self.normalize = normalize
        self.expr = expr
        
        self.mean_ch = np.array(mean_ch)
        self.std_ch = np.array(std_ch)
        self.skip_vals_ch = skip_vals_ch
        self.bias_ch = np.array([5 for _ in range(3)])

        self.mean_tb = np.array(mean_tb)
        self.std_tb = np.array(std_tb)
        self.skip_vals_tb = skip_vals_tb
        self.bias_tb = {ind: 5 for ind in skip_vals_tb.keys()}

        self._file = None
        self.headers = None
        self.pics = None
        self.labels = None

        with h5py.File(self.file_path, 'r') as f:
            self.length = f[self.group_name]['headers'].shape[0]

    def _init_file(self):
        if self._file is None: # если файл не открыт
            self._file = h5py.File(self.file_path, 'r', swmr=True)
            self.headers = self._file[self.group_name]['headers']
            self.pics = self._file[self.group_name]['pics']
            self.labels = self._file[self.group_name]['labels'] if not self.expr else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_file()

        header = self.headers[idx]
        pic = self.pics[idx]
        label = self.labels[idx] if not self.expr else -1

        ### 2d-признаки
        if self.normalize:
            c, h, w = pic.shape

            skip_vals_ch = np.zeros(3)
            for i in range(3):
                skip_vals_ch[i] = self.skip_vals_ch[i]
            
            skip_vals_ch = np.broadcast_to(skip_vals_ch[:, None, None], (c, h, w))
            mean_ch = np.broadcast_to(self.mean_ch[:, None, None], (c, h, w))
            std_ch = np.broadcast_to(self.std_ch[:, None, None], (c, h, w))
            bias_ch = np.broadcast_to(self.bias_ch[:, None, None], (c, h, w))

            # создаём маску, где значения равны пропущенным
            mask = pic == skip_vals_ch

            result = np.empty_like(pic)

            result[mask] = pic[mask] - skip_vals_ch[mask] - bias_ch[mask] # пропущенные влево
            result[~mask] = (pic[~mask] - mean_ch[~mask]) / std_ch[~mask] # существующие нормируем

            pic = result

        pic_tensor = torch.tensor(pic, dtype=torch.float32)

        for key in self.skip_vals_tb.keys():
            assert key in self.features_indeces, f"skip_vals_tb key {key} not in features_indeces"

        # табличные признаки
        if self.normalize:
            features = []
            for ind in self.features_indeces:
                if ind in self.skip_vals_tb.keys():
                    if header[ind] == self.skip_vals_tb[ind]:
                        features.append(header[ind] - self.skip_vals_tb[ind] - self.bias_tb[ind]) # двигаем влево и добавляем в список
                    else:
                        normalized = (header[ind] - self.mean_tb[ind]) / self.std_tb[ind] # нормируем
                        features.append(normalized + self.bias_tb[ind]) # двигаем вправо и добавляем в список
                else:
                    normalized = (header[ind] - self.mean_tb[ind]) / self.std_tb[ind]
                    features.append(normalized)
        else:
            features = [header[ind] for ind in self.features_indeces]
            
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return pic_tensor, features_tensor, label
            

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def __getstate__(self):
        # Исключаем открытый файл из сериализации
        state = self.__dict__.copy()
        state['_file'] = None
        state['headers'] = None
        state['pics'] = None
        state['labels'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

