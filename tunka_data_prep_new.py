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
from typing import List, Tuple, Dict


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
                group_name = file_name.split('.')[0]
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

def vals_to_pic(points: np.ndarray, table_vals: np.ndarray, vals: np.ndarray,
                       grid_size: int, fill_value: float) -> np.ndarray:
    """
    Преобразует значения с каждого детектора в картинку, пригодную для обработки сверточной сетью.

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        vals (np.ndarray) - массив со значениями измеренных величин для каждого детектора.
            -10.0 означает несрабатывание выбранного детектора.
            Размер 19 x 5 (номер детектора, lg(расстояние до оси ШАЛ для наземного детектора),
            lg(ro_el), lg(ro_mu), lg(расстояние до оси ШАЛ для подземного детектора)
        table_vals (np.ndarray) - табличные значения (нужны для восстановления расстояний от детекторов до оси ШАЛ). Размер 19 x 1
        grid_size(int) - размер сетки по осям X и Y
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
    theta = np.deg2rad(table_vals[3]) # восстановленные углы
    phi =  np.deg2rad(table_vals[4])
    last_x = table_vals[10]
    last_y = table_vals[11]

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


def batch_vals_to_pic(points: np.ndarray, grid_size: int, batch_table_vals,
                             batch_vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Берет батч из событий с детекторов n x 19 x 5 (в 5 один индекс соответствует номеру детектора, его убираем)
    Возвращает батч из тензоров  для сверточной нейросети, полученных с помощью griddata из scipy
    Размер n x 3 x 5 x 5, где n - число элементов в батче

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        grid_size(int) - размер сетки по осям X и Y
        batch_table_vals (np.ndarray) - батч из восстановленных значений. Размер n x 19
        batch_vals (np.ndarray) - батч из значений детекторов. Размер n x 19 x 5
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов

    Returns:
        np.ndarray - батч из тензоров размера n x 4 x grid_size x grid_size

    """

    batch_pics = [vals_to_pic(points, table_vals, vals, grid_size, fill_value) for table_vals, vals in zip(batch_table_vals, batch_vals)]
    return np.stack(batch_pics)


# Изначально код писался для работы с большими данными, где k-fold бы не потребовался,
# но для экономии времени и чтобы ничего не поломать в предыдущем коде, я решил написать функцию, которая просто создает
# h5 файл с некоторым выбранным фолдом в качестве тестового
# учесть угол тета в пересчете N_E, N_mu, Ro_200 !!!
def get_train_test_kfold(file_name: str, new_file: str, fold: int, grid_size: int = 10, fill_value: float = -1.0, batch_size: int = 1000) -> None:
    """
    Создает new_file.h5 с train и test датасетами, где тестовым является некоторый выбранный фолд.
    Перед перемешиванием фиксируем random_seed, так что при разных значениях fold трейн и тест не смешиваются.

    Args:
        file_name (str) - имя входного файла
        new_file (str) - имя выходного файла
        grid_size (int) - размер сетки
        fill_value (float) - чем заполнять значения несработавших детекторов
        batch_size (int) - размер батча для считывания из h5-файла

    Returns:

    """
    assert fold <= 10

    np.random.seed(42)

    points = get_points(coords_filename='Koordinaty_oldGeantGrande.txt')

    with h5py.File(file_name, 'r') as hdf5_file, h5py.File(new_file, 'w') as new_hdf5:
        gamma_size = hdf5_file['gamma/headers'].shape[0]
        proton_size = hdf5_file['proton/headers'].shape[0]

        min_size = min(gamma_size, proton_size) # в данных примерно поровну протонных и фотонных событий, так что почти ничего не теряется

        train_group = new_hdf5.create_group('train')
        test_group = new_hdf5.create_group('test')


        train_headers_dset = train_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        train_pics_dset = train_group.create_dataset('pics', shape=(0, 3, 5, 5),
                                                           maxshape=(None, 3, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 3, 5, 5))


        test_headers_dset = test_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        test_pics_dset = test_group.create_dataset('pics', shape=(0, 3, 5, 5),
                                                           maxshape=(None, 3, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 3, 5, 5))

        # создаем массивы из перемешанных индексов для считывания батчей
        gamma_indeces = np.arange(min_size)
        np.random.shuffle(gamma_indeces)

        proton_indeces = np.arange(min_size)
        np.random.shuffle(proton_indeces)

        # для проверки, что random_seed зафиксирован,а след-но трейн и тест не перемешиваются при выбранном фолде fold в качестве тестового
        # print('gamma_indeces:  ',gamma_indeces[1000:1050])
        # print('proton_indeces: ',proton_indeces[1000:1050])


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
            gamma_headers = hdf5_file['gamma/headers'][gamma_indeces_batch]
            gamma_data_blocks = hdf5_file['gamma/data_blocks'][gamma_indeces_batch]
            proton_headers = hdf5_file['proton/headers'][proton_indeces_batch]
            proton_data_blocks = hdf5_file['proton/data_blocks'][proton_indeces_batch]

            # объединяем gamma и proton
            headers = np.vstack([gamma_headers, proton_headers])
            data_blocks = np.vstack([gamma_data_blocks, proton_data_blocks])

            # перемешиваем батч
            indices = np.random.permutation(len(headers))
            headers = headers[indices]
            data_blocks = data_blocks[indices]

            # преобразуем батч из data_blocks в пригодные для сверточной сети тензоры
            pics = batch_vals_to_pic(points, grid_size, headers, data_blocks, fill_value)
       
            if (i+1) != fold:
                # записываем в hdf5 с изменением размера
                train_headers_dset.resize(train_headers_dset.shape[0] + headers.shape[0], axis=0)
                train_headers_dset[-headers.shape[0]:] = headers
    
                train_pics_dset.resize(train_pics_dset.shape[0] + pics.shape[0], axis=0)
                train_pics_dset[-pics.shape[0]:] = pics

            else:
                test_headers_dset.resize(test_headers_dset.shape[0] + headers.shape[0], axis=0)
                test_headers_dset[-headers.shape[0]:] = headers
    
                test_pics_dset.resize(test_pics_dset.shape[0] + pics.shape[0], axis=0)
                test_pics_dset[-pics.shape[0]:] = pics

        print(f"{new_file} was created")




def get_mean_std_tb(hdf5_filename: str, skip_ind: Dict[int, float]) -> List[Tuple[float, float]]:
    """
    Батчами вычисляем среднее и стандартное отклонение для всех табличных признаков без учета пропущенных значений

    Args:
    hdf5_filename - название h5-файла с train и test датасетами
    skip_ind - индексы табличных признаков (из headers), где есть пропущенные значения (номер индекса в headers: значение)

    Return:
    List[Tuple[float, float]] - массив размером (len(features_indeces), 2), (среднее, стандартное отклонение)

    """

    with h5py.File(hdf5_filename, "r") as hdf5_file:
        batch_size = 10000
        train_len = hdf5_file['train']['headers'].shape[0]
        num_feats = hdf5_file['train']['headers'].shape[1]

        mean_std = []
        for feature_index in range(num_feats):
            sum_x = 0.0
            sum_x2 = 0.0
            count = 0
            
            if feature_index in skip_ind.keys():
                for i in range(0, train_len, batch_size):
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
                    batch = hdf5_file['train']['headers'][i:i+batch_size, feature_index]
                    sum_x += np.sum(batch)
                    sum_x2 += np.sum(batch ** 2)
                    count += len(batch)
                
                mean = sum_x / count
                variance = (sum_x2 / count) - mean ** 2

                mean_std.append((mean, np.sqrt(variance)))
                
    return mean_std


def get_mean_std_ch(hdf5_filename: str, skip_ind: Dict[int, float]) -> List[Tuple[float, float]]:
    """
    Батчами вычисляем среднее и стандартное отклонение для всех 2d-признаков без учета пропущенных значений

    Args:
    hdf5_filename - название h5-файла с train и test датасетами
    skip_ind - индексы 2d-признаков (из pics), где есть пропущенные значения (номер канала: значение)

    Return:
    List[Tuple[float, float]] - массив размером (len(features_indeces), 2), (среднее, стандартное отклонение)

    """

    with h5py.File(hdf5_filename, "r") as hdf5_file:
        batch_size = 10000
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

        with h5py.File(self.file_path, 'r') as f:
            self.length = f[self.group_name]['headers'].shape[0]

    def _init_file(self):
        if self._file is None: # если файл не открыт
            self._file = h5py.File(self.file_path, 'r', swmr=True)
            self.headers = self._file[self.group_name]['headers']
            self.pics = self._file[self.group_name]['pics']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_file()

        header = self.headers[idx]
        pic = self.pics[idx]

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

        if not self.expr:
            particle_type = header[5]
            target = get_target(particle_type)
            return pic_tensor, features_tensor, target
            
        else:
            return pic_tensor, features_tensor

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def __getstate__(self):
        # Исключаем открытый файл из сериализации
        state = self.__dict__.copy()
        state['_file'] = None
        state['headers'] = None
        state['pics'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

