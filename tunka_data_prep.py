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
from typing import List, Tuple


def create_particles_h5(file_names: List[str], hdf5_filename: str, warns: bool) -> List[List[int]]:
    """
    Переводит батчами данные МК-моделирования из txt в h5df.
    Возвращает номера пропущенных строк в каждом файле, нужно для проверки корректности работы функции

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
                       grid_size: int, fill_value: float, interp: bool = True) -> np.ndarray:
    """
    Преобразует значения с каждого детектора в картинку, пригодную для обработки сверточной сетью
    с использование кубической интерполяции на сетке с помощью griddata из scipy или без неё.

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        vals (np.ndarray) - массив со значениями измеренных величин для каждого детектора.
            -10.0 означает несрабатывание выбранного детектора.
            Размер 19 x 5 (номер детектора, lg(расстояние до оси ШАЛ для наземного детектора),
            lg(ro_el), lg(ro_mu), lg(расстояние до оси ШАЛ для подземного детектора)
        table_vals (np.ndarray) - табличные значения (нужны для восстановления расстояний от детекторов до оси ШАЛ). Размер 19 x 1
        grid_size(int) - размер сетки по осям X и Y
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов
        interp(bool) - с интерполяцией или нет


    Returns:
        np.ndarray - тензор размера 4 x grid_size x grid_size

    """
    if interp:
        coords = points[:, 1 : 3]
        grid_layers = []

        # найдем расстояние от j-го детектора до оси ШАЛ
        vals_r_layer = np.zeros(19)
        theta = np.deg2rad(table_vals[3]) # восстановленные углы
        phi =  np.deg2rad(table_vals[4])
        last_x = table_vals[10]
        last_y = table_vals[11]

        n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        for j in range(19):
            x_i = coords[j, 0]
            y_i = coords[j, 1]

            a_j = np.array([x_i - last_x, y_i - last_y, 0])
            l_j = np.linalg.norm(a_j - np.dot(a_j, n) * n)
            lg_l_j = np.log10(l_j)

            vals_r_layer[j] = round(lg_l_j, 2)


        for i in range(1, 5):
            if i % 2 != 0: # обрабатываем слои с расстояниями r
                vals_layer = vals_r_layer
                # код ниже для отладки
                # for j in range(19):
                #     if vals[j, 1] != -10.0:
                #         print(vals_layer[j], vals[j, 1])

            else: # обрабатываем слои с плотностями rho
                vals_layer = vals[:, i]
                for j in range(19):
                    if vals_layer[j] == -10.0:
                        vals_layer[j] = fill_value


            grid_y, grid_x = np.mgrid[-500:500:complex(grid_size), -500:500:complex(grid_size)]
            grid_values = griddata(coords, vals_layer, (grid_x, grid_y), method = 'cubic', fill_value = fill_value)
            grid_layers.append(grid_values)

        tensor = np.stack(grid_layers, axis=0)
        return tensor

    else:
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

        tensor = np.full((4, 5, 5), fill_value)

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

        for i in range(1, 5):
            if i % 2 != 0: # обрабатываем слои с расстояниями r
                vals_layer = vals_r_layer

                for j in range(19):
                    x, y = num_to_ind[j + 1]
                    tensor[i - 1][y][x] = vals_layer[j]

                # код ниже для отладки
                # for j in range(19):
                #     if vals[j, 1] != -10.0:
                #         print(vals_layer[j], vals[j, 1])

            else: # обрабатываем слои с плотноятми rho
                vals_layer = vals[:, i]
                for j in range(19):
                    x, y = num_to_ind[j + 1]

                    if vals_layer[j] == -10.0:
                        tensor[i - 1][y][x] = fill_value
                    else:
                        tensor[i - 1][y][x] = vals_layer[j]

        return tensor


def batch_vals_to_pic(points: np.ndarray, grid_size: int, batch_table_vals,
                             batch_vals: np.ndarray, fill_value: float, interp: bool = True) -> np.ndarray:
    """
    Берет батч из событий с детекторов n x 19 x 5 (в 5 один индекс соответствует номеру детектора, его убираем)
    Возвращает батч из тензоров  для сверточной нейросети, полученных с помощью griddata из scipy
    Размер n x 4 x grid_size x grid_size, где n - число элементов в батче

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        grid_size(int) - размер сетки по осям X и Y
        batch_table_vals (np.ndarray) - батч из восстановленных значений. Размер n x 19
        batch_vals (np.ndarray) - батч из значений детекторов. Размер n x 19 x 5
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов

    Returns:
        np.ndarray - батч из тензоров размера n x 4 x grid_size x grid_size

    """

    batch_pics = [vals_to_pic(points, table_vals, vals, grid_size, fill_value, interp) for table_vals, vals in zip(batch_table_vals, batch_vals)]
    return np.stack(batch_pics)


def visualise_grid_interp(points: np.ndarray, grid_values: np.ndarray, center = (-1000, -1000)) -> None:
    coords = points[:, 1 : 3]
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_values, origin = 'lower', extent=(-500, 500, -500, 500), cmap='viridis', interpolation='none')
    plt.colorbar(label='Detector Values')
    plt.title('Visualization of Detector Values on Grid with interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    for point in points:
        plt.scatter(point[1], point[2], color = 'black', label = point[0])
        plt.text(point[1] + 10, point[2] + 10, f'{point[0]:.0f}', fontsize=9, color='red', ha='center', va='center')

    if center != (-1000, -1000):
        plt.scatter(center[0], center[1], marker = 'x', color = 'red', s = 150)

    plt.show()

def visualise_grid(grid_values: np.ndarray) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_values, origin = 'lower', extent=(-500, 500, -500, 500), cmap='viridis', interpolation='none')
    plt.colorbar(label='Detector Values')
    plt.title('Visualization of Detector Values on Grid without interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def get_train_test(file_name: str, new_file: str, grid_size: int = 10, fill_value: float = -1.0, batch_size: int = 1000) -> None:
    """
    Создает new_file.h5 с train и test датасетами, состоящими из перемешанных протонных и фотонных событий.
    В каждом датасете соотношения протонных и фотонных событий 1 к 1.
    Считывание осуществляется батчами.

    Args:
        file_name (str) - имя входного файла
        new_file (str) - имя выходного файла
        grid_size (int) - размер сетки
        fill_value (float) - чем заполнять значения несработавших детекторов
        batch_size (int) - размер батча для считывания из h5-файла

    Returns:

    """
    np.random.seed(42)

    with h5py.File(file_name, 'r') as hdf5_file, h5py.File(new_file, 'w') as new_hdf5:
        gamma_size = hdf5_file['gamma/headers'].shape[0]
        proton_size = hdf5_file['proton/headers'].shape[0]

        min_size = min(gamma_size, proton_size)

        train_group = new_hdf5.create_group('train')
        test_group = new_hdf5.create_group('test')


        train_headers_dset = train_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        train_pics_interp_dset = train_group.create_dataset('pics_interp', shape=(0, 4, grid_size, grid_size),
                                                           maxshape=(None, 4, grid_size, grid_size), dtype='float32',
                                                           chunks=(batch_size, 4, grid_size, grid_size))

        train_pics_dset = train_group.create_dataset('pics', shape=(0, 4, 5, 5),
                                                           maxshape=(None, 4, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 4, 5, 5))


        test_headers_dset = test_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        test_pics_interp_dset = test_group.create_dataset('pics_interp', shape=(0, 4, grid_size, grid_size),
                                                           maxshape=(None, 4, grid_size, grid_size), dtype='float32',
                                                           chunks=(batch_size, 4, grid_size, grid_size))

        test_pics_dset = test_group.create_dataset('pics', shape=(0, 4, 5, 5),
                                                           maxshape=(None, 4, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 4, 5, 5))

        # создаем массивы из перемешанных индексов для считывания батчей
        gamma_indeces = np.arange(min_size)
        np.random.shuffle(gamma_indeces)

        proton_indeces = np.arange(min_size)
        np.random.shuffle(proton_indeces)


        # заполняем новый файл по батчам, которые сформированы случайным образом
        for start in range(0, min_size, batch_size):
            end = min(start + batch_size, min_size)

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

            # преобразуем батч из data_blocks в пригодные для сверточной сети тензоры,
            # полученные двумя разными способами: с интерполяцией и без
            pics_interp = batch_vals_to_pic(points, grid_size, headers,
                             data_blocks, fill_value, interp = True)
            pics = batch_vals_to_pic(points, grid_size, headers, data_blocks, fill_value, interp = False)

            print(pics_interp.shape)
            print(pics.shape)

            # разделяем на train и test
            split = int(len(headers) * 0.9)
            train_headers, test_headers = headers[:split], headers[split:]
            # train_data, test_data = data_blocks[:split], data_blocks[split:]
            train_pics_interp, test_pics_interp = pics_interp[:split], pics_interp[split:]
            train_pics, test_pics = pics[:split], pics[split:]

            # записываем в hdf5 с изменением размера
            train_headers_dset.resize(train_headers_dset.shape[0] + train_headers.shape[0], axis=0)
            train_headers_dset[-train_headers.shape[0]:] = train_headers

            train_pics_interp_dset.resize(train_pics_interp_dset.shape[0] + train_pics_interp.shape[0], axis=0)
            train_pics_interp_dset[-train_pics_interp.shape[0]:] = train_pics_interp

            train_pics_dset.resize(train_pics_dset.shape[0] + train_pics.shape[0], axis=0)
            train_pics_dset[-train_pics.shape[0]:] = train_pics



            test_headers_dset.resize(test_headers_dset.shape[0] + test_headers.shape[0], axis=0)
            test_headers_dset[-test_headers.shape[0]:] = test_headers

            test_pics_interp_dset.resize(test_pics_interp_dset.shape[0] + test_pics_interp.shape[0], axis=0)
            test_pics_interp_dset[-test_pics_interp.shape[0]:] = test_pics_interp

            test_pics_dset.resize(test_pics_dset.shape[0] + test_pics.shape[0], axis=0)
            test_pics_dset[-test_pics.shape[0]:] = test_pics


# Изначально код писался для работы с большими данными, где k-fold бы не потребовался,
# но для экономии времени и чтобы ничего не поломать в предыдущем коде, я решил написать функцию, которая просто создает
# h5 файл с некоторым выбранным фолдом в качестве тестового
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

        min_size = min(gamma_size, proton_size)

        train_group = new_hdf5.create_group('train')
        test_group = new_hdf5.create_group('test')


        train_headers_dset = train_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        train_pics_interp_dset = train_group.create_dataset('pics_interp', shape=(0, 4, grid_size, grid_size),
                                                           maxshape=(None, 4, grid_size, grid_size), dtype='float32',
                                                           chunks=(batch_size, 4, grid_size, grid_size))

        train_pics_dset = train_group.create_dataset('pics', shape=(0, 4, 5, 5),
                                                           maxshape=(None, 4, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 4, 5, 5))


        test_headers_dset = test_group.create_dataset('headers', shape=(0, 19), maxshape=(None, 19), dtype='float32', chunks=(batch_size, 19))

        test_pics_interp_dset = test_group.create_dataset('pics_interp', shape=(0, 4, grid_size, grid_size),
                                                           maxshape=(None, 4, grid_size, grid_size), dtype='float32',
                                                           chunks=(batch_size, 4, grid_size, grid_size))

        test_pics_dset = test_group.create_dataset('pics', shape=(0, 4, 5, 5),
                                                           maxshape=(None, 4, 5, 5), dtype='float32',
                                                           chunks=(batch_size, 4, 5, 5))

        # создаем массивы из перемешанных индексов для считывания батчей
        gamma_indeces = np.arange(min_size)
        np.random.shuffle(gamma_indeces)

        proton_indeces = np.arange(min_size)
        np.random.shuffle(proton_indeces)

        # для проверки, что трейн и тест не перемешиваются при выбранном фолде fold в качестве тестового
        print('gamma_indeces:  ',gamma_indeces[1000:1050])
        print('proton_indeces: ',proton_indeces[1000:1050])


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

            # преобразуем батч из data_blocks в пригодные для сверточной сети тензоры,
            # полученные двумя разными способами: с интерполяцией и без
            pics_interp = batch_vals_to_pic(points, grid_size, headers,
                             data_blocks, fill_value, interp = True)
            pics = batch_vals_to_pic(points, grid_size, headers, data_blocks, fill_value, interp = False)

            # print(pics_interp.shape)
            # print(pics.shape)

            if (i+1) != fold:
                # записываем в hdf5 с изменением размера
                train_headers_dset.resize(train_headers_dset.shape[0] + headers.shape[0], axis=0)
                train_headers_dset[-headers.shape[0]:] = headers
    
                train_pics_interp_dset.resize(train_pics_interp_dset.shape[0] + pics_interp.shape[0], axis=0)
                train_pics_interp_dset[-pics_interp.shape[0]:] = pics_interp
    
                train_pics_dset.resize(train_pics_dset.shape[0] + pics.shape[0], axis=0)
                train_pics_dset[-pics.shape[0]:] = pics

            else:
                test_headers_dset.resize(test_headers_dset.shape[0] + headers.shape[0], axis=0)
                test_headers_dset[-headers.shape[0]:] = headers
    
                test_pics_interp_dset.resize(test_pics_interp_dset.shape[0] + pics_interp.shape[0], axis=0)
                test_pics_interp_dset[-pics_interp.shape[0]:] = pics_interp
    
                test_pics_dset.resize(test_pics_dset.shape[0] + pics.shape[0], axis=0)
                test_pics_dset[-pics.shape[0]:] = pics

            i += 1



def get_mean_variance(hdf5_filename: str, structure_name: str, features_indeces: List[int]) -> List[Tuple[int]]:
    """
    Батчами вычисляем среднее и дисперсию для выбранных признаков

    Args:
    hdf5_filename (str) - название h5-файла с train и test датасетами
    features_indeces (list[int])  - индексы выбранных признаков
    structure_name (str) - файл со структурой признаков

    Return:

    list[tuple[int]] - массив размером len(features_indeces) x 2, (среднее, дисперсия)


    """

    with h5py.File(hdf5_filename, "r") as hdf5_file, open(structure_name, "r") as file_struct:
        struct = file_struct.readlines()
        columns = struct[0].split(sep = ',')
        columns = [column.strip() for column in columns] # len(columns) = 19


        batch_size = 10000
        train_len = hdf5_file['train']['headers'].shape[0]

        mean_variance = []
        for feature_index in features_indeces:
            sum_x = 0
            sum_x2 = 0
            count = 0

            for i in range(0, train_len, batch_size):
                batch = hdf5_file['train']['headers'][i:i+batch_size, feature_index]

                sum_x += np.sum(batch)
                sum_x2 += np.sum(batch ** 2)
                count += len(batch)

            mean = sum_x / count
            variance = (sum_x2 / count) - mean ** 2

            mean_variance.append((mean, np.sqrt(variance)))

    return mean_variance


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

       
class CustomDataset(Dataset):
    """
    Датасет, пригодный для подачи в нейросеть
    
    normalize - нормировать или нет
    skip - пропускаем или нет отсутствующие значения при нормировке. Если да, то в i-м канале они сдвинутся на bias[i]
    interp - картинки с интерполяцией или нет
    bias - сдвиги для несработавших детекторов (нужны, только если skip == True)
    
    Для табличных признаков аналогично
    skip_ind - индексы признаков с пропущенными  значениями из features_indeces(в данном случае такой индекс только один)
    
    """
    def __init__(self, file_path: str, structure_name: str, group_name: str, features_indeces: List[int],
                       normalize: bool,
                       interp: bool,
                       skip: bool,
                       mean: List[float], std: List[float], channel_mins: List[float], # игнорируется, если normalize == False
                       bias: List[float], # игнорируется, если skip == False

                       skip_ind: List[int] = [],
                       skip_feat: bool = False,
                       mean_feat: List[float] = [], std_feat: List[float] = [],
                       bias_feat: List[float] = []
                      ):
        self.file_path = file_path
        self.group_name = group_name
        self.features_indeces = features_indeces

        self.normalize = normalize
        self.skip = skip
        self.interp = interp
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.channel_mins = np.array(channel_mins)
        self.bias = np.array(bias)

        self.skip_ind = skip_ind
        self.mean_feat = np.array(mean_feat)
        self.std_feat = np.array(std_feat)
        self.bias_feat = np.array(bias_feat)

        self._file = None
        self.headers = None
        self.pics_interp = None

        with h5py.File(self.file_path, 'r') as f:
            self.length = f[self.group_name]['headers'].shape[0]

    def _init_file(self):
        if self._file is None: # если файл не открыт
            self._file = h5py.File(self.file_path, 'r', swmr=True)
            self.headers = self._file[self.group_name]['headers']
            if self.interp:
                self.pics_interp = self._file[self.group_name]['pics_interp']
            else:
                self.pics_interp = self._file[self.group_name]['pics']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_file()

        header = self.headers[idx]

        pic = self.pics_interp[idx]
        if self.normalize:
            if self.skip:
                c, h, w = pic.shape

                channel_mins = np.broadcast_to(self.channel_mins[:, None, None], (c, h, w))
                bias = np.broadcast_to(self.bias[:, None, None], (c, h, w))
                mean = np.broadcast_to(self.mean[:, None, None], (c, h, w))
                std = np.broadcast_to(self.std[:, None, None], (c, h, w))

                # создаём маску, где значения меньше порога
                mask = pic < channel_mins

                result = np.empty_like(pic)

                result[mask] = pic[mask] + bias[mask]
                result[~mask] = (pic[~mask] - mean[~mask]) / std[~mask]

                pic = result

            else:
                for c in range(pic.shape[0]):
                    pic[c] = (pic[c] - self.mean[c]) / self.std[c]

        pic_tensor = torch.tensor(pic, dtype=torch.float32)


        if self.normalize:
            features = []
            for i, ind in enumerate(self.features_indeces):
                if self.skip_ind and (ind in self.skip_ind) and (header[ind] == -10): # поправить !!!
                    features.append(header[ind] + self.bias_feat[self.skip_ind.index(ind)])
                else:
                    features.append( (header[ind] - self.mean_feat[i]) / self.std_feat[i] )
                    
        else:
            features = [header[ind] for ind in self.features_indeces]
            
        features_tensor = torch.tensor(features, dtype=torch.float32)

        particle_type = header[5]
        target = get_target(particle_type)

        return pic_tensor, features_tensor, target

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def __getstate__(self):
        # Исключаем открытый файл из сериализации
        state = self.__dict__.copy()
        state['_file'] = None
        state['headers'] = None
        state['pics_interp'] = None
        state['pics'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)



def normalize_fit(dataset: Dataset, skip: bool = False, skip_value: float = -1.0) -> Tuple[np.ndarray]:
    """
    Считаем среднее и дисперсию картинок по выборке dataset для каждого канала.
    Усреднение ведется по размеру батча, ширине и высоте картинки.

    Возвращает среднее, стандартное отклонение и минимальное значение величины в каждом канале,
    при этом при рассчете минимальных значений объекты со значением величины, равной skip_value, не учитываются

    skip - пропускаем или нет пропущенные значения при подсчете среднего и дисперсии (но не минимумов)

    """
    dataset_len = dataset.pics_interp.shape[0]
    c = dataset.pics_interp.shape[1]
    h = dataset.pics_interp.shape[2]
    w = dataset.pics_interp.shape[3]

    channel_sum = np.zeros(c, dtype = np.float64) # для подсчета среднего и дисперсии
    channel_squared_sum = np.zeros(c, dtype = np.float64)

    channel_mins = np.full(c, 10**9, dtype = np.float64)  # массив из минимумов

    valid_counts = np.zeros(c, dtype=np.int64) # для подсчета не маскированных значений в каждом канале

    batch_size = 1000

    # предполагается, что пропущенные значения во всех каналах заполнены одним и тем же значением skip_value
    for i in range(0, dataset_len, batch_size):
        batch = dataset.pics_interp[i:i+batch_size] # [1000, 4, 10, 10]

        # для подсчета минимумов
        masked = np.where(batch == skip_value, np.inf, batch)
        min_vals = masked.min(axis = (0, 2, 3))

        for j in range(c): # обновляем минимумы в каждом канале
            if min_vals[j] < channel_mins[j]:
                channel_mins[j] = min_vals[j]


        # для подсчета среднего и дисперсии
        if skip:
            mask = batch != skip_value
            # mask = np.abs(batch - skip_value) > 0.3

            channel_sum += (batch * mask).sum(axis=(0, 2, 3))
            channel_squared_sum += ((batch ** 2) * mask).sum(axis=(0, 2, 3))
            valid_counts += mask.sum(axis=(0, 2, 3))

        else:
            channel_sum += batch.sum(axis=(0, 2, 3))
            channel_squared_sum += (batch ** 2).sum(axis=(0, 2, 3))
            valid_counts += batch.shape[0] * h * w  # весь батч


    mean = channel_sum / valid_counts
    std = np.sqrt(channel_squared_sum / valid_counts - mean ** 2)

    return mean, std, channel_mins


def normalize_fit_features(dataset: Dataset, skip_ind: List[int], features_indeces: List[str], skip: bool = False, skip_value: float = -10.0) -> Tuple[np.ndarray]:
    """
    Считаем среднее и дисперсию табличный признаков по выборке dataset.
    Усреднение ведется по размеру батча.

    Для табличных признаков пропущенные значения есть только для last_N_mu

    Возвращает среднее, стандартное отклонение для каждого табличного признака.
    Если skip == True, то в last_N_mu пропускаем значения со skip_value
    skip_ind - индексы признаков с пропущенными значениями из features_indeces(в данном случае такой индекс только один)

    """
    skip_ind = [features_indeces.index(feat_idx) for feat_idx in skip_ind]
    
    dataset_len = dataset.headers.shape[0]
    c = len(features_indeces)

    feature_sum = np.zeros(c, dtype = np.float64) # для подсчета среднего и дисперсии
    feature_squared_sum = np.zeros(c, dtype = np.float64)

    valid_counts = np.zeros(c, dtype=np.int64) # для подсчета не маскированных значений в каждом канале

    batch_size = 1000

    for i in range(0, dataset_len, batch_size):
        batch = dataset.headers[i:i + batch_size][:, features_indeces]

        for j in range(c):
            column = batch[:, j]
            
            if skip and j in skip_ind:
                valid_mask = column != skip_value
                feature_sum[j] += column[valid_mask].sum()
                feature_squared_sum[j] += (column[valid_mask] ** 2).sum()
                valid_counts[j] += valid_mask.sum()
                
            else:
                feature_sum[j] += column.sum()
                feature_squared_sum[j] += (column ** 2).sum()
                valid_counts[j] += column.shape[0]


    mean = feature_sum / valid_counts
    std = np.sqrt(feature_squared_sum / valid_counts - mean ** 2)

    return mean, std