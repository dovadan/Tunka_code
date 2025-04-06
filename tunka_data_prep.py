import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def create_particles_h5(file_names: list[str], hdf5_filename: str, warns: bool) -> list[list[int]]:
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

def vals_to_pic_interp(points: np.ndarray, vals: np.ndarray, grid_size: int, fill_value: float) -> np.ndarray:
    """
    Преобразует значения с каждого детектора в картинку, пригодную для обработки сверточной сетью
    с использование кубической интерполяции на сетке с помощью griddata из scipy.

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        vals (np.ndarray) - массив со значениями измеренных величин для каждого детектора.
            -10.0 означает несрабатывание выбранного детектора.
            Размер 19 x 5 (номер детектора, lg(расстояние до оси ШАЛ для наземного детектора),
            lg(ro_el), lg(ro_mu), lg(расстояние до оси ШАЛ для подземного детектора)
        grid_size(int) - размер сетки по осям X и Y
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов


    Returns:
        np.ndarray - тензор размера 4 x grid_size x grid_size

    """

    coords = points[:, 1 : 3]
    grid_layers = []
    for i in range(1, 5):
        vals_layer = vals[:, i]
        for j in range(19):
            if vals_layer[j] == -10.0:
                vals_layer[j] = fill_value


        grid_y, grid_x = np.mgrid[-500:500:complex(grid_size), -500:500:complex(grid_size)]
        grid_values = griddata(coords, vals_layer, (grid_x, grid_y), method = 'cubic', fill_value = fill_value)
        grid_layers.append(grid_values)

    tensor = np.stack(grid_layers, axis=0)
    return tensor


def batch_vals_to_pic_interp(points: np.ndarray, grid_size: int, batch_vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Берет батч из событий с детекторов n x 19 x 5 (в 5 один индекс соответствует номеру детектора, его убираем)
    Возвращает батч из тензоров  для сверточной нейросети, полученных с помощью griddata из scipy
    Размер n x 4 x grid_size x grid_size, где n - число элементов в батче

    Args:
        points (np.ndarray) - координаты детекторов. Размер 19 x 3 (номер детектора, x, y)
        grid_size(int) - размер сетки по осям X и Y
        batch_vals (np.ndarray) - батч из значений детекторов. Размер n x 19 x 5
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов

    Returns:
        np.ndarray - батч из тензоров размера n x 4 x grid_size x grid_size

    """


    batch_pics = [vals_to_pic_interp(points, vals, grid_size, fill_value) for vals in batch_vals]

    return np.stack(batch_pics)

def vals_to_pic(vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Преобразует значения с каждого детектора в картинку, пригодную для обработки сверточной сетью
    с помощью дополнения шестиугольника из детекторов до параллелограмма.

    Args:
        vals (np.ndarray) - массив со значениями измеренных величин для каждого детектора.
            -10.0 означает несрабатывание выбранного детектора.
            Размер 19 x 5 (номер детектора, lg(расстояние до оси ШАЛ для наземного детектора),
            lg(ro_el), lg(ro_mu), lg(расстояние до оси ШАЛ для подземного детектора)
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов


    Returns:
        np.ndarray - тензор размера 4 x 5 x 5

    """
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
    for i in range(1, 5):
        vals_layer = vals[:, i]
        for j in range(19):
            x, y = num_to_ind[j + 1]

            if vals_layer[j] == -10.0:
                tensor[i - 1][y][x] = fill_value
            else:
                tensor[i - 1][y][x] = vals_layer[j]


    return tensor


def batch_vals_to_pic(batch_vals: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Берет батч из событий с детекторов n x 19 x 5 (в 5 один индекс соответствует номеру детектора, его убираем)
    Возвращает батч из тензоров  для сверточной нейросети, полученных с помощью дополнения сетки детекторов до параллелограмма
    Размер n x 4 x 5 x 5, где n - число элементов в батче

    Args:
        batch_vals (np.ndarray) - батч из значений детекторов. Размер n x 19 x 5
        fill_value (float) - значение, которым заполняются пропущенные показания детекторов

    Returns:
        np.ndarray - батч из тензоров размера n x 4 x 5 x 5

    """

    batch_pics = [vals_to_pic(vals, fill_value) for vals in batch_vals]

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


def get_train_test(file_name: str, new_file: str) -> None:
    """
    Создает new_file.h5 с train и test датасетами, состоящими из перемешанных протонных и фотонных событий.
    В каждом датасете соотношения протонных и фотонных событий 1 к 1.
    Считывание осуществляется батчами.

    Args:
        file_name (str) - имя входного файла
        new_file (str) - имя выходного файла

    Returns:

    """
    np.random.seed(42)

    batch_size = 1000

    with h5py.File(file_name, 'r') as hdf5_file, h5py.File(new_file, 'w') as new_hdf5:
        gamma_size = hdf5_file['gamma/headers'].shape[0]
        proton_size = hdf5_file['proton/headers'].shape[0]

        min_size = min(gamma_size, proton_size)

        grid_size = 10
        fill_value = -10.0

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
            pics_interp = batch_vals_to_pic_interp(points, grid_size, data_blocks, fill_value)
            pics = batch_vals_to_pic(data_blocks, fill_value)

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
