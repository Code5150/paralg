import numpy as np


def input_n_x0():
    print('Введите n:')
    num = int(input())
    print('Введите x0:')
    x = int(input())
    return num, x


def generate_arr(num):
    gen = np.random.default_rng()
    arr = gen.normal(2, 1, num).astype(np.int32)
    arr[arr == 0] = 1
    return arr


def count_indices(arr, task_cnt):
    length = arr.shape[0]
    _ind = np.full(task_cnt, length // task_cnt)
    _ind[0:length % task_cnt] += 1
    return _ind


def input_i_n_x0():
    print('Введите i:')
    _i = int(input())
    _n, _x0 = input_n_x0()
    return _i, _n, _x0


def make_matrix(arr):
    if arr.shape[0] == 2:
        arr[1] = [0, 1]
    else:
        arr[1:, :] = 0
        arr[arr.shape[0] - 1, arr.shape[1] - 1] = 1
        np.fill_diagonal(arr[1:arr.shape[0]-1, :arr.shape[1]-2], 1)
    return arr


def generate_matrices(_i, _n):
    result = np.asarray(
        [make_matrix(
            np.concatenate((generate_arr(_n), np.zeros((_n*(_n-1))).astype(np.int32))).reshape((_n, _n))
        ) for _ in range(_i)]
    )
    return result


