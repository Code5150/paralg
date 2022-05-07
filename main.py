# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from time import time_ns
# from mpi4py import MPI


# def mpi_helloworld():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     print('Hello from process {} out of {}'.format(rank, size))


def ax_expression(arr, x, task_cnt=4):
    res = x
    for t in range(task_cnt):
        res *= np.prod(arr[list(range(t, arr.shape[0], task_cnt))].astype(object))
    return res


def check_result(arr, res, x, print_res_value=True):
    check_res = x
    for i in range(arr.shape[0]):
        check_res *= int(arr[i])
    if print_res_value:
        print('Result:', res)
        print('Check result:', check_res)
    print('Results equal:', res == check_res)


def input_n_x0():
    print('Введите n:')
    num = int(input())
    print('Введите x0:')
    x = int(input())
    return num, x


def input_i_n_x0():
    print('Введите i:')
    _i = int(input())
    _n, _x0 = input_n_x0()
    return _i, _n, _x0


def generate_arr(num):
    gen = np.random.default_rng()
    arr = gen.normal(4, 2, num).astype(np.int32)
    arr[arr == 0] = 1
    return arr


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


def ax_plus_b_expression(arr, x_vec, task_cnt):
    result = None
    ind = np.full(task_cnt, arr.shape[0] // task_cnt)
    ind[0:i % task_cnt] += 1
    # print(ind)
    for t in range(task_cnt):
        task_ind = 0 if t == 0 else np.sum(ind[0:t])
        # print(task_ind)
        taken = np.linalg.multi_dot(arr[list(range(task_ind, ind[t] + task_ind))]) if ind[t] > 1 else arr[task_ind]
        if t == 0:
            result = taken
        else:
            result = result @ taken
    return result @ x_vec


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_count = 4
    i, n, x0 = input_i_n_x0()
    x0_vec = np.array([[x0], [1]])
    a = generate_matrices(i, n)

    # print(a[[0, 2]])
    # print(np.linalg.multi_dot(a[[0, 2]]))

    # print(a)
    # print(x0_vec)
    print(ax_plus_b_expression(a, x0_vec, task_count))
    print(np.linalg.multi_dot(a) @ x0_vec)

    # ind = np.full(task_count, i // task_count)
    # ind[0:i % task_count] += 1
    # print(ind)
    # print(np.linalg.multi_dot(a[[0,1,2]]))
    # print(np.linalg.multi_dot(a[[3,4]]))
    # print(np.linalg.multi_dot(a[[0,1,2]]) @ np.linalg.multi_dot(a[[3,4]]))

    # n, x0 = input_n_x0()
    # a = make_matrix(np.concatenate((generate_arr(n), np.zeros((n*(n-1))).astype(np.int32))).reshape((n, n)))
    # print(a)
    # print("="*20)
    # print(a[1:a.shape[0]-1, :a.shape[1]-2])
    # print('x = ' + '*'.join(np.char.mod('%d', a)) + '*x0')
    # result = ax_expression(a, x0, task_count)
    # check_result(a, result, x0)
    # ax_plus_b_expression()
