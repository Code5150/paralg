import numpy as np
import myutils as mu


def ax_expression(arr, x, count, displ, task_cnt=4):
    res = x
    for t in range(task_cnt):
        res *= np.prod(arr[list(range(displ[t], displ[t] + count[t]))].astype(object))
    return res


def check_result_ax_expression(arr, res, x, print_res_value=True):
    check_res = x
    for _i in range(arr.shape[0]):
        check_res *= int(arr[_i])
    if print_res_value:
        print('Result:', res)
        print('Check result:', check_res)
    print('Results equal:', res == check_res)


def check_result_ax_plus_b_expression(arr, res, x, print_res_value=True):
    check_res = (np.linalg.multi_dot(arr) @ x)[0, 0]
    if print_res_value:
        print('Result:', res)
        print('Check result:', check_res)
    print('Results equal:', res == check_res)


def ax_plus_b_expression(arr, x_vec, count, displ, task_cnt=4):
    _result = None
    # print(ind)
    for t in range(task_cnt):
        # print(task_ind)
        taken = np.linalg.multi_dot(arr[list(range(displ[t], displ[t] + count[t]))]) if count[t] > 1 else arr[displ[t]]
        if t == 0:
            _result = taken
        else:
            _result = _result @ taken
    return _result @ x_vec


debug = True
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_count = 4
    n, x0 = mu.input_n_x0()
    a = mu.generate_arr(n)
    count = mu.count_indices(a, task_count)
    displ = np.roll(np.cumsum(count), 1)
    displ[0] = 0
    result = ax_expression(a, x0, count, displ, task_count)
    check_result_ax_expression(a, result, x0, debug)

    i, n, x0 = mu.input_i_n_x0()
    x0_vec = np.array([[x0], [1]])
    a = mu.generate_matrices(i, n)
    count = mu.count_indices(a, task_count)
    displ = np.roll(np.cumsum(count), 1)
    displ[0] = 0
    result = ax_plus_b_expression(a, x0_vec, count, displ, task_count)[0, 0]
    check_result_ax_plus_b_expression(a, result, x0_vec, debug)
