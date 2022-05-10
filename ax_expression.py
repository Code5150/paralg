import numpy as np
from mpi4py import MPI
import myutils as mu


def check_result_ax_expression(arr, res, x, print_res_value=True):
    check_res = x
    for _i in range(arr.shape[0]):
        check_res *= int(arr[_i])
    if print_res_value:
        print('Result:', res)
        print('Check result:', check_res)
    print('Results equal:', res == check_res)


debug = True

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
    n, x0 = mu.input_n_x0()
    sendbuf = mu.generate_arr(n)
    count = mu.count_indices(sendbuf, size)
    displ = np.roll(np.cumsum(count), 1)
    displ[0] = 0
    if debug:
        print(sendbuf)
        print(count)
        print(displ)
else:
    sendbuf = None
    count = np.zeros(size, dtype=np.int32)
    displ = None

comm.Bcast(count, root=0)
recvbuf = np.zeros(count[rank], dtype=np.int32)

comm.Scatterv([sendbuf, count, displ, MPI.INT], recvbuf, root=0)

if debug:
    print('After Scatterv, process {} has data:'.format(rank), recvbuf)

task_result = np.prod(recvbuf.astype(object))
results_arr = comm.gather(task_result, root=0)

if rank == 0:
    result = np.prod(results_arr)*x0
    check_result_ax_expression(sendbuf, result, x0, debug)

