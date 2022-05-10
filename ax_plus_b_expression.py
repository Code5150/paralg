import numpy as np
from mpi4py import MPI
import myutils as mu


def check_result_ax_plus_b_expression(arr, res, x, print_res_value=True):
    check_res = (np.linalg.multi_dot(arr) @ x)[0, 0]
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
    i, n, x0 = mu.input_i_n_x0()
    x0_vec = np.array([[x0], [1]])
    sendbuf = mu.generate_matrices(i, n)
    count = mu.count_indices(sendbuf, size) * n * n
    displ = np.roll(np.cumsum(count), 1)
    displ[0] = 0
    if debug:
        print(sendbuf)
        print(count)
        print(displ)
else:
    n = None
    sendbuf = None
    count = np.zeros(size, dtype=np.int32)
    displ = None

n = comm.bcast(n, root=0)
comm.Bcast(count, root=0)
recvbuf = np.zeros(count[rank], dtype=np.int32)

comm.Scatterv([sendbuf, count, displ, MPI.INT], recvbuf, root=0)
recvbuf = recvbuf.reshape((int(count[rank] / n**2), n, n))

if debug:
    print('After Scatterv, process {} has data:'.format(rank))
    print(recvbuf)
    print("Total matrices: ", int(count[rank] / n**2))

task_result = recvbuf[0] if int(count[rank] / n**2) < 2 else np.linalg.multi_dot(recvbuf)
result = np.asarray(comm.gather(task_result, root=0))

if rank == 0:
    result = (np.linalg.multi_dot(result) @ x0_vec)[0, 0]
    check_result_ax_plus_b_expression(sendbuf, result, x0_vec, debug)
