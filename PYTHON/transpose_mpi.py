#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# *******************************************************************
#
# NAME:    transpose
#
# PURPOSE: This program measures the time for the transpose of a
#          column-major stored matrix into a row-major stored matrix.
#
# USAGE:   Program input is the matrix order and the number of times to
#          repeat the operation:
#
#          transpose <# iterations> <matrix_size>
#
#          The output consists of diagnostics to make sure the
#          transpose worked and timing statistics.
#
# HISTORY: Written by  Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
# *******************************************************************

import sys
import numpy as np
import mpi4py
from mpi4py import MPI

import time


def main():
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    iterations = None
    order = None
    num_procs = comm.Get_size()

    if rank == 0:
        if len(sys.argv) != 3:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: ./transpose <# iterations> <matrix order>")

        iterations = int(sys.argv[1])
        if iterations < 1:
            sys.exit("ERROR: iterations must be >= 1")

        order = int(sys.argv[2])
        if order < 1:
            sys.exit("ERROR: order must be >= 1")

        if order % num_procs != 0:
            sys.exit("ERROR: order must be multiple times of num_procs")

    if rank == 0:
        print('Python version = ', str(sys.version_info.major) + '.' + str(sys.version_info.minor))
        print('Numpy version  = ', np.version.version)
        print('mpi4py version = ', mpi4py.__version__)

        print('Parallel Research Kernels version ')  # , PRKVERSION
        print('Python Ray Matrix transpose: B = A^T')

        print('Number of iterations = ', iterations)
        print('Matrix order         = ', order)
        print('Number or processes  = ', num_procs)

    iterations = comm.bcast(iterations, root=0)
    order = comm.bcast(order, root=0)

    block_order = order // num_procs
    index_start = rank * block_order
    ma = np.fromfunction(lambda i, j: (index_start + j) * order + i,
                         (order, block_order), dtype=np.float)
    mb = np.zeros((order, block_order), dtype=np.float)

    for k in range(0, iterations + 1):

        if k == 1:
            comm.barrier()
            t0 = time.time()

        start = index_start
        end = start + block_order
        mb[start: end, :] += ma[start: end, :].T
        ma[start: end, :] += 1.0

        work_in = np.zeros((block_order, block_order), dtype=np.float)
        work_out = np.zeros((block_order, block_order), dtype=np.float)
        for phase in range(1, num_procs):
            to_index = (rank + phase) % num_procs
            from_index = (rank - phase + num_procs) % num_procs

            start = to_index * block_order
            end = start + block_order
            work_out = ma[start: end, :]

            send = comm.Isend(work_out, dest=to_index)
            receive = comm.Irecv(work_in, source=from_index)
            receive.wait()
            send.wait()

            ma[start: end, :] += 1.0

            start = from_index * block_order
            end = start + block_order
            mb[start: end, ] += work_in.T

    t1 = time.time()
    local_time = t1 - t0
    trans_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    A = np.fromfunction(lambda i, j: i * order + j + index_start,
                        (order, block_order), dtype=np.float)
    A = A * (iterations + 1.0) + (iterations + 1.0) * iterations / 2.0
    abserr = np.linalg.norm(np.reshape(mb - A, order * block_order), ord=1)

    abserr = comm.reduce(abserr, op=MPI.SUM, root=0)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    if rank == 0:
        epsilon = 1.e-8
        nbytes = 2 * order ** 2 * 8  # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
        if abserr < epsilon:
            print('Solution validates')
            avgtime = trans_time / iterations
            print('Rate (MB/s): ', 1.e-6 * nbytes / avgtime, ' Avg time (s): ', avgtime)
        else:
            print('error ', abserr, ' exceeds threshold ', epsilon)
            sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()
