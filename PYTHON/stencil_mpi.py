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
#
#
# *******************************************************************
#
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#
# *******************************************************************

import sys
import math
import time
import os

import numpy as np
import mpi4py
from mpi4py import MPI

os.environ['OMP_NUM_THREADS']='1'


def factor(r):
    fac1 = int(math.sqrt(r + 1.0))
    fac2 = 0
    while fac1 > 0:
        if r % fac1 == 0:
            fac2 = int(r / fac1)
            break

        fac1 -= 1

    return fac1, fac2


def compute_star_end(i, n, axis):
    q = n // axis
    r = n % axis

    if i < r:
        start = (q + 1) * i
        end = start + q + 1
    else:
        start = (q + 1) * r + q * (i - r)
        end = start + q

    return start, end


def main():
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    iterations = None
    n = None
    num_procs = comm.Get_size()
    r = 2

    if rank == 0:
        if len(sys.argv) < 2:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: ./stencil <# iterations> <array dimension>")

        iterations = int(sys.argv[1])
        if iterations < 1:
            sys.exit("ERROR: iterations must be >= 1")

        n = int(sys.argv[2])
        if n < 1:
            sys.exit("ERROR: array dimension must be >= 1")

        if len(sys.argv) > 3:
            r = int(sys.argv[3])
            if r < 1:
                sys.exit("ERROR: Stencil radius should be positive")
            if (2 * r + 1) > n:
                sys.exit("ERROR: Stencil radius exceeds grid size")
        else:
            r = 2  # radius=2 is what other impls use right now

    if rank == 0:
        print('Python version     = ', str(sys.version_info.major) + '.' + str(sys.version_info.minor))
        print('mpi4pyt version    = ', mpi4py.__version__)
        print('Numpy version      = ', np.version.version)

        print('Parallel Research Kernels version ')  # , PRKVERSION
        print('Python stencil execution on 2D grid')

        print('Grid size            = ', n)
        print('Radius of stencil    = ', r)
        print('Type of stencil      = ', 'star')
        print('Data type            = double precision')
        print('Compact representation of stencil loop body')
        print('Number of iterations = ', iterations)
        print('Number of processes  = ', num_procs)

    iterations = comm.bcast(iterations, root=0)
    n = comm.bcast(n, root=0)
    r = comm.bcast(r, root=0)

    num_procs_x, num_procs_y = factor(num_procs)

    id_x = rank % num_procs_x
    id_y = rank // num_procs_x

    i_start, i_end = compute_star_end(id_x, n, num_procs_x)

    width = i_end - i_start
    if width == 0:
        print(f"ERROR: rank {rank} has no work to do")
        sys.exit(1)

    j_start, j_end = compute_star_end(id_y, n, num_procs_y)
    height = j_end - j_start
    if height == 0:
        print(f"ERROR: rank {rank} has no work to do")
        sys.exit(1)

    if width < r or height < r:
        print(f"ERROR: rank {rank} has work tile smaller then stencil radius")
        sys.exit(1)

    right_nbr_id = rank + 1
    left_nbr_id = rank - 1
    top_nbr_id = rank + num_procs_x
    bottom_nbr_id = rank - num_procs_x

    weight = np.zeros(((2 * r + 1), (2 * r + 1)))
    for i in range(1, r + 1):
        weight[r, r + i] = +1. / (2 * i * r)
        weight[r + i, r] = +1. / (2 * i * r)
        weight[r, r - i] = -1. / (2 * i * r)
        weight[r - i, r] = -1. / (2 * i * r)

    def fill(i, j):
        return i_start + i - r + j_start + j - r

    ma = np.fromfunction(fill, (width + 2 * r, height + 2 * r), dtype=float)
    ma[0: r, :] = 0
    ma[-r:, :] = 0
    ma[:, 0: r] = 0
    ma[:, -r:] = 0
    mb = np.zeros((width, height))

    top_in = np.zeros((width, r))
    bottom_in = np.zeros((width, r))
    left_in = np.zeros((r, height))
    right_in = np.zeros((r, height))
    for k in range(iterations + 1):
        # start timer after a warmup iteration
        if k == 1:
            comm.barrier()
            t0 = time.time()

        # send
        top_send = None
        bottom_send = None
        left_send = None
        right_send = None
        if id_y < (num_procs_y - 1):
            top_receive = comm.Irecv(top_in, source=top_nbr_id)
            i_s = r
            i_e = i_s + width
            j_s = r + height - r
            j_e = j_s + r
            out = ma[i_s: i_e, j_s: j_e].copy()
            top_send = comm.Isend(out, dest=top_nbr_id)

        if id_y > 0:
            bottom_receive = comm.Irecv(bottom_in, source=bottom_nbr_id)
            i_s = r
            i_e = i_s + width
            j_s = r
            j_e = j_s + r
            out = ma[i_s: i_e, j_s: j_e].copy()
            bottom_send = comm.Isend(out, dest=bottom_nbr_id)

        if id_x < (num_procs_x - 1):
            right_receive = comm.Irecv(right_in, source=right_nbr_id)
            i_s = width + r - r
            i_e = i_s + r
            j_s = r
            j_e = j_s + height
            out = ma[i_s: i_e, j_s: j_e].copy()
            right_send = comm.Isend(out, dest=right_nbr_id)

        if id_x > 0:
            left_receive = comm.Irecv(left_in, source=left_nbr_id)
            i_s = r
            i_e = i_s + r
            j_s = r
            j_e = j_s + height
            out = ma[i_s: i_e, j_s: j_e].copy()
            left_send = comm.Isend(out, dest=left_nbr_id)

        if left_send:
            left_send.wait()
        if right_send:
            right_send.wait()
        if top_send:
            top_send.wait()
        if bottom_send:
            bottom_send.wait()

        # receive
        if id_y < (num_procs_y - 1):
            top_receive.wait()
            i_s = r
            i_e = i_s + width
            j_s = height + r
            j_e = j_s + r
            ma[i_s: i_e, j_s: j_e] = top_in

        if id_y > 0:
            bottom_receive.wait()
            i_s = r
            i_e = i_s + width
            j_s = 0
            j_e = j_s + r
            ma[i_s: i_e, j_s: j_e] = bottom_in

        if id_x > 0:
            left_receive.wait()
            i_s = 0
            i_e = i_s + r
            j_s = r
            j_e = j_s + height
            ma[i_s: i_e, j_s: j_e] = left_in

        if id_x < (num_procs_x - 1):
            right_receive.wait()
            i_s = width + r
            i_e = i_s + r
            j_s = r
            j_e = j_s + height
            ma[i_s: i_e, j_s: j_e] = right_in

        # apply stencil operator

        # for A index
        i_s = max(i_start, r) - i_start + r
        i_e = min(i_end, n - r) - i_start + r
        j_s = max(j_start, r) - j_start + r
        j_e = min(j_end, n - r) - j_start + r

        # for B index
        i_b_s = i_s - r
        i_b_e = i_e - r
        j_b_s = j_s - r
        j_b_e = j_e - r

        W = weight
        mb[i_b_s: i_b_e, j_b_s: j_b_e] += W[r, r] * ma[i_s: i_e, j_s: j_e]
        for s in range(1, r + 1):
            mb[i_b_s: i_b_e, j_b_s: j_b_e] += W[r, r - s] * ma[i_s: i_e, j_s - s: j_e - s] \
                                            + W[r, r + s] * ma[i_s: i_e, j_s + s: j_e + s] \
                                            + W[r - s, r] * ma[i_s - s: i_e - s, j_s: j_e] \
                                            + W[r + s, r] * ma[i_s + s: i_e + s, j_s: j_e]

        ma[r: width + r, r: height + r] += 1

    t1 = time.time()
    local_time = t1 - t0
    stencil_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    # ******************************************************************************
    # * Analyze and output results.
    # ******************************************************************************

    abserr = np.linalg.norm(np.reshape(mb, mb.size), ord=1)
    norm = comm.reduce(abserr, op=MPI.SUM, root=0)

    if rank == 0:
        active_points = (n - 2 * r) ** 2
        norm /= active_points

        epsilon = 1.e-8

        # verify correctness
        reference_norm = 2 * (iterations + 1)
        if abs(norm - reference_norm) < epsilon:
            print('Solution validates')
            flops = (2 * r + 1) * active_points
            avgtime = stencil_time / iterations
            print('Rate (MFlops/s): ', 1.e-6 * flops / avgtime, ' Avg time (s): ', avgtime)
        else:
            print('ERROR: L1 norm = ', norm, ' Reference L1 norm = ', reference_norm)
            sys.exit()


if __name__ == '__main__':
    main()
