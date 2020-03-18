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

import numpy as np
import ray

if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer


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
        end = start + q
    else:
        start = (q + 1) * r + q * (i - r)
        end = start + q - 1

    return start, end


@ray.remote
class Executor:
    def __init__(self, order, r, index, num_procs,
                 num_procs_x, num_procs_y,
                 i_start, i_end,
                 j_start, j_end):

        self._order = order
        self._r = r
        self._index = index
        self._num_procs = num_procs
        self._num_procs_x = num_procs_x
        self._num_procs_y = num_procs_y
        self._i_start = i_start
        self._i_end = i_end + 1
        self._j_start = j_start
        self._j_end = j_end + 1

        self._width = self._i_end - self._i_start
        self._height = self._j_end - self._j_start

        self._id_x = index % self._num_procs_x
        self._id_y = index // self._num_procs_x

        self._right_nbr_id = index + 1
        self._left_nbr_id = index - 1
        self._top_nbr_id = index + num_procs_x
        self._bottom_nbr_id = index - num_procs_x

        self._weight = np.zeros(((2 * r + 1), (2 * r + 1)))
        for i in range(1, r + 1):
            self._weight[r, r + i] = +1. / (2 * i * r)
            self._weight[r + i, r] = +1. / (2 * i * r)
            self._weight[r, r - i] = -1. / (2 * i * r)
            self._weight[r - i, r] = -1. / (2 * i * r)

        outer = self
        def fill(i, j):
            return outer._i_start + i - outer._r + outer._j_start + j - outer._r

        self._ma = np.fromfunction(fill,
                                   (self._width + 2 * self._r, self._height + 2 * self._r),
                                   dtype=float)
        self._ma[0: r, :] = 0
        self._ma[-r: , :] = 0
        self._ma[:, 0: r] = 0
        self._ma[:, -r: ] = 0
        self._mb = np.zeros((self._width, self._height))

    def register(self, left, right, top, bottom):
        self._right_nbr = right
        self._left_nbr = left
        self._top_nbr = top
        self._bottom_nbr = bottom

    def execute(self):
        ids = []
        if self._id_y < (self._num_procs_y - 1):
            i_start = self._r
            i_end = i_start + self._width
            j_start = self._r + self._height - self._r
            j_end = j_start + self._r
            out = self._ma[i_start: i_end, j_start: j_end].copy()
            ids.append(self._top_nbr.receive.remote(self._index, out))

        if self._id_y > 0:
            i_start = self._r
            i_end = i_start + self._width
            j_start = self._r
            j_end = j_start + self._r
            out = self._ma[i_start: i_end, j_start: j_end].copy()
            ids.append(self._bottom_nbr.receive.remote(self._index,out))

        if self._id_x < (self._num_procs_x - 1):
            i_start = self._width + self._r - self._r
            i_end = i_start + self._r
            j_start = self._r
            j_end = j_start + self._height
            out = self._ma[i_start: i_end, j_start: j_end].copy()
            ids.append(self._right_nbr.receive.remote(self._index,out))

        if self._id_x > 0:
            i_start = self._r
            i_end = i_start + self._r
            j_start = self._r
            j_end = j_start + self._height
            out = self._ma[i_start: i_end, j_start: j_end].copy()
            ids.append(self._left_nbr.receive.remote(self._index,out))

        return ids

    def receive(self, id, data):
        if id == self._top_nbr_id:
            i_start = self._r
            i_end = i_start + self._width
            j_start = self._height + self._r
            j_end = j_start + self._r
            self._ma[i_start: i_end, j_start: j_end] = data

        if id == self._bottom_nbr_id:
            i_start = self._r
            i_end = i_start + self._width
            j_start = 0
            j_end = j_start + self._r
            self._ma[i_start: i_end, j_start: j_end] = data

        if id == self._left_nbr_id:
            i_start = 0
            i_end = i_start + self._r
            j_start = self._r
            j_end = j_start + self._height
            self._ma[i_start: i_end, j_start: j_end] = data

        if id == self._right_nbr_id:
            i_start = self._width + self._r
            i_end = i_start + self._r
            j_start = self._r
            j_end = j_start + self._height
            self._ma[i_start: i_end, j_start: j_end] = data

    def apply_stencil_operator(self):
        A = self._ma
        B = self._mb
        W = self._weight
        r = self._r
        # for A index
        i_start = max(self._i_start, self._r) - self._i_start + self._r
        i_end = min(self._i_end, self._order - self._r) - self._i_start + self._r
        j_start = max(self._j_start, self._r) - self._j_start + self._r
        j_end = min(self._j_end, self._order - self._r) - self._j_start + self._r

        # for B index
        i_b_s = i_start - self._r
        i_b_e = i_end - self._r
        j_b_s = j_start - self._r
        j_b_e = j_end - self._r

        B[i_b_s: i_b_e, j_b_s: j_b_e] += W[r, r] * A[i_start: i_end, j_start: j_end]
        for s in range(1, self._r + 1):
            B[i_b_s: i_b_e, j_b_s: j_b_e] += W[r, r - s] * A[i_start: i_end, j_start - s: j_end - s] \
                                           + W[r, r + s] * A[i_start: i_end, j_start + s: j_end + s] \
                                           + W[r - s, r] * A[i_start - s: i_end - s, j_start: j_end] \
                                           + W[r + s, r] * A[i_start + s: i_end + s, j_start: j_end] \

        self._ma[self._r: self._width + self._r, self._r: self._height + self._r] += 1

    def abserror(self):
        i_start = max(self._i_start, self._r) - self._i_start + self._r
        i_end = min(self._i_end, self._order - self._r) - self._i_start + self._r
        j_start = max(self._j_start, self._r) - self._j_start + self._r
        j_end = min(self._j_end, self._order - self._r) - self._j_start + self._r
        m = self._mb[i_start: i_end, j_start: j_end]
        return np.linalg.norm(np.reshape(m, m.size), ord=1)


def main():
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python version = ', str(sys.version_info.major) + '.' + str(sys.version_info.minor))
    print('Ray version    = ', ray.__version__)
    print('Numpy version  = ', np.version.version)

    print('Parallel Research Kernels version ')  # , PRKVERSION
    print('Python stencil execution on 2D grid')

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./stencil <# iterations> <array dimension> <num of processes>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    num_procs = int(sys.argv[3])

    if len(sys.argv) > 4:
        r = int(sys.argv[4])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
        if (2 * r + 1) > n:
            sys.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2  # radius=2 is what other impls use right now

    print('Grid size            = ', n)
    print('Radius of stencil    = ', r)
    print('Type of stencil      = ', 'star')
    print('Data type            = double precision')
    print('Compact representation of stencil loop body')
    print('Number of iterations = ', iterations)
    print('Number of processes  = ', num_procs)

    ray.init(address='auto', node_ip_address='sr231', redis_password='123')

    num_procs_x, num_procs_y = factor(num_procs)

    executors = []
    for i in range(num_procs):
        id_x = i % num_procs_x
        id_y = i // num_procs_x

        i_start, i_end = compute_star_end(id_x, n, num_procs_x)
        width = i_end - i_start + 1
        if width == 0:
            print(f"ERROR: rank {i} has no work to do")
            sys.exit(1)

        j_start, j_end = compute_star_end(id_y, n, num_procs_y)
        height = j_end - j_start + 1
        if height == 0:
            print(f"ERROR: rank {i} has no work to do")
            sys.exit(1)

        if width < r or height < r:
            print(f"ERROR: rank {i} has work tile smaller then stencil radius")
            sys.exit(1)

        executors.append(Executor.remote(n, r, i, num_procs, num_procs_x, num_procs_y,
                                         i_start, i_end, j_start, j_end))

    for i in range(num_procs):
        id_x = i % num_procs_x
        id_y = i // num_procs_x

        left = executors[i - 1] if id_x > 0 else None
        right = executors[i + 1] if id_x < num_procs_x - 1 else None
        top = executors[i + num_procs_x] if id_y < num_procs_y - 1 else None
        bottom = executors[i - num_procs_x] if id_y > 0 else None

        ray.get(executors[i].register.remote(left, right, top, bottom))

    for k in range(iterations + 1):
        # start timer after a warmup iteration
        if k < 1:
            t0 = timer()

        idss = ray.get([executors[i].execute.remote() for i in range(num_procs)])
        for ids in idss:
            ray.get(ids)

        ray.get([executors[i].apply_stencil_operator.remote() for i in range(num_procs)])

    t1 = timer()
    stencil_time = t1 - t0

    # ******************************************************************************
    # * Analyze and output results.
    # ******************************************************************************

    abserrors = [ray.get(executors[i].abserror.remote()) for i in range(num_procs)]
    norm = sum(abserrors)
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
