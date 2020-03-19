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
import ray

import time


@ray.remote
class TransposeExecutor:
    def __init__(self, index, num_procs, order):
        import numpy as np
        self._index = index
        self._num_procs = num_procs
        self._order = order
        self._block_order = order // num_procs
        self._index_start = self._index * self._block_order
        self._ma = np.fromfunction(lambda i, j: (self._index_start + j) * self._order + i,
                                   (self._order, self._block_order), dtype=np.float)
        self._mb = np.zeros((self._order, self._block_order), dtype=np.float)

    def register_executor(self, handlers):
        self._handlers = handlers

    def transpose(self):
        start = self._index_start
        end = start + self._block_order
        self._mb[start: end, :] += self._ma[start: end, :].T
        self._ma[start: end, :] += 1.0

        ids = []
        for phases in range(1, self._num_procs):
            to_index = (self._index + phases) % self._num_procs
            assert to_index != self._index
            start = to_index * self._block_order
            end = start + self._block_order
            work_out = self._ma[start: end, :].copy()
            ids.append(self._handlers[to_index].receive.remote(self._index, work_out.T))
            self._ma[start: end, :] += 1.0

        return ids

    def receive(self, from_index, data):
        start = from_index * self._block_order
        end = start + self._block_order
        self._mb[start: end, ] += data

    def get_matrix(self):
        # For debug purpose
        return self._ma, self._mb

    def abserr(self, iterations):
        A = np.fromfunction(lambda i, j: i * self._order + j + self._index_start,
                            (self._order, self._block_order), dtype=np.float)
        A = A * (iterations + 1.0) + (iterations + 1.0) * iterations / 2.0
        abserr = np.linalg.norm(np.reshape(self._mb - A, self._order * self._block_order), ord=1)
        return abserr


def main():
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python version = ', str(sys.version_info.major) + '.' + str(sys.version_info.minor))
    print('Numpy version  = ', np.version.version)
    print('Ray version = ', ray.__version__)

    print('Parallel Research Kernels version ')  # , PRKVERSION
    print('Python Ray Matrix transpose: B = A^T')

    if len(sys.argv) != 4:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order> <num_procs>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])
    if order < 1:
        sys.exit("ERROR: order must be >= 1")

    num_procs = int(sys.argv[3])
    if order % num_procs != 0:
        sys.exit("ERROR: order must be multiple times of num_procs")

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', order)
    print('Number or processes  = ', num_procs)

    ray.init(address='auto', node_ip_address='sr231', redis_password='123')  # ray init

    # create all transpose executors
    executors = [TransposeExecutor.remote(i, num_procs, order) for i in range(num_procs)]
    # register the remote transpose executors
    for i in range(num_procs):
        tmp = executors[i]
        executors[i] = None
        ray.get(tmp.register_executor.remote(executors))  # wait the register finish
        executors[i] = tmp

    for k in range(0, iterations + 1):

        if k == 1:
            t0 = time.time()

        idss = ray.get([executors[i].transpose.remote() for i in range(num_procs)])
        for ids in idss:
            ray.get(ids)

    t1 = time.time()
    trans_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    abserrs = ray.get([executors[i].abserr.remote(iterations) for i in range(num_procs)])
    abserr = sum(abserrs)

    ray.shutdown()

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
