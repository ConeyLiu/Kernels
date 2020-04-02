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

# place this controller on the same node as driver
@ray.remote(resources={"0": 1})
class Controller:
    def __init__(self, iterations):
        self._iterations = iterations + 1 # 1 for warm up
        self._executors = []
        self._cur_iteration = 0
        self._cur_registered = 0
        self._executor_durations = []
        self._finished = False
        self._error = None
    
    def set_up(self, executors):
        self._executors = executors
        self._executor_durations = [0] * len(self._executors)
    
    def transpose(self):
        for e in self._executors:
            e.transpose.remote(-1, self._cur_iteration, None)

    def register_durations(self, worker_index, iteration_index, duration):
        if self._finished:
            return

        if iteration_index != self._cur_iteration:
            # error
            self._finished = True
            self._error = (self._cur_iteration, worker_index, iteration_index)
        
        if self._cur_iteration > 0:
            self._executor_durations[worker_index] += duration

        self._cur_registered += 1
        if self._cur_registered == len(self._executors):
            self._cur_iteration += 1
            self._cur_registered = 0

            if self._cur_iteration < self._iterations:
                self.transpose()
            else:
                self._finished = True
    
    def get_results(self):
        max_durations = 0
        if self._finished:
            max_durations = np.max(self._executor_durations)
        return self._finished, self._error, max_durations


@ray.remote(num_cpus=1)
class TransposeExecutor:
    def __init__(self, index, num_procs, order, controller):
        import numpy as np
        self._index = index
        self._num_procs = num_procs
        self._controller = controller
        self._order = order
        self._block_order = order // num_procs
        self._index_start = self._index * self._block_order
        self._ma = np.fromfunction(lambda i, j: (self._index_start + j) * self._order + i,
                                   (self._order, self._block_order), dtype=np.float)
        self._mb = np.zeros((self._order, self._block_order), dtype=np.float)

        self._iteration_index = None
        self._iteration_start = None
        self._phases = 1
        self._transposed = False # indicate whether tranpose self

    def register_executor(self, handlers):
        self._handlers = handlers

    def tranpose(self, from_index, iteration_index, data):
        if not self._iteration_start:
            self._iteration_start = time.time()
        
        if iteration_index == -1:
            # call from Controller, tranpose itself
            self._iteration_index = iteration_index
            start = self._index_start
            end = start + self._block_order
            self._mb[start: end, :] += self._ma[start: end, :].T
            self._ma[start: end, :] += 1.0
            self._transposed = True
        else:
            # call from other executors
            start = from_index * self._block_order
            end = start + self._block_order
            self._mb[start: end, ] += data.T
        
        if self._phases < self._num_procs:
            to_index = (self._index + self._phases) % self._num_procs
            assert to_index != self._index
            start = to_index * self._block_order
            end = start + self._block_order
            work_out = self._ma[start: end, :]
            self._handlers[to_index].tranpose.remote(self._index, None, work_out)
            self._ma[start: end, :] += 1.0
            self._phases += 1

        if self._transposed and (self._phases == self._num_procs):
            duration = time.time() - self._iteration_start
            self._controller.register_durations.remote(self._index, self._iteration_index, duration)
            self._phases = 1
            self._iteration_index = None
            self._iteration_start = None
            self._transposed = False
            return

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

    # set up controller
    controller = Controller.remote(iterations)
    # create all transpose executors
    executors = [TransposeExecutor.remote(i, num_procs, order, controller) for i in range(num_procs)]
    # register the remote transpose executors
    for i in range(num_procs):
        tmp = executors[i]
        executors[i] = None
        ray.get(tmp.register_executor.remote(executors))  # wait the register finish
        executors[i] = tmp

    controller.set_up.remote(executors)

    controller.transpose.remote()

    results = ray.get(controller.get_results.remote())
    while not results[0]:
        time.sleep(1)
        results = ray.get(controller.get_results.remote())
    
    if results[1]:
        print("ERROR iteration:", results[1])
        sys.exit(-1)

    trans_time = results[2]  # max duration

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
