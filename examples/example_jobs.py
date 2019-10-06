#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes
import logging
import multiprocessing as mp
import os
import random
import time
import numpy as np

from sciutils.jobs import Job, JobRunner, local, create_progress_tracker
from sciutils.timer import Timer


class SumJob(Job):
    def __init__(self, data):
        super(SumJob, self).__init__()
        self._data = data
        self.sum = 0

    def run(self, split_size):
        i = 0
        while i < len(self._data):
            self.schedule_job(self.partial_sum, i, i + split_size)
            i += split_size

    def partial_sum(self, start: int, end: int):
        ps = sum(self._data[start:end])
        self.print(f'Partial sum {ps} (from pid: {self.worker.pid})')
        self.add_partial_sum(ps)

    @local
    def add_partial_sum(self, ps: int):
        self.sum += ps


def sum_example():
    n = 1000
    data = list(range(1, n + 1))
    random.seed(123)
    random.shuffle(data)

    sum_job = SumJob(data)
    with JobRunner(4) as runner:
        runner.schedule_job(sum_job, 100)
    print(f'Result: {sum_job.sum}, correct: {n * (n + 1) // 2}')


class ChainsJob(Job):
    def __init__(self, data, multiplier=1.0):
        super(ChainsJob, self).__init__()
        self._data = data
        self._multiplier = multiplier

    def run(self, i, j=0):
        size = self._data[i][j]
        time.sleep(size * self._multiplier)
        self.print(f'[{i}, {j}] done (size={size}, from pid: {self.worker.pid})')
        j += 1
        if j < len(self._data[i]):
            self.chain_job(self.run, i, j)


def chains_example():
    data = [
        [10, 10],
    ] + 5 * [[1] * 5]
    sums = [sum(c) for c in data]
    print(f'sums={sums}')
    print(f'(max_chain={max(sums)}, total = {sum(sums)})')

    with Timer() as t:
        with JobRunner(4) as runner:
            for i in range(len(data)):
                runner.schedule_job(ChainsJob(data, 0.05), i)
    print(f'Done in {t.elapsed:.2f}s.')


class MergeSortJob(Job):
    def __init__(self):
        super(MergeSortJob, self).__init__()
        self._done = {}

    def run(self, data, path=()):
        if len(data) < 2:
            self.on_merged(data, path)
        elif len(data) == 2:
            self.on_merged((min(data), max(data)), path)
        else:
            k = len(data) // 2
            self.schedule_job(self.run, data[:k], path + (0,))
            self.schedule_job(self.run, data[k:], path + (1,))

    @local
    def on_merged(self, data, path):
        if len(path) == 0:
            self.print(f'Done: {data}')
            return
        side = path[-1]
        other_path = path[:-1] + ((side + 1) % 2,)
        other_data = self._done.get(other_path, None)
        if other_data is not None:
            if side == 0:
                self.schedule_job(self.merge, data, other_data, path[:-1])
            else:
                self.schedule_job(self.merge, other_data, data, path[:-1])
            del self._done[other_path]
        else:
            self._done[path] = data

    def merge(self, left_data, right_data, path):
        i_left = 0
        i_right = 0
        n_left = len(left_data)
        n_right = len(right_data)
        result = []
        while i_left < n_left or i_right < n_right:
            if i_left == n_left:
                val = right_data[i_right]
                i_right += 1
            elif i_right == n_right:
                val = left_data[i_left]
                i_left += 1
            elif left_data[i_left] < right_data[i_right]:
                val = left_data[i_left]
                i_left += 1
            else:
                val = right_data[i_right]
                i_right += 1
            result.append(val)
        self.on_merged(result, path)


def merge_sort_example():
    n = 32
    data = list(range(1, n + 1))
    random.seed(456)
    random.shuffle(data)
    print(data)

    merge_sort_job = MergeSortJob()
    with JobRunner(4) as runner:
        runner.schedule_job(merge_sort_job, data)


class SharedMemoryMergeSortJob(Job):
    def __init__(self, data):
        super(SharedMemoryMergeSortJob, self).__init__()
        self._done = {}
        self._data = data

    def run(self, i=0, j=None, path=()):
        if j is None:
            self.print('Started: ', self._data)
            j = len(self._data)
        if j - i < 2:
            self.on_merged(i, j, path)
        elif j - i == 2:
            data = self._data
            if data[i] > data[i + 1]:
                tmp = data[i + 1]
                data[i + 1] = data[i]
                data[i] = tmp
            self.on_merged(i, j, path)
        else:
            k = (j + i) // 2
            self.schedule_job(self.run, i, k, path + (0,))
            self.schedule_job(self.run, k, j, path + (1,))

    @local
    def on_merged(self, i, j, path):
        if len(path) == 0:
            self.print('Done: ', self._data)
            return
        side = path[-1]
        other_path = path[:-1] + ((side + 1) % 2,)
        other_coords = self._done.get(other_path, None)
        if other_coords is not None:
            if side == 0:
                self.schedule_job(self.merge, i, j, other_coords[1], path[:-1])
            else:
                self.schedule_job(self.merge, other_coords[0], i, j, path[:-1])
            del self._done[other_path]
        else:
            self._done[path] = (i, j)

    def merge(self, i, mid, j, path):
        i_left = 0
        i_right = mid - i
        end_left = i_right
        end_right = j - i
        i_res = i
        data = self._data[i:j].copy()
        result = self._data
        while i_res < j:
            if i_left == end_left:
                val = data[i_right]
                i_right += 1
            elif i_right == end_right:
                val = data[i_left]
                i_left += 1
            elif data[i_left] < data[i_right]:
                val = data[i_left]
                i_left += 1
            else:
                val = data[i_right]
                i_right += 1
            result[i_res] = val
            i_res += 1
        self.on_merged(i, j, path)


def shared_memory_merge_sort_example():
    n = 32
    data_type = ctypes.c_int
    raw_array = mp.RawArray(data_type, n)
    np_array = np.frombuffer(raw_array, dtype=data_type).reshape(n)
    np.copyto(np_array, np.arange(1, n + 1))  # can also use initializer?
    np.random.seed(456)
    np.random.shuffle(np_array)

    merge_sort_job = SharedMemoryMergeSortJob(np_array)
    with JobRunner(2) as runner:
        runner.schedule_job(merge_sort_job)


class MockException(Exception):
    pass


class ExceptionJob(Job):
    def __init__(self, cls=MockException):
        super(ExceptionJob, self).__init__()
        self._cls = cls

    def run(self, s, k, dt):
        self.print(f'{s}: {k}...')
        if k > 0:
            time.sleep(dt)
            self.schedule_job(self.run, s, k - 1, dt)
        else:
            self.chain_job(self.run, 'CHAINED', k + 2, dt)
            raise self._cls("Mock exception")


def exception_example():
    try:
        with JobRunner(4, suppress_exceptions=True) as runner:
            runner.schedule_job(ExceptionJob(MockException), 'A', 10, 0.1)
            runner.schedule_job(ExceptionJob(MockException), 'B', 3, 0.1)
    except MockException as ex:
        print(ex)
    print('Exited.')


class LoggingJob(Job):
    def run(self, s, k, dt):
        self.print(f'{s}: {k}...')
        self.logger.warning('warning %s %d', s, k)
        self.logger.info('info %s %d', s, k)
        self.logger.debug('debug %s %d', s, k)
        if k > 0:
            time.sleep(dt)
            self.schedule_job(self, s, k - 1, dt)


def logging_example():
    logging.basicConfig(level=logging.DEBUG)

    log = logging.getLogger()
    log.warning('Normal logger warning...')

    logfile = './logging_test.txt'
    if os.access('.', os.W_OK):
        print('>>> Can write log file.')
        handlers = JobRunner.basic_logging_handlers(logging.WARNING, logging.DEBUG, logfile)
    else:
        print('>>> Cannot write log file.')
        handlers = JobRunner.basic_logging_handlers(logging.DEBUG)

    with JobRunner(2, logging_handlers=handlers) as runner:
        runner.schedule_job(LoggingJob(), 'A', 3, 0.1)
        runner.schedule_job(LoggingJob(), 'B', 3, 0.1)
    print('Exited.')


class DataTransferJob(Job):
    def __init__(self):
        super(DataTransferJob, self).__init__()
        self.attr1 = 0
        self.attr2 = []

    def run(self, k, dt):
        self.print(f'{k}...')
        if k > 0:
            time.sleep(dt)
            self.schedule_job(self, k - 1, dt)
            self.attr1 = k
            self.attr2 = [k]
            self.report_results('attr1', 'attr2')

    def reduce_results(self, current_data, new_data):
        return [
            current_data[0] + new_data[0],
            current_data[1] + new_data[1]
        ]


def data_transfer_example():
    job = DataTransferJob()
    with JobRunner(2) as runner:
        runner.schedule_job(job, 3, 0.1)
    print(f'attr1={job.attr1} attr2={job.attr2}')
    print('Exited.')


class ProgressJob(Job):
    def __init__(self, parts, delay, extra):
        super(ProgressJob, self).__init__()
        self.parts = parts
        self.delay = delay
        self.extra = extra

    def run(self):
        self.set_progress(0, self.parts)
        for i in range(self.parts):
            self.schedule_job(self.do_part, i)

    def do_part(self, i: int):
        time.sleep(self.delay)
        self.increase_progress(1)
        self.part_finished(i)

    @local
    def part_finished(self, i: int):
        if i == 10:
            self.increase_progress(0, self.extra)
            for j in range(self.parts, self.parts + self.extra):
                self.schedule_job(self.do_part, j)


def progress_example():
    interval = 0.2
    tracker = create_progress_tracker(interval, f'Progress example interval={interval:.3f}s')
    with JobRunner(4, progress_tracker=tracker) as runner:
        runner.schedule_job(ProgressJob(20, 0.1, 5))
        runner.schedule_job(ProgressJob(20, 0.1, 5))
    print('Finished - progress example.')


def main():
    mp.set_start_method('fork')  # unnecessary, but for the record (this is the default for linux)
    print(f'Main process pid={os.getpid()}, the start method is "{mp.get_start_method(True)}".')
    # print('SUM ' + '=' * 40)
    # sum_example()
    # print('CHAINS ' + '=' * 40)
    # chains_example()
    # print('MERGE SORT ' + '=' * 40)
    # merge_sort_example()
    # print('SHARED MEMORY MERGE SORT ' + '=' * 40)
    # shared_memory_merge_sort_example()
    # print('EXCEPTION ' + '=' * 40)
    # exception_example()
    # print('DATA TRANSFER ' + '=' * 40)
    # data_transfer_example()
    print('PROGRESS ' + '=' * 40)
    progress_example()
    # print('LOGGING ' + '=' * 40)
    # logging_example()


if __name__ == "__main__":
    main()
