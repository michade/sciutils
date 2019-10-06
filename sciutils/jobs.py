# -*- coding: utf-8 -*-
from __future__ import annotations

import curses
import functools
import inspect
import itertools
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import traceback
import typing
from collections import deque, OrderedDict, Counter
from typing import Union, List, Dict, Optional

from .timer import Timer

_LAST_UNIQUE_IDS = Counter()


def get_pretty_unique_id(obj):
    namespace = obj.__class__.__name__
    global _LAST_UNIQUE_IDS
    id_ = _LAST_UNIQUE_IDS[namespace]
    _LAST_UNIQUE_IDS[namespace] += 1
    return namespace, id_


class RpcQueue(object):
    FINISHED_TOKEN = 'FINISHED'

    def __init__(self, queue=mp.Queue):
        if callable(queue):
            queue = queue()
        self._queue = queue
        self._targets: Dict = {}
        self._is_finished = False
        self._n_producers = 0
        self._n_consumers = 0

    def __getitem__(self, key):
        return self._targets[key]

    def __contains__(self, key):
        return key in self._targets

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def n_consumers(self):
        return self._n_consumers

    def add_consumer(self) -> None:
        self._n_consumers += 1

    def remove_consumer(self) -> None:
        self._n_consumers -= 1

    @property
    def n_producers(self):
        return self._n_producers

    def add_producer(self) -> None:
        self._n_producers += 1

    def remove_producer(self) -> None:
        self._n_producers -= 1

    def get_object_key(self, obj):
        return obj.rpc_id

    def _serialize_method(self, method):
        serialized = method.__name__
        return serialized

    def _deserialize_method(self, target, serialized):
        method = getattr(target, serialized)
        return method

    def register(self, obj):
        key = self.get_object_key(obj)
        self._targets[key] = obj

    def unregister(self, obj):
        key = self.get_object_key(obj)
        self.unregister_key(key)

    def unregister_key(self, key):
        del self._targets[key]

    def purge_objects(self):
        keys = list(self._targets.keys())
        for key in keys:
            self.unregister_key(key)

    def put(self, method, *args, **kwargs):
        self.put_raw(method.__self__, method, args, kwargs)

    def put_raw(self, target, method, args, kwargs):
        msg = (self.get_object_key(target), self._serialize_method(method), args, kwargs)
        self._queue.put(msg)

    def get(self):
        msg = self._queue.get()
        if msg == RpcQueue.FINISHED_TOKEN:
            return None
        key, serialized_method, args, kwargs = msg
        target = self._targets[key]
        method = self._deserialize_method(target, serialized_method)
        return method, args, kwargs

    def finish(self):
        self._is_finished = True
        for _ in range(self._n_consumers):
            self._queue.put(RpcQueue.FINISHED_TOKEN)

    def close(self) -> None:
        self._queue.close()

    REMOTE_METHOD_ATTRIBUTE_NAME = '_message_action_'

    def wrap_method(self, method):

        @functools.wraps(method)
        def _wrapper(target_self, *args, **kwargs):
            self.put_raw(target_self, method, args, kwargs)

        return _wrapper

    def wrap_target(self, target):
        members = inspect.getmembers(target, lambda m: hasattr(m, RpcQueue.REMOTE_METHOD_ATTRIBUTE_NAME))
        for name, method in members:
            wrapped = self.wrap_method(method)
            setattr(target, name, wrapped.__get__(target, target.__class__))

    @staticmethod
    def remote_method(message):
        setattr(message, RpcQueue.REMOTE_METHOD_ATTRIBUTE_NAME, True)
        return message


local = RpcQueue.remote_method


class Job(object):
    def __init__(self):
        self.rpc_id = get_pretty_unique_id(self)
        self._progress = 0
        self._max_progress = 1
        self._worker: Optional[Worker] = None
        self._logger: Optional[logging.LoggingAdapter] = None
        self._timer = Timer()
        self._part_timer = Timer()
        self._is_started = False

    def on_started(self, run_info: str):
        self.logger.debug(f'Starting:\t{run_info}')
        self._part_timer.reset()
        if not self._is_started:
            self._is_started = True
            self._timer.start()
            self.on_started_local()
        self._part_timer.start()

    @local
    def on_started_local(self):
        if not self._timer.has_started:
            self.logger.debug(f'Starting.')
            self._timer.start()

    def on_finished(self, ok: bool, run_info: str):
        self._part_timer.stop()
        self.logger.info(f'{"Finished" if ok else "Failed"}:\t{run_info}\t[time={self._part_timer.elapsed:.2f}s, total={self._timer.elapsed:.2f}s]')

    def get_run_info(self, method, *args, **kwargs) -> str:
        extra_info = self.get_run_extra_info(method, *args, **kwargs)
        return f'{method.__name__}({extra_info})'

    def get_run_extra_info(self, method, *args, **kwargs) -> str:
        s = ', '.join(itertools.chain(
            (str(a) for a in args),
            (f'{k}={v}' for k, v in kwargs.items())
        ))
        return s

    def finished(self):
        self.on_finished_local()

    @local
    def on_finished_local(self):
        if not self._timer.has_stopped:
            self._timer.stop()
            self.logger.info(f'Finished in {self._timer.elapsed:.2}s')

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}[{self.rpc_id[1]}]'

    @property
    def worker(self) -> Worker:
        return self._worker

    @property
    def logger(self) -> logging.LoggingAdapter:
        return self._logger

    @worker.setter
    def worker(self, worker):
        if worker != self.worker:
            adapter = logging.LoggerAdapter(
                worker.get_base_logger(),
                {
                    'worker_id': worker.worker_id,
                    'job_name': self.name
                }
            )
            self._logger = adapter
        self._worker = worker

    def print(self, *objs):
        self.worker.print(*objs)

    @typing.no_type_check
    def schedule_job(self, method, *args, **kwargs):
        return self.worker.schedule_job(method, *args, **kwargs)

    @typing.no_type_check
    def chain_job(self, method, *args, **kwargs):
        return self.worker.chain_job(method, *args, **kwargs)

    def get_default_method(self):
        return self.run

    @typing.no_type_check
    def run(self, *args, **kwargs):
        pass

    def report_results(self, *attrs):
        data = [(attr, getattr(self, attr)) for attr in attrs]
        self._assign_attrs(data)

    def reduce_results(self, current_data, new_data) -> List:
        return new_data

    @local
    def _assign_attrs(self, new_data):
        attrs = [attr for attr, _ in new_data]
        current_data = [getattr(self, attr) for attr in attrs]
        merged_data = self.reduce_results(current_data, [val for _, val in new_data])
        for attr, val in zip(attrs, merged_data):
            setattr(self, attr, val)

    @property
    def progress(self):  # Note: values on workers will be null
        return self._progress

    @property
    def max_progress(self):  # Note: values on workers will be null
        return self._max_progress

    @local
    def set_progress(self, progress: int, max_progress: int):  # should be called before anything else
        self._progress = progress
        self._max_progress = max_progress

    @local
    def increase_progress(self, progress_delta: int, max_progress_delta: int = 0):
        self._progress += progress_delta
        self._max_progress += max_progress_delta


class Worker(object):
    def __init__(
            self,
            job_queue: RpcQueue, msg_queue: RpcQueue, log_queue: mp.Queue,
            suppress_exceptions=False
    ):
        self.rpc_id = get_pretty_unique_id(self)
        self._worker_id = self.rpc_id[1]
        self._pid = None
        self._chained_jobs = None
        self._job_queue = job_queue
        self._msg_queue = msg_queue
        self._runner: Union[None, JobRunner] = None
        self._suppress_exceptions = suppress_exceptions

        self._base_logger = logging.Logger(f'JobLogger-{self._worker_id}')
        handler = logging.handlers.QueueHandler(log_queue)
        self._base_logger.addHandler(handler)
        adapter = logging.LoggerAdapter(self.get_base_logger(), {
            'worker_id': self.worker_id,
            'job_name': ''
        })
        self._logger = adapter

    @property
    def runner(self) -> JobRunner:
        return self._runner

    @property
    def logger(self) -> logging.LoggerAdapter:
        return self._logger

    def get_base_logger(self):
        return self._base_logger

    @runner.setter
    def runner(self, runner: JobRunner):
        self._runner = runner

    @property
    def is_remote(self) -> bool:
        return self._pid is not None

    @property
    def pid(self):
        return self._pid

    @property
    def worker_id(self) -> int:
        return self._worker_id

    def run(self) -> None:
        self._pid = os.getpid()
        self._msg_queue.wrap_target(self)
        self._chained_jobs = deque()
        self.logger.debug('Worker started.')
        while True:
            msg = self._job_queue.get()
            if msg is None:
                break
            self._chained_jobs.append(msg)
            while len(self._chained_jobs) > 0:
                job_method, args, kwargs = self._chained_jobs.popleft()
                if not self._run_job(job_method, args, kwargs):
                    self._chained_jobs.clear()
            self.job_finished(job_method.__self__.rpc_id)
        self.logger.debug('Worker finished.')
        self.finished()

    def _run_job(self, job_method, args, kwargs):
        job: Job = job_method.__self__
        ok: bool = False
        if job.worker is None:
            job.worker = self
            self._msg_queue.wrap_target(job)
        try:
            extra_info = job.get_run_info(job_method, *args, **kwargs)
            job.on_started(extra_info)
            job_method(*args, **kwargs)
            ok = True
            job.on_finished(ok, extra_info)
        except Exception:
            e_type, e_value, e_traceback = sys.exc_info()
            text = traceback.format_exception(e_type, e_value, e_traceback)
            self.print_remote_traceback(text)
            if not self._suppress_exceptions:
                job.logger.critical(f'Unsuppressed exception {e_type}:{e_value}.')
                raise
            job.logger.error(f'Exception {e_type}:{e_value}.')
            return False
        finally:
            pass
        return True

    @local
    def print_remote_traceback(self, text):
        sys.stderr.writelines(text)

    def chain_job(self, job_method, *args, **kwargs):
        if hasattr(job_method, 'run'):
            job_method = job_method.run
        self._chained_jobs.append((job_method, args, kwargs))

    def schedule_job(self, job_method, *args, **kwargs):
        if hasattr(job_method, 'run'):
            job_method = job_method.run
        self.job_started(job_method.__self__.rpc_id)
        if hasattr(job_method, 'run'):
            job_method = job_method.run
        self._job_queue.put(job_method, *args, **kwargs)

    def print(self, *objs, info=True) -> None:
        texts = []
        if info:
            texts.append(f'[{self.worker_id:02d}]:\t')
        texts.extend(str(o) for o in objs)
        self._print_local(texts)

    @local
    def _print_local(self, texts) -> None:
        print(*texts)

    @local
    def job_started(self, job_id) -> None:
        self.runner.on_job_started(job_id)

    @local
    def job_finished(self, job_id) -> None:
        self.runner.on_job_finished(job_id)

    @local
    def finished(self) -> None:
        self.runner.on_worker_finished()


class WorkerProcess(mp.Process):
    def __init__(self, worker: Worker):
        super(WorkerProcess, self).__init__(name=f'Worker-{worker.worker_id:02d}')
        self._worker = worker

    @property
    def worker(self):
        return self._worker

    def run(self) -> None:
        self.worker.run()


class ProgressTracker(object):
    def __init__(self, print_progress_interval, header):
        self._progress_timer = Timer()
        self._total_timer = Timer()
        self._print_progress_interval = print_progress_interval
        self._header = header

    def start(self):
        self._total_timer.start()
        self._progress_timer.start()

    def update(self, jobs):
        if self._print_progress_interval is not None and self._progress_timer.elapsed > self._print_progress_interval:
            self._progress_timer.reset()
            self._progress_timer.start()
            self.print(jobs)

    def get_header(self):
        return self._header

    def get_summary_text(self, jobs):
        return f'Running {len(jobs)} jobs. Total time: {self._total_timer.elapsed:.2f}s.'

    def get_job_info(self, job):
        progress = job.progress
        max_progress = job.max_progress
        percent = 100.0 * progress / max_progress
        job_info = f'{job.name}: {percent:6.2f}% [{progress}/{max_progress}]'
        return job_info

    def print_line(self, line):
        sys.stdout.write(line)
        sys.stdout.write('\n')

    def print(self, jobs):
        header = self.get_header()
        if header:
            self.print_line(header)
        summary = self.get_summary_text(jobs)
        self.print_line(summary)
        for job in jobs:
            job_info = self.get_job_info(job)
            self.print_line(job_info)
        self.flush()

    def flush(self):
        sys.stdout.flush()

    def stop(self):
        self._progress_timer.stop()
        self._total_timer.stop()


class QuietProgressTracker(ProgressTracker):
    def __init__(self, print_progress_interval, header):
        super(QuietProgressTracker, self).__init__(print_progress_interval, header)

    def print(self, jobs):
        pass

    def flush(self):
        pass


class CursesProgressTracker(ProgressTracker):
    def __init__(self, print_progress_interval, header):
        super(CursesProgressTracker, self).__init__(print_progress_interval, header)
        self._stdscr = None
        self._window = None
        self._last_line_count = 0
        self._init_curses()

    def _init_curses(self):
        self._stdscr = curses.initscr()
        height, width = self._stdscr.getmaxyx()
        self._window = curses.newwin(height, width)

    def print(self, jobs):
        self._window.erase()
        super(CursesProgressTracker, self).print(jobs)
        self._last_line_count = 1 + len(jobs)

    def flush(self):
        self._window.refresh()

    def print_line(self, line):
        self._window.addstr(line)
        self._window.addstr('\n')

    def _shutdown_curses(self):
        if self._stdscr is not None:
            y, _ = self._window.getyx()
            outputs = '\n'.join(
                self._window.instr(i, 0).decode('ascii')
                for i in range(y)
            )
            curses.endwin()
            self._stdscr = None
            self._window = None
            print(outputs)

    def stop(self):
        super(CursesProgressTracker, self).stop()
        self._shutdown_curses()


def create_progress_tracker(interval, header, quiet=False):
    if quiet:
        return QuietProgressTracker(interval, header)
    try:
        tracker = CursesProgressTracker(interval, header)
    except curses.error:
        logging.warning("Unable to initialize curses.")
        tracker = ProgressTracker(interval, header)
    return tracker


class JobRunner(object):
    def __init__(
            self,
            n_workers: Optional[int],
            suppress_exceptions: bool = False,
            logging_handlers: Union[None, str, int, List] = None,
            progress_tracker=None,
            _queue_class=mp.Queue
    ):
        self.rpc_id = get_pretty_unique_id(self)
        if n_workers is None or n_workers == 0:
            n_workers = mp.cpu_count()
        self._n_workers = n_workers
        self._worker_processes: Optional[List[WorkerProcess]] = None
        self._tmp_job_queue = deque()
        self._job_queue = RpcQueue(queue=_queue_class)
        self._msg_queue = RpcQueue(queue=_queue_class)
        self._log_queue = _queue_class()
        self._log_listener = None
        self._logging_handlers = logging_handlers
        self._local_worker = None
        self._suppress_exceptions = suppress_exceptions
        self._logger = logging.getLogger('JobRunner')
        self._tracked_jobs = OrderedDict()
        if progress_tracker is None:
            self._progress_tracker = ProgressTracker(0.5, None)
        else:
            self._progress_tracker = progress_tracker

    @property
    def is_started(self):
        return self._worker_processes is not None

    @property
    def logger(self):
        return self._logger

    @staticmethod
    def basic_logging_handlers(console_level=None, file_level=None, filename=None):
        handlers = []
        if console_level is not None:
            ch = logging.StreamHandler()
            ch.setLevel(console_level)
            fmt = logging.Formatter('%(levelname)s\t%(worker_id)s\t%(job_name)s:\t%(message)s')
            ch.setFormatter(fmt)
            handlers.append(ch)
        if filename is not None:
            if file_level is None:
                if console_level is None:
                    file_level = logging.INFO
                else:
                    file_level = console_level
            fh = logging.FileHandler(filename, mode='w')
            fh.setLevel(file_level)
            fmt = logging.Formatter('%(levelname)s\t%(worker_id)s\t%(job_name)s:\t%(message)s')
            fh.setFormatter(fmt)
            handlers.append(fh)
        return handlers

    def _config_loggers(self):
        if self._logging_handlers is None:
            self._logging_handlers = []
        elif self._logging_handlers == 'default':
            self._logging_handlers = JobRunner.basic_logging_handlers(logging.INFO)
        elif isinstance(self._logging_handlers, int):
            level = self._logging_handlers
            self._logging_handlers = JobRunner.basic_logging_handlers(level)
        self._log_listener = logging.handlers.QueueListener(
            self._log_queue, *self._logging_handlers, respect_handler_level=True
        )

    def start(self) -> None:
        if self.is_started:
            return
        self._progress_tracker.start()
        # create workers
        self._local_worker = Worker(self._job_queue, self._msg_queue, self._log_queue, self._suppress_exceptions)
        self._worker_processes = [
            WorkerProcess(
                Worker(self._job_queue, self._msg_queue, self._log_queue, self._suppress_exceptions)
            )
            for _ in range(1, self._n_workers + 1)
        ]

        # do fork
        self.logger.debug(f'Starting {len(self._worker_processes)} workers (method: {mp.get_start_method(True)})...')
        for process in self._worker_processes:
            process.start()

        # setup logger
        self.logger.debug(f'Starting remote loggers...')
        self._config_loggers()
        self._log_listener.start()

        # do setup in main process
        self.logger.debug(f'Setting up main process...')
        self._msg_queue.register(self)
        self._msg_queue.add_consumer()
        self._local_worker.runner = self
        for process in self._worker_processes:
            process.worker.runner = self
            self._msg_queue.register(process.worker)
            self.on_worker_started()

        # push scheduled jobs to queue
        while len(self._tmp_job_queue) > 0:
            method, args, kwargs = self._tmp_job_queue.popleft()
            self.schedule_job(method, *args, **kwargs)

        self.logger.debug(f'Runner started.')

    def schedule_job(self, job, *args, **kwargs):
        if hasattr(job, 'run'):
            method = job.run
        else:
            method = job
            job: Job = method.__self__
        self._job_queue.register(job)
        self._msg_queue.register(job)
        self._tracked_jobs[job.rpc_id] = job
        if self.is_started:
            self.logger.info(f'Scheduling job: {job.name}...')
            self.on_job_started(job)
            job.worker = self._local_worker
            self._job_queue.put(method, *args, **kwargs)
        else:
            self._tmp_job_queue.append((method, args, kwargs))

    def _process_message(self) -> None:
        msg = self._msg_queue.get()
        method, args, kwargs = msg
        method(*args, **kwargs)
        self._progress_tracker.update(self._tracked_jobs.values())

    def finish_jobs(self) -> None:
        self.logger.debug(f'Waiting for {self._job_queue.n_producers} jobs to finish...')
        while self._job_queue.n_producers > 0:
            self._process_message()

    def process_messages(self) -> None:
        self.logger.debug(f'Waiting for {self._msg_queue.n_producers} workers to finish sending messages...')
        while self._msg_queue.n_producers > 0:
            self._process_message()

    def close(self) -> None:
        self.logger.debug(f'Shutting down job queue...')
        self._job_queue.finish()
        self.process_messages()
        self.logger.debug(f'Shutting down message queue...')
        self._msg_queue.close()
        self.logger.debug(f'Stopping remote loggers...')
        self._log_listener.stop()
        self.logger.debug(f'Joining worker processes...')
        for process in self._worker_processes:
            process.join()
        self._progress_tracker.print(self._tracked_jobs.values())
        self._progress_tracker.stop()
        self.logger.debug(f'Runner closed.')

    def on_job_started(self, job_id) -> None:
        self._job_queue.add_producer()

    def on_job_finished(self, job_id) -> None:
        self._job_queue.remove_producer()

    def on_worker_started(self) -> None:
        self._job_queue.add_consumer()
        self._msg_queue.add_producer()

    def on_worker_finished(self) -> None:
        self._job_queue.remove_consumer()
        self._msg_queue.remove_producer()

    def __enter__(self) -> JobRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.start()
        self.finish_jobs()
        self.close()
        return False
