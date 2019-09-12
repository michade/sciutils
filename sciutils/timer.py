# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import time


class Timer(object):
    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.time()

    def stop(self) -> None:
        self._stop_time = time.time()
        if not self.has_started:
            self._start_time = self._stop_time

    @property
    def has_started(self) -> bool:
        return self._start_time is not None

    @property
    def has_stopped(self) -> bool:
        return self._stop_time is not None

    @property
    def is_running(self) -> bool:
        return self.has_started and not self.has_stopped

    @property
    def elapsed(self) -> float:
        if self.has_started:
            if not self.has_stopped:
                return time.time() - self._start_time
            else:
                return self._stop_time - self._start_time
        return 0.0

    def __enter__(self) -> Timer:
        self.reset()
        self.start()
        return self

    def __exit__(self, type_, value, traceback) -> bool:
        self.stop()
        return False  # reraise any exception

    def reset(self):
        self._start_time = None
        self._stop_time = None

    def __repr__(self):
        return f'Timer({self._start_time},{self._stop_time})'
