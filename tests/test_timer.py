import pytest

import time

from sciutils.timer import Timer


@pytest.fixture
def timer():
    return Timer()


def test_timer_init(timer):
    assert not timer.has_started
    assert not timer.is_running
    assert not timer.has_stopped


def test_timer_start(timer):
    timer.start()

    assert timer.has_started
    assert timer.is_running
    assert not timer.has_stopped


def test_timer_stop(timer):
    timer.start()
    timer.stop()

    assert timer.has_started
    assert not timer.is_running
    assert timer.has_stopped


def test_timer_elapsed(timer):
    timer.start()
    time.sleep(0.01)  # TODO: would be better with some sort of mocks for time module...

    assert timer.elapsed >= 0.01


def test_timer_elapsed_not_started(timer):
    time.sleep(0.01)

    assert timer.elapsed == 0.0


def test_timer_elapsed_stopped(timer):
    timer.start()
    time.sleep(0.01)
    timer.stop()
    time.sleep(0.01)

    assert timer.elapsed >= 0.01
    assert timer.elapsed < 0.02


def test_timer_start_reset(timer):
    timer.start()
    timer.reset()

    assert not timer.has_started
    assert not timer.is_running
    assert not timer.has_stopped


def test_timer_start_stop_reset(timer):
    timer.start()
    timer.stop()
    timer.reset()

    assert not timer.has_started
    assert not timer.is_running
    assert not timer.has_stopped


def test_timer_context_manager():
    with Timer() as timer:
        time.sleep(0.01)
    time.sleep(0.01)

    assert timer.has_started
    assert not timer.is_running
    assert timer.has_stopped
    assert timer.elapsed >= 0.01