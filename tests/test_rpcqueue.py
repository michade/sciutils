import pytest


from sciutils.jobs import RpcQueue, local
from collections import deque


class MockQueue(object):
    def __init__(self):
        self._queue = deque()

    def put(self, obj):
        self._queue.append(obj)

    def get(self):
        return self._queue.popleft()

    def __len__(self):
        return len(self._queue)


MOCK_QUEUE_CLASS = MockQueue


class MockObject(object):
    def __init__(self, rpc_id):
        self.rpc_id = rpc_id

    def mock_method(self, *args, **kwargs):
        return self, args, kwargs

    @local
    def remote_method(self, *args, **kwargs):
        return self, args, kwargs


@pytest.fixture
def queue():
    return RpcQueue(queue=MOCK_QUEUE_CLASS)


@pytest.fixture
def queue_with_obj():
    q = RpcQueue(queue=MOCK_QUEUE_CLASS)
    mo = MockObject(123)
    q.register(mo)
    return q, mo


def test_initial_state(queue):
    assert queue.n_producers == 0
    assert queue.n_consumers == 0
    assert not queue.is_finished


def test_register(queue_with_obj):
    queue, mo = queue_with_obj
    assert queue.get_object_key(mo) == mo.rpc_id


def test_counting_producers(queue):
    queue.add_producer()
    assert queue.n_producers == 1
    assert queue.n_consumers == 0
    assert not queue.is_finished
    queue.add_producer()
    assert queue.n_producers == 2
    assert queue.n_consumers == 0
    assert not queue.is_finished
    queue.remove_producer()
    assert queue.n_producers == 1
    assert queue.n_consumers == 0
    assert not queue.is_finished
    queue.remove_producer()
    assert queue.n_producers == 0
    assert queue.n_consumers == 0
    assert not queue.is_finished


def test_counting_consumers(queue):
    queue.add_consumer()
    assert queue.n_producers == 0
    assert queue.n_consumers == 1
    assert not queue.is_finished
    queue.add_consumer()
    assert queue.n_producers == 0
    assert queue.n_consumers == 2
    assert not queue.is_finished
    queue.remove_consumer()
    assert queue.n_producers == 0
    assert queue.n_consumers == 1
    assert not queue.is_finished
    queue.remove_consumer()
    assert queue.n_producers == 0
    assert queue.n_consumers == 0
    assert not queue.is_finished


def test_put(queue_with_obj):
    queue, mo = queue_with_obj
    old_len = len(queue._queue)
    queue.put(mo.mock_method)
    assert len(queue._queue) == old_len + 1


def test_put_raw(queue_with_obj):
    queue, mo = queue_with_obj
    old_len = len(queue._queue)
    queue.put_raw(mo, mo.mock_method, (), {})
    assert len(queue._queue) == old_len + 1


def test_get(queue_with_obj):
    queue, mo = queue_with_obj
    args = (1, 2)
    kwargs = {'kw1': 1, 'kw2': 2}
    queue.put(mo.mock_method, *args, **kwargs)

    method, args_res, kwargs_res = queue.get()
    assert method.__self__ == mo
    assert method == mo.mock_method
    assert args_res == args
    assert kwargs_res == kwargs


def test_finish(queue):
    n_cons = 3
    for _ in range(n_cons):
        queue.add_consumer()

    queue.finish()
    assert queue.is_finished
    for i in range(n_cons):
        assert queue._queue.get() == RpcQueue.FINISHED_TOKEN


def test_wrap_method(queue_with_obj):
    queue, mo = queue_with_obj
    args = (1, 2)
    kwargs = {'kw1': 1, 'kw2': 2}

    wrapped = queue.wrap_method(mo.remote_method)
    assert callable(wrapped)

    wrapped(mo, *args, **kwargs)
    remote_res = queue.get()
    assert remote_res == (mo.remote_method, args, kwargs)


def test_wrap_target(queue_with_obj):
    queue, mo = queue_with_obj
    args = (1, 2)
    kwargs = {'kw1': 1, 'kw2': 2}

    wrapped = queue.wrap_target(mo)

    local_res = mo.mock_method(*args, **kwargs)
    assert local_res == (mo, args, kwargs)

    assert mo.remote_method(*args, **kwargs) is None
    remote_res = queue.get()
    assert remote_res == (mo.remote_method, args, kwargs)