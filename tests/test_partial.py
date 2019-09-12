import pytest

from collections import namedtuple

from sciutils.partial import ProperPartial

_foo_args_fields = ['p1', 'p2', 'k1', 'k2', 'args', 'kwargs']
foo_args = namedtuple(
    'foo_args',
    field_names=_foo_args_fields,
    defaults=[None] * (len(_foo_args_fields) - 2) + [(), {}]
)


def foo(p1, p2, k1=None, k2=None, *args, **kwargs):
    return foo_args(p1=p1, p2=p2, k1=k1, k2=k2, args=args, kwargs=kwargs)


@pytest.fixture(scope='module')
def foo_fun():
    return foo, foo_args


def test_partial_1st_pos(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, 1)
    res = pp(2)

    assert res == args(1, 2)


def test_partial_2nd_pos(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, p2=2)
    res = pp(1)

    assert res == args(1, 2)


def test_partial_key(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, k2=4)
    res = pp(1, 2)

    assert res == args(1, 2, None, 4)


def test_partial_pos_and_key(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, p2=2, k2=4)
    res = pp(1, 3)

    assert res == args(1, 2, 3, 4)


def test_partial_pos_as_key(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, p2=2, k2=4)
    res = pp(1, 3)

    assert res == args(1, 2, 3, 4)


def test_partial_key_as_pos(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, 1)
    res = pp(2, 3, 4)

    assert res == args(1, 2, 3, 4)


# TODO: implement this
@pytest.mark.skip
def test_partial_args(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, 1, 2, 3, 4, 5)
    res = pp(6, 7, 8, 9, 10)

    assert res == args(1, 2, 3, 4, (5, 6, 7, 8, 9, 10), {})


def test_partial_kwargs(foo_fun):
    fun, args = foo_fun

    pp = ProperPartial(fun, 1, k2=4, xkw1=-1)
    res = pp(2, k1=3, xkw2=-2)
    assert res == args(1, 2, 3, 4, (), {'xkw1': -1, 'xkw2': -2})