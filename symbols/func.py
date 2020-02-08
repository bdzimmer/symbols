"""

Utilities for functional programming.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import inspect


def pipe(*funcs):
    """compose single argument functions left to right (evaluated
    right to left)"""

    def piped(*args, **kwargs):
        """helper"""
        arg = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            arg = func(arg)
        return arg

    piped.__signature__ = inspect.signature(funcs[0])

    return piped


def apply(x, *funcs):
    """apply a series of functions"""
    return pipe(*funcs)(x)


def unzip(iterable):
    """unzip a list of tuples into several lists"""
    return zip(*iterable)


def function_param_order(func):
    """get names of args infunction signature"""
    return inspect.signature(func).parameters.keys()


def partial(func):
    def wrapper(*args, **kwargs):
        return lambda x: func(x, *args, **kwargs)
    return wrapper
