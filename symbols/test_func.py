"""

Test functional programming utils.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from symbols import func


def test_pipe_apply():
    """test composing a bunch of functions"""

    add_one = lambda x: x + 1
    add_two = lambda x: x + 2
    add_three = lambda x: x + 3

    add_six = func.pipe(add_one, add_two, add_three)

    assert add_six(1) == 7
    assert func.apply(1, add_one, add_two, add_three) == 7


def test_unzip():
    """test unzip"""

    list_test = [("a", 1), ("b", 2), ("c", 3)]
    list_expected = [("a", "b", "c"), (1, 2, 3)]

    assert list(func.unzip(list_test)) == list_expected
    assert list(func.unzip([])) == []


def test_function_param_order():
    """test some reflection"""

    def example_function(a_val, b_val, c_val):
        """just for reflection test"""
        return a_val + b_val + c_val

    example_function(1, 2, 3)  # so the linters don't complain

    assert func.function_param_order(example_function) == ["a_val", "b_val", "c_val"]
