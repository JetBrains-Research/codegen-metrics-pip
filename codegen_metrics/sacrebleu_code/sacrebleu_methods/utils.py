import math


def my_log(num: float) -> float:
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)


def sum_of_lists(lists):
    """Aggregates list of numeric lists by summing."""
    if len(lists) == 1:
        return lists[0]

    # Preserve datatype
    size = len(lists[0])
    init_val = type(lists[0][0])(0.0)
    total = [init_val] * size
    for ll in lists:
        for i in range(size):
            total[i] += ll[i]
    return total
