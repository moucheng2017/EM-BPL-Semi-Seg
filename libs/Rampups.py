import numpy as np


def sigmoid_rampup(current, rampup_length, limit):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    phase = 1.0 - current / rampup_length
    weight = float(np.exp(-5.0 * phase * phase))
    if weight > limit:
        return float(limit)
    else:
        return weight


def cyclic_sigmoid_rampup(current, rampup_length, limit):
    # calculate the relative current:
    cyclic_index = current // rampup_length
    relative_current = current - cyclic_index*rampup_length
    phase = 1.0 - relative_current / rampup_length
    weight = float(np.exp(-5.0 * phase * phase))
    if weight > limit:
        return float(limit)
    else:
        return weight


def exp_rampup(current, base, limit):
    weight = float(base*(1.05**current))
    if weight > limit:
        return float(limit)
    else:
        weight


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length