"""Various methods for doing evaluation"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def get_nb_accuracy(correct_neg, correct_pos):
    accuracy = (sum(correct_neg) + sum(correct_pos)) / (len(correct_neg) + len(correct_pos))
    return accuracy


def combination(N, k):
    f = math.factorial
    return f(N) / (f(k) * f(N - k))


def sign_test(predictionsA, predictionsB, targets, q=0.5):
    a_better = np.sum((predictionsA == targets) * (predictionsB != targets))
    b_better = np.sum((predictionsA != targets) * (predictionsB == targets))
    null = np.sum(
        (predictionsA != targets) * (predictionsB != targets) + (predictionsA == targets) * (predictionsB == targets))
    assert (a_better + b_better + null == len(targets))

    N = 2 * (null // 2) + a_better + b_better
    k = null // 2 + min(a_better, b_better)
    logger.info("Got sign test values: a_better={0}, b_better={1}, null={2}, N={3}, k={4}".format(
        a_better, b_better, null, N, k
    ))
    p_value = 0
    for i in range(0, k + 1):
        p_value += combination(N, i) * math.pow(q, i) * math.pow(1 - q, N - i)
    p_value = p_value * 2

    return p_value
