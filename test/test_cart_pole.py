import unittest

import numpy as np
from numpy.testing import assert_array_equal

from main.cart_pole import discretize_state


class TestCartPole(unittest.TestCase):
    def test_discretize_state_should_scale_observation(self):
        observation_1 = np.array([-2, 0.112, 5.51, 2])
        result_1 = discretize_state(observation_1)
        expected_1 = np.array([0, 0, 6, 3])
        assert_array_equal(result_1, expected_1)

        observation_2 = np.array([2.2221, -5.6312, 0.01, -.1])
        result_2 = discretize_state(observation_2)
        expected_2 = np.array([0, 0, 1, 0])
        assert_array_equal(result_2, expected_2)
