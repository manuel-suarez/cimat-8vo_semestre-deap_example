import numpy as np
from numpy import random
from process import unknown


def sample(inputs):
    return np.array([unknown(inp) + random.normal(5.0) for inp in inputs])
