from typing import Tuple

import numpy as np

class Qubit:
    val: np.ndarray

    def __init__(self, val: np.ndarray):
        assert type(val) == np.ndarray
        assert val.shape == (2,)

        self.val = val

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, Qubit):
            return np.allclose(self.val, other.val)
        return False

    def is_normalized(self):
        return (self.val >= 0).all() and (self.val <= 1).all() and (self.val * self.val).sum() == 1

    def probability_of(self, state: int):
        assert state == 0 or state == 1

        return abs(self.val[state]) ** 2

class Gate:
    tab: np.ndarray

    def __init__(self, tab: np.ndarray):
        assert type(tab) == np.ndarray
        assert tab.shape == (2, 2)
        self.tab = tab

    def __mul__(self, other):
        if isinstance(other, Gate):
            return Gate(np.matmul(self.tab, other.tab))
        if isinstance(other, Qubit):
            return Qubit(np.matmul(self.tab, other.val))
        raise ValueError()

    def __str__(self):
        return str(self.tab)

    def __eq__(self, other):
        if isinstance(other, Gate):
            return np.allclose(self.tab, other.tab)
        return False

e0 = Qubit(np.asarray([1., 0.]))
e1 = Qubit(np.asarray([0., 1.]))
plus = Qubit(np.asarray([.5 ** .5, .5 ** .5]))
minus = Qubit(np.asarray([.5 ** .5, -.5 ** .5]))

I = Gate(np.asarray([[1., 0.], [0., 1.]]))
X = Gate(np.asarray([[0., 1.], [1., 0.]]))
Z = Gate(np.asarray([[1., 0.], [0., -1.]]))
Y = Gate(np.asarray([[0., -1.j], [1.j, 0.]]))
H = Gate(np.asarray([[.5 ** .5, .5 ** .5], [.5 ** .5, -.5 ** .5]]))

if __name__ == '__main__':
    print(f"""e0 = 
{e0}
e1 =
{e1}
plus =
{plus}
minus =
{minus}
I = 
{I}
X = 
{X}
Z = 
{Z}
H = 
{H}""")
