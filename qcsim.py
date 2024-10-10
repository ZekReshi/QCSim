import sys
from typing import Tuple, List

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
        return (abs(self.val) ** 2 >= 0).all() and (abs(self.val) ** 2 <= 1).all() and np.isclose((abs(self.val) ** 2).sum(), 1)

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

    def is_own_inverse(self):
        return self * self == I

    def valid_quantum_operation(self):
        return np.linalg.cond(self.tab) < 1/sys.float_info.epsilon

    @staticmethod
    def compose(gates: List['Gate']):
        res = I
        for gate in gates:
            res *= gate
        return res

e0 = Qubit(np.asarray([1.+0j, 0.+0j]))
e1 = Qubit(np.asarray([0.+0j, 1.+0j]))
plus = Qubit(np.asarray([(.5+0j) ** (.5+0j), (.5+0j) ** (.5+0j)]))
minus = Qubit(np.asarray([(.5+0j) ** (.5+0j), -(.5+0j) ** (.5+0j)]))

I = Gate(np.asarray([[1.+0j, 0.+0j], [0., 1.+0j]]))
X = Gate(np.asarray([[0.+0j, 1.+0j], [1.+0j, 0.+0j]]))
Z = Gate(np.asarray([[1.+0j, 0.+0j], [0.+0j, -1.+0j]]))
Y = Gate(np.asarray([[0.+0j, -1.j], [1.j, 0.+0j]]))
H = Gate(np.asarray([[(.5+0j) ** (.5+0j), (.5+0j) ** (.5+0j)], [(.5+0j) ** (.5+0j), -(.5+0j) ** (.5+0j)]]))

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
Y = 
{Y}
H = 
{H}""")
