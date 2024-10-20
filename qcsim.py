import random
from numbers import Number
from typing import Tuple, List, Union

import numpy as np


class Qubit:
    val: np.ndarray

    def __init__(self, val: Union[np.ndarray, Tuple[Number, Number], Number], b: Number = None):
        if isinstance(val, Number):
            assert isinstance(b, Number)
            val = np.asarray((val, b))
        else:
            assert b is None
            if isinstance(val, tuple):
                assert len(val) == 2
                val = np.asarray(val)

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

    def readout(self):
        return random.choices(range(2), weights=[self.probability_of(0), self.probability_of(1)])[0]

class Gate:
    tab: np.ndarray

    def __init__(self, tab: Union[np.ndarray, Tuple[Tuple[Number, Number], Tuple[Number, Number]], Number], b: Number = None, c: Number = None, d: Number = None):
        if isinstance(tab, Number):
            assert isinstance(b, Number) and isinstance(c, Number) and isinstance(d, Number)
            tab = np.asarray([[tab, b], [c, d]])
        else:
            assert b is None and c is None and d is None
            if isinstance(tab, tuple):
                assert len(tab) == 2
                assert len(tab[0]) == 2
                assert len(tab[1]) == 2
                tab = np.asarray(tab)

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
        print(np.matmul(self.tab, self.conjugate_transpose().tab))
        return np.allclose(np.matmul(self.tab, self.conjugate_transpose().tab), np.identity(2))

    def conjugate_transpose(self):
        return Gate(self.tab.conj().T)

    @staticmethod
    def compose(gates: List['Gate']):
        res = I
        for gate in gates:
            res *= gate
        return res

    @staticmethod
    def commuting(a: 'Gate', b: 'Gate'):
        return np.allclose(np.matmul(a.tab, b.tab), np.matmul(b.tab, a.tab))

    @staticmethod
    def pauli_x(theta: float):
        return Gate(np.cos(theta/2), -1j*np.sin(theta/2), -1j*np.sin(theta/2), np.cos(theta/2))

    @staticmethod
    def pauli_y(theta: float):
        return Gate(np.cos(theta/2), -np.sin(theta/2), np.sin(theta/2), np.cos(theta/2))

    @staticmethod
    def pauli_z(theta: float):
        return Gate(np.exp(-1j*theta/2), 0+0j, 0+0j, np.exp(1j*theta/2))

e0 = Qubit(1.+0j, 0.+0j)
e1 = Qubit(0.+0j, 1.+0j)
plus = Qubit((.5+0j) ** (.5+0j), (.5+0j) ** (.5+0j))
minus = Qubit((.5+0j) ** (.5+0j), -(.5+0j) ** (.5+0j))
iplus = Qubit((.5+0j) ** (.5+0j), 1j * (.5+0j) ** (.5+0j))
iminus = Qubit((.5+0j) ** (.5+0j), 1j * -(.5+0j) ** (.5+0j))

I = Gate(1.+0j, 0.+0j, 0.+0j, 1.+0j)
X = Gate(0.+0j, 1.+0j, 1.+0j, 0.+0j)
Z = Gate(1.+0j, 0.+0j, 0.+0j, -1.+0j)
Y = Gate(0.+0j, -1.j, 1.j, 0.+0j)
H = Gate((.5+0j) ** (.5+0j), (.5+0j) ** (.5+0j), (.5+0j) ** (.5+0j), -(.5+0j) ** (.5+0j))
S = Gate(1+0j, 0, 0, 1j)
T = Gate(1, 0, 0, np.exp(1j*np.pi/4))

if __name__ == '__main__':
    print(f"""e0 = 
{e0}
e1 =
{e1}
plus =
{plus}
minus =
{minus}
iplus =
{iplus}
iminus =
{iminus}
I = 
{I}
X = 
{X}
Z = 
{Z}
Y = 
{Y}
H = 
{H}
S = 
{S}
T = 
{T}""")
