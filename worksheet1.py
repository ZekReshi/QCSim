from qcsim import *

# Exercise 1
print("# Exercise 1")
# 1.1
print("# 1.1")
print(e0)
print(e1)

# 1.2
print("# 1.2")
print(plus)
print(minus)

# 1.3
print("# 1.3")
print(e0.is_normalized())
print(minus.is_normalized())
print(Qubit(.1, .1).is_normalized())

# 1.4
print("# 1.4")
print(e0.probability_of(0))
print(plus.probability_of(0))

# 1.5
print("# 1.5")
print(e0.probability_of(1))
print(plus.probability_of(1))

# Exercise 2
print("# Exercise 2")
# 2.1
print("# 2.1")
print(I)
print(Z)
print(X)
print(H)

# 2.2
print("# 2.2")
print(I * e0)
print(I * e1)
print(X * e0)
print(X * e1)
print(Z * e0)
print(Z * e1)
print(H * e0)
print(H * e1)

# 2.3
print("# 2.3")
print(I * I * e0)
print(I * I * e1)
print(X * X * e0)
print(X * X * e1)
print(Z * Z * e0)
print(Z * Z * e1)
print(H * H * e0)
print(H * H * e1)

# 2.4
print("# 2.4")
print(I.is_own_inverse())
print(X.is_own_inverse())
print(Z.is_own_inverse())
print(H.is_own_inverse())
print(Gate(.5, .5, .5, .5).is_own_inverse())

# 2.5
print("# 2.5")
print(I.valid_quantum_operation())
print(X.valid_quantum_operation())
print(Z.valid_quantum_operation())
print(H.valid_quantum_operation())
print(Gate(.5, .5, .5, .5).valid_quantum_operation())

# 2.6
print("# 2.6")
# TODO

# 2.7
print("# 2.7")
print(Gate.compose([I, H, H, I]))
print(Gate.compose([I, H, I]))
print(Gate.compose([H, X, H]))
print(Gate.compose([H, Z, H]))

# 2.8
print("# 2.8")
print(Gate.compose([H, H]) == I)
print(Gate.compose([Z, Z]) == I)
print(Gate.compose([I, I]) == I)

# 2.9
print("# 2.9")
print(Gate.compose([H, X, H]) == Z)
print(Gate.compose([H, Z, H]) == X)

# Exercise 3
# 3.1
print("# 3.1")
print((H * e0).probability_of(0))

# 3.2
print("# 3.2")
print((H * e0).probability_of(1))

# 3.3
print("# 3.3")
print("Yes")

# 3.4
print("# 3.4")
# No
# TODO

# Exercise 4
print("# Exercise 4")
# TODO
