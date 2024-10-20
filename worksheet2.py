import numpy as np

from qcsim import *

# Instructions 1
print("# Instructions 1")

# 1.2
print("# 1.2")
print(iplus)
print(iminus)

# 1.3
print("# 1.3")
print(Y)
print(S)
print(T)

# 1.4
print("# 1.4")
print(Gate.pauli_x(0))
print(Gate.pauli_y(0))
print(Gate.pauli_z(0))

# 1.5
print("# 1.5")
print(Gate.commuting(I, I))
print(Gate.commuting(I, H))
print(Gate.commuting(H, I))
print(Gate.commuting(H, Z))

# Instructions 2
print("# Instructions 2")
def student_action(k: float, state: Qubit):
    return Gate.pauli_y(k * np.pi) * state

k_tot = 493 # Hidden from students
k_list = [k_tot/3, 7 * k_tot /15, k_tot /5] # Each student gets one number.

st = e0
for no in k_list:
    st = student_action(no, st)

print("Even" if st.readout() == 0 else "Odd")
print(Gate.pauli_y(np.pi) * e0)
