from qcsim import *


def main():
    print(e0)
    print(I * H * H * I * e0)
    print(e0.is_normalized())
    print(Qubit((0.1, 0.1)).is_normalized())
    print(e0.probability_of(0))
    print(e0.probability_of(1))
    print((H * e0).probability_of(0))
    print((H * e0).probability_of(1))

if __name__ == '__main__':
    main()
