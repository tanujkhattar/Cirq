import numpy as np

import cirq

# yapf: disable
SWAP = np.array([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
# yapf: enable


def time_kak_decomposition(target):
    cirq.kak_decomposition(target)


time_kak_decomposition.params = [
    [np.eye(4), SWAP, SWAP * 1j, CZ, CNOT, SWAP.dot(CZ)]
]  # + [cirq.testing.random_unitary(4) for _ in range(10)]
time_kak_decomposition.params_names = ["gate"]
