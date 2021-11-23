# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility methods for efficiently preparing two qubit states."""

from typing import Sequence, TYPE_CHECKING
import numpy as np

from cirq import ops, qis
from cirq.optimizers import decompositions

if TYPE_CHECKING:
    import cirq


def _1q_matrices_to_ops(g0, g1, q0, q1, include_identity=False):
    ret = []
    for g, q in zip(map(decompositions.single_qubit_matrix_to_phxz, [g0, g1]), [q0, q1]):
        if g is not None:
            ret.append(g.on(q))
        elif include_identity:
            ret.append(ops.I.on(q))
    return ret


def prepare_two_qubit_state_using_sqrt_iswap(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    state: 'cirq.STATE_VECTOR_LIKE',
    *,
    use_sqrt_iswap_inv: bool = True,
) -> Sequence['cirq.Operation']:
    """Prepares the given 2q state from |00> using at-most 1 √iSWAP gate + single qubit rotations.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.
        use_sqrt_iswap_inv: If True, uses `cirq.SQRT_ISWAP_INV` instead of `cirq.SQRT_ISWAP`.

    Returns:
        List of operations (a single CZ + single qubit rotations) preparing `state` from |00>.
    """
    state = qis.to_valid_state_vector(state, num_qubits=2)
    state = state / np.linalg.norm(state)
    u, s, vh = np.linalg.svd(state.reshape(2, 2))
    if np.isclose(s[0], 1):
        # Product state can be prepare with just single qubit unitaries.
        return _1q_matrices_to_ops(u, vh.T, q0, q1, True)
    alpha = np.arccos(np.sqrt(np.clip(1 - s[0] * 2 * s[1], 0, 1)))
    sign = -1 if use_sqrt_iswap_inv else 1
    iSWAP_state_matrix = np.array(
        [
            [np.cos(alpha), 1j * sign * np.sin(alpha) / np.sqrt(2)],
            [np.sin(alpha) / np.sqrt(2), 0],
        ]
    )
    u_iSWAP, _, vh_iSWAP = np.linalg.svd(iSWAP_state_matrix)
    sqrt_iswap_gate = ops.SQRT_ISWAP_INV if use_sqrt_iswap_inv else ops.SQRT_ISWAP
    return [
        ops.ry(2 * alpha).on(q0),
        sqrt_iswap_gate.on(q0, q1),
        *_1q_matrices_to_ops(
            np.dot(u, np.linalg.inv(u_iSWAP)), np.dot(vh.T, np.linalg.inv(vh_iSWAP.T)), q0, q1
        ),
    ]


def prepare_two_qubit_state_using_cz(
    q0: 'cirq.Qid', q1: 'cirq.Qid', state: 'cirq.STATE_VECTOR_LIKE'
) -> Sequence['cirq.Operation']:
    """Prepares the given 2q state from |00> using at-most 1 CZ gate + single qubit rotations.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.

    Returns:
        List of operations (a single CZ + single qubit rotations) preparing `state` from |00>.
    """
    state = qis.to_valid_state_vector(state, num_qubits=2)
    state = state / np.linalg.norm(state)
    u, s, vh = np.linalg.svd(state.reshape(2, 2))
    if np.isclose(s[0], 1):
        # Product state can be prepare with just single qubit unitaries.
        return _1q_matrices_to_ops(u, vh.T, q0, q1, True)
    alpha = np.arccos(np.clip(s[0], 0, 1))
    CZ_state_matrix = np.array(
        [
            [np.cos(alpha) / np.sqrt(2), np.cos(alpha) / np.sqrt(2)],
            [np.sin(alpha) / np.sqrt(2), -np.sin(alpha) / np.sqrt(2)],
        ]
    )
    u_CZ, _, vh_CZ = np.linalg.svd(CZ_state_matrix)
    return [
        ops.ry(2 * alpha).on(q0),
        ops.H.on(q1),
        ops.CZ.on(q0, q1),
        *_1q_matrices_to_ops(
            np.dot(u, np.linalg.inv(u_CZ)), np.dot(vh.T, np.linalg.inv(vh_CZ.T)), q0, q1
        ),
    ]
