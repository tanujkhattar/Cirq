# Copyright 2023 The Cirq Developers
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
from typing import Optional, List
from attr import frozen
from cirq._compat import cached_property
import cirq
import math

from cirq_ft import infra


@frozen
class KitaevPhaseEstimation(infra.GateWithRegisters):
    r"""Kitaev's Phase Estimation algorithm

    Originally introduced by Kitaev in https://arxiv.org/abs/quant-ph/9511026.

    Args:
        m_bits: Bitsize of the phase register to be used during phase estimation
        unitary: A cirq Gate representing the unitary to run the phase estimation protocol on.
    """

    m_bits: int
    unitary: cirq.Gate

    @classmethod
    def from_precision_and_eps(cls, unitary: cirq.Gate, precision: int, eps: float):
        """Obtain accurate estimate of $\phi$ to $precision$ bits with $1-eps$ success probability.

        Uses Eq 5.35 from Neilson and Chuang to estimate the size of phase register s.t. we can
        estimate the phase $\phi$ to $precision$ bits of accuracy with probability at least
        $1 - eps$.

        $$
            t = n + ceil(\log(2 + \frac{1}{2\eps}))
        $$

        Args:
            unitary: Unitary operation to obtain phase estimate of.
            precision: Number of bits of precision
            eps: Probability of success.
        """
        m_bits = precision + math.ceil(math.log(2 + 1 / (2 * eps)))
        return KitaevPhaseEstimation(m_bits=m_bits, unitary=unitary)

    @cached_property
    def target_registers(self) -> infra.Registers:
        if isinstance(self.unitary, infra.GateWithRegisters):
            return self.unitary.registers
        else:
            return infra.Registers.build(target=cirq.num_qubits(self.unitary))

    @cached_property
    def phase_registers(self) -> infra.Registers:
        return infra.Registers.build(phase_reg=self.m_bits)

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([*self.phase_registers, *self.target_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        if isinstance(self.unitary, infra.GateWithRegisters):
            target_quregs = {name: quregs[name] for name in self.target_registers}
            unitary_op = self.unitary.on_registers(**target_quregs)
        else:
            unitary_op = self.unitary(*quregs['target'])

        phase_qubits = quregs['phase_reg']

        yield cirq.H.on_each(*phase_qubits)
        for i, qbit in enumerate(phase_qubits[::-1]):
            yield cirq.pow(unitary_op.controlled_by(qbit), 2**i)
        yield cirq.qft(*phase_qubits, inverse=True)
        # Reverse qubits after inverse.
        for i in range(len(phase_qubits) // 2):
            yield cirq.SWAP(phase_qubits[i], phase_qubits[-i - 1])
