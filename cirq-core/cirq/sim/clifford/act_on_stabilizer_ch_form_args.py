# Copyright 2020 The Cirq Developers
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

from typing import Any, Dict, TYPE_CHECKING, List, Sequence, Iterable

import numpy as np

from cirq._compat import deprecated_parameter
from cirq.ops import common_gates, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.act_on_args import ActOnArgs
from cirq.sim.clifford.stabilizer_state_ch_form import StabilizerStateChForm

if TYPE_CHECKING:
    import cirq
    from typing import Optional


def _rewrite_deprecated_args(args, kwargs):
    if len(args) > 2:
        kwargs['axes'] = args[2]
    if len(args) > 3:
        kwargs['prng'] = args[3]
    if len(args) > 4:
        kwargs['log_of_measurement_results'] = args[4]
    if len(args) > 5:
        kwargs['qubits'] = args[5]
    return args[:2], kwargs


class ActOnStabilizerCHFormArgs(ActOnArgs):
    """Wrapper around a stabilizer state in CH form for the act_on protocol.

    To act on this object, directly edit the `state` property, which is
    storing the stabilizer state of the quantum system with one axis per qubit.
    """

    @deprecated_parameter(
        deadline='v0.13',
        fix='No longer needed. `protocols.act_on` infers axes.',
        parameter_desc='axes',
        match=lambda args, kwargs: 'axes' in kwargs
        or ('prng' in kwargs and len(args) == 3)
        or (len(args) > 3 and isinstance(args[3], np.random.RandomState)),
        rewrite=_rewrite_deprecated_args,
    )
    def __init__(
        self,
        state: StabilizerStateChForm,
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
        qubits: Sequence['cirq.Qid'] = None,
        axes: Iterable[int] = None,
    ):
        """Initializes with the given state and the axes for the operation.
        Args:
            state: The StabilizerStateChForm to act on. Operations are expected
                to perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStabilizerCHFormArgs.record_measurement_result`.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
        """
        super().__init__(prng, qubits, axes, log_of_measurement_results)
        self.state = state

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool):
        strats = []
        if allow_decompose:
            strats.append(_strat_act_on_stabilizer_ch_form_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, self, qubits)
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the stabilizer state form."""
        return [self.state._measure(self.qubit_map[q], self.prng) for q in qubits]

    def copy(self) -> 'cirq.ActOnStabilizerCHFormArgs':
        return ActOnStabilizerCHFormArgs(
            state=self.state.copy(),
            qubits=self.qubits,
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results.copy(),
        )


def _strat_act_on_stabilizer_ch_form_from_single_qubit_decompose(
    val: Any, args: 'cirq.ActOnStabilizerCHFormArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    if num_qubits(val) == 1:
        if not has_unitary(val):
            return NotImplemented
        u = unitary(val)
        clifford_gate = SingleQubitCliffordGate.from_unitary(u)
        if clifford_gate is not None:
            # Gather the effective unitary applied so as to correct for the
            # global phase later.
            final_unitary = np.eye(2)
            for axis, quarter_turns in clifford_gate.decompose_rotation():
                gate = None  # type: Optional[cirq.Gate]
                if axis == pauli_gates.X:
                    gate = common_gates.XPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)
                elif axis == pauli_gates.Y:
                    gate = common_gates.YPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)
                else:
                    assert axis == pauli_gates.Z
                    gate = common_gates.ZPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)

                final_unitary = np.matmul(unitary(gate), final_unitary)

            # Find the entry with the largest magnitude in the input unitary.
            k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
            # Correct the global phase that wasn't conserved in the above
            # decomposition.
            args.state.omega *= u[k] / final_unitary[k]
            return True

    return NotImplemented
