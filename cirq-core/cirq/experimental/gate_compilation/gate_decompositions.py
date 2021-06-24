import cirq
import numpy as np


def is_gate(op):
    return op and op.gate


def op_to_native_decompose(op):
    if not is_gate(op):
        raise ValueError(f"{op} is not supported")
    return cirq.decompose_once(op)


def single_qubit_op_to_phased_x_z(op):
    if not is_gate(op) or not op.gate.num_qubits() == 1:
        raise ValueError(f"{op} is not supported")
    # Single qubit gate to Paulis.
    gates = cirq.single_qubit_matrix_to_phased_x_z(cirq.unitary(op.gate))
    return [gate.on(*op.qubits) for gate in gates]


def single_qubit_op_to_paulis(op):
    if not is_gate(op) or not op.gate.num_qubits() == 1:
        raise ValueError(f"{op} is not supported")
    # Single qubit gate to Paulis.
    gates = cirq.single_qubit_matrix_to_gates(cirq.unitary(op.gate))
    return [gate.on(*op.qubits) for gate in gates]


def two_qubit_op_to_czs(op):
    if not is_gate(op) or not op.gate.num_qubits() == 2:
        raise ValueError(f"{op} is not supported")
    # Decomposes a two-qubit operation into Z/X/Y/CZ gates
    q0, q1 = op.qubits
    return cirq.two_qubit_matrix_to_operations(
        q0, q1, cirq.unitary(op.gate), allow_partial_czs=False
    )


def two_qubit_op_to_partial_czs(op):
    if not is_gate(op) or not op.gate.num_qubits() == 2:
        raise ValueError(f"{op} is not supported")
    # Decomposes a two-qubit operation into Z/XY/CZ gates
    q0, q1 = op.qubits
    return cirq.two_qubit_matrix_to_operations(
        q0, q1, cirq.unitary(op.gate), allow_partial_czs=True
    )


def two_qubit_op_to_ion_ops(op):
    if not is_gate(op) or not op.gate.num_qubits() == 2:
        raise ValueError(f"{op} is not supported")
    # Decomposes a two-qubit operation into MS / single qubit rotation gates
    q0, q1 = op.qubits
    return cirq.two_qubit_matrix_to_ion_operations(q0, q1, cirq.unitary(op.gate))


def three_qubit_op_to_czs_cnots(op):
    if not is_gate(op) or not op.gate.num_qubits() == 3:
        raise ValueError(f"{op} is not supported")
    q0, q1, q2 = op.qubits
    return cirq.three_qubit_matrix_to_operations(q0, q1, q2, cirq.unitary(op.gate))


def multi_controlled_rotation_to_cx_ccx(op):
    if not is_gate(op) or not (
        isinstance(op, cirq.ControlledOperation) and len(op.sub_operation.qubits) == 1
    ):
        raise ValueError(f"{op} is not supported")
    return cirq.decompose_multi_controlled_rotation(
        cirq.unitary(op.sub_operation.gate), op.controls, *op.sub_operation.qubits
    )


def create_czpow_to_fsim_and_rotations(fsim_gate):
    def czpow_to_fsim_and_rotations(op):
        if not is_gate(op) or not isinstance(op.gate, cirq.CZPowGate):
            raise ValueError(f"{op} is not supported")
        # Decomposes CZPowGate into two FSimGates + single qubit rotations
        return cirq.decompose_cphase_into_two_fsim(op.gate, fsim_gate=fsim_gate, qubits=op.qubits)

    return czpow_to_fsim_and_rotations


def czpow_to_sqrt_iswap(op):
    if not is_gate(op) or not isinstance(op.gate, cirq.CZPowGate):
        raise ValueError(f"{op} is not supported")
    # Decomposes CZPowGate into two sqrt_iswaps + rotations
    return [
        *cirq.google.optimizers.convert_to_sqrt_iswap.cphase_to_sqrt_iswap(
            op.qubits[0], op.qubits[1], op.gate.exponent
        )
    ]


syc_fsim = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)

all_decomposers = [
    op_to_native_decompose,
    single_qubit_op_to_paulis,
    # single_qubit_op_to_phased_x_z,
    two_qubit_op_to_czs,
    two_qubit_op_to_partial_czs,
    two_qubit_op_to_ion_ops,
    three_qubit_op_to_czs_cnots,
    multi_controlled_rotation_to_cx_ccx,
    create_czpow_to_fsim_and_rotations(syc_fsim),
    czpow_to_sqrt_iswap,
]
