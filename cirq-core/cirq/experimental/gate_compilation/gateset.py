from collections.abc import Iterable

import cirq
from cirq.ops import raw_types, op_tree, controlled_gate
import cirq.protocols as protocols
import itertools
from cirq import value


@value.value_equality()
class ConstrainedGate:
    def __init__(self, gate_type, predicate=None):
        default_predicate = ConstrainedGate._default_predicate(gate_type)
        self._predicate = default_predicate if not predicate else predicate
        self._gate_type = gate_type

    @staticmethod
    def _default_predicate(gate_type):
        if isinstance(gate_type, raw_types.Gate):
            if protocols.is_parameterized(gate_type):
                raise ValueError(f"ConstrainedGates do not support parameterized gate {gate_type}")
            if isinstance(gate_type, controlled_gate.ControlledGate):
                return ConstrainedGate._default_controlled_gate_predicate
            else:
                return ConstrainedGate._default_instance_predicate
        elif isinstance(gate_type, type) and issubclass(gate_type, raw_types.Gate):
            return ConstrainedGate._default_class_predicate
        else:
            raise ValueError(
                f"ConstrainedGates support gates or gate instances. Unrecognized value {gate_type}."
            )

    @property
    def gate(self):
        return self._gate_type

    @staticmethod
    def _default_class_predicate(self, op):
        if isinstance(op, type):
            # op is a class type.
            return issubclass(op, self._gate_type)
        if isinstance(op, self._gate_type):
            # op is an instance of gate_type
            return True
        return False

    def _value_equality_values_(self):
        if isinstance(self._gate_type, type):
            return (self._gate_type.__name__,)
        else:
            return (self._gate_type,)

    @staticmethod
    def _default_instance_predicate(self, op):
        if isinstance(op, raw_types.Gate):
            return op == self._gate_type
        return False

    @staticmethod
    def _default_controlled_gate_predicate(self, op):
        # TODO(tanujkhattar): Decide how to handle controlled gates by default.
        if isinstance(op, controlled_gate.ControlledGate):
            return (
                op.sub_gate == self._gate_type.sub_gate
                and self._gate_type.num_controls() >= op.num_controls()
            )
        return False

    def __contains__(self, op):
        return self._predicate(self, op)

    def __repr__(self):
        if isinstance(self._gate_type, type):
            return f"ConstrainedGate({self._gate_type.__name__}, {self._predicate.__name__})"
        else:
            return f"ConstrainedGate({repr(self._gate_type)},  {self._predicate.__name__})"

    def __str__(self):
        if isinstance(self._gate_type, type):
            return f"CG({self._gate_type.__name__})"
        else:
            return f"CG({str(self._gate_type)})"


@value.value_equality()
class Gateset:
    def __init__(self, gates, *, name: str = None, remove_duplicates=True):
        if not isinstance(gates, Iterable):
            gates = [gates]
        gates_mutable = set()
        for gate in gates:
            if not isinstance(gate, ConstrainedGate):
                gates_mutable.add(ConstrainedGate(gate))
            else:
                gates_mutable.add(gate)
        self._name = name
        if remove_duplicates:
            gates_mutable = self._remove_duplicates(gates_mutable)
        self._gates = frozenset(gates_mutable)

    @staticmethod
    def from_ops(optree: 'cirq.OP_TREE', keep_instances=False) -> 'Gateset':
        gates = []
        for op in op_tree.flatten_to_ops(optree):
            if isinstance(op.gate, controlled_gate.ControlledGate) or keep_instances:
                gates.append(op.gate)
            else:
                gates.append(type(op.gate))
        return Gateset(gates)

    def _remove_duplicates(self, gates):
        to_remove = set()
        for g1, g2 in itertools.product(gates, gates):
            if g1 != g2 and g1.gate in g2:
                to_remove.add(g1)
        gates -= to_remove
        return gates

    def __contains__(self, check_gate):
        for constrained_gate in self._gates:
            if check_gate in constrained_gate:
                return True
        return False

    def __or__(self, other):
        return Gateset(self._gates | other._gates)

    def __sub__(self, other):
        return Gateset(self._gates - other._gates)

    def __xor__(self, other):
        return Gateset(self._gates ^ other._gates)

    def __and__(self, other):
        return Gateset(self._gates & other._gates)

    def __len__(self):
        return len(self._gates)

    def __str__(self):
        if self._name:
            return self._name
        return f"Gateset({', '.join(str(g) for g in self._gates)})"

    def __repr__(self):
        return f"Gateset({repr(self._gates)})"

    def _value_equality_values_(self):
        return (self._gates,)
