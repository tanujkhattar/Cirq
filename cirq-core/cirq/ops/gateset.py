from collections.abc import Iterable

import cirq
from cirq.ops import raw_types, op_tree, controlled_gate
import cirq.protocols as protocols
import itertools
from cirq import value

@value.value_equality()
class ConstrainedGate:
    def __init__(self, gate_type, predicate=None):
        if isinstance(gate_type, raw_types.Gate):
            if protocols.is_parameterized(gate_type):
                raise ValueError(f"ConstrainedGates do not support parameterized gate {gate_type}")
            default_predicate = ConstrainedGate._default_instance_predicate
        elif isinstance(gate_type, type) and issubclass(gate_type, raw_types.Gate):
            default_predicate = ConstrainedGate._default_class_predicate
        else:
            raise ValueError(
                f"ConstrainedGates support gates or gate instances. Unrecognized value {gate_type}."
            )
        self._predicate = default_predicate if not predicate else predicate
        self._gate_type = gate_type

    def _default_predicate(self, gate_type):
        if isinstance(gate_type, raw_types.Gate):
            if protocols.is_parameterized(gate_type):
                raise ValueError(f"ConstrainedGates do not support parameterized gate {gate_type}")
            default_predicate = ConstrainedGate._default_instance_predicate
        elif isinstance(gate_type, type) and issubclass(gate_type, raw_types.Gate):
            default_predicate = ConstrainedGate._default_class_predicate
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
        if isinstance(op, cirq.ControlledGate):
            return _default_class_predicate(self, op.sub_gate)
        return False

    @staticmethod
    def _default_instance_predicate(self, op):
        if isinstance(op, raw_types.Gate):
            return op == self._gate_type
        return False

    def _default_controlled_gate_predicate(self, op):
        return False

    def _value_equality_values_(self):
        return self._gate_type, self._predicate

    def __contains__(self, op):
        return self._predicate(self, op)

    def __repr__(self):
        if isinstance(self._gate_type, type):
            return f"ConstrainedGate({self._gate_type.__name__}, {self._predicate.__name__})"
        else:
            return f"ConstrainedGate({repr(self._gate_type)},  {self._predicate.__name__})"


class Gateset:
    def __init__(self, gates, *, name: str = None, remove_duplicates = True):
        if not isinstance(gates, Iterable):
            gates = [gates]
        self._gates = set()
        for gate in gates:
            if not isinstance(gate, ConstrainedGate):
                self._gates.add(ConstrainedGate(gate))
            else:
                self._gates.add(gate)
        self._name = name
        if remove_duplicates:
            self._remove_duplicates()

    @staticmethod
    def from_optree(optree : 'cirq.OP_TREE', keep_instances = False) -> 'Gateset':
        gates = []
        for op in op_tree.flatten_to_ops(optree):
            if isinstance(op.gate, controlled_gate.ControlledGate):
                gates.append(type(op.gate.sub_gate))
            elif keep_instances:
                gates.append(op.gate)
            else:
                gates.append(type(op.gate))
        return Gateset(gates)

    def _remove_duplicates(self):
        to_remove = set()
        for g1, g2 in itertools.product(self._gates, self._gates):
            if g1 != g2 and g1.gate in g2:
                to_remove.add(g1)
        self._gates -= to_remove

    def __contains__(self, check_gate):
        for constrained_gate in self._gates:
            if check_gate in constrained_gate:
                return True
        return False

    def __or__(self, other):
        return Gateset(self._gates | other._gates)

    def __repr__(self):
        return f"Gateset({repr(self._gates)})"
