import dataclasses
import functools
from collections import defaultdict
import itertools
from heapq import heappush, heappop
import numpy as np

from cirq import value
import cirq

from .gateset import Gateset

####### Dividing Operations into Equivalence Classes ##############
@dataclasses.dataclass()
class OpEquivalenceClass:
    """Equivalence class corresponding to various operations"""

    decomp_mask: 'int'
    repr_op: 'cirq.GateOperation'
    repr_gs: 'Gateset'
    all_ops: 'List[cirq.GateOperation]'

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not all([op.gate in self.repr_gs for op in self.all_ops]):
            raise ValueError(
                f"operations in equivalence class {self.all_ops} should be in {self.repr_gs}"
            )


def can_accept(decomposer, op) -> bool:
    try:
        return decomposer(op)
    except:
        return False


def get_equivalence_classes(ops, decomposers) -> 'List[OpEquivalenceClass]':
    # print("GetEquivalenceClasses Start", ops)
    eq_classes = defaultdict(lambda: {})

    for op in ops:
        if not op.gate:
            # For operations like cirq.GlobalPhaseOperation etc.
            continue
        mask = 0
        for (i, decomposer) in enumerate(decomposers):
            if can_accept(decomposer, op):
                mask |= 1 << i
        sub_class = eq_classes[mask]

        op_inserted = False
        op_gs = Gateset.from_ops([op])
        for repr_op, repr_gs in list(sub_class.keys()):
            # print(sub_class)
            if repr_op.gate in op_gs:
                sub_class[(op, op_gs)] = sub_class.pop((repr_op, repr_gs))
                repr_op, repr_gs = op, op_gs

            if op.gate in repr_gs:
                sub_class[(repr_op, repr_gs)].append(op)
                op_inserted = True
                break
        if not op_inserted:
            sub_class[(op, op_gs)] = [op]
    ret = [
        OpEquivalenceClass(decomp_mask, repr_op, repr_gs, all_ops)
        for decomp_mask, subclass in eq_classes.items()
        for (repr_op, repr_gs), all_ops in subclass.items()
    ]
    return ret


############### Define Nodes and Edges of the Decomposition Graph ##########################
@dataclasses.dataclass()
class Edge:
    from_node: 'Node'
    to_node: 'Node'
    op_to_decomposers: 'List[Dict[op, Decomposer]]'
    cost: float = 0


@value.value_equality()
class Node:
    def __init__(self, op_eq_classes: 'Iterable[OpEquivalenceClass]'):
        self._op_eq_classes = tuple(op_eq_classes)
        self._gateset = functools.reduce(
            lambda a, b: a | b, [eq_class.repr_gs for eq_class in op_eq_classes]
        )

    def _value_equality_values_(self):
        return (self._gateset, len(self._op_eq_classes))

    def __lt__(self, other):
        return len(self._gateset) < len(other._gateset)

    def __str__(self):
        return str(self._op_eq_classes)

    def __repr__(self):
        return repr(self._op_eq_classes)

    def get_equivalence_class(self, test_op):
        for eq_class in self._op_eq_classes:
            if test_op in eq_class.repr_gs:
                return eq_class
        return None


def is_target_node(node: 'Node', target_gateset):
    return all(eq_class.repr_op.gate in target_gateset for eq_class in node._op_eq_classes)


def get_cost(eq_class, decomp_ops):
    return len(eq_class.all_ops) * len(decomp_ops)


def get_edges_for_node(node: 'Node', target_gateset, decomposers, debug_print):
    assert not is_target_node(node, target_gateset)
    op_to_decomps = defaultdict(lambda: [])
    ops_in_tgs = []
    for eq_class in node._op_eq_classes:
        repr_op = eq_class.repr_op
        decomp_mask = eq_class.decomp_mask
        if repr_op.gate in target_gateset:
            ops_in_tgs.append(repr_op)
            continue
        if not decomp_mask:
            return []
        for i in range(len(decomposers)):
            if decomp_mask & (1 << i):
                decomp_ops = decomposers[i](repr_op)
                if not decomp_ops:
                    raise ValueError(f"{decomp_ops} empty for {i}, {repr_op}")
                # Greedy heuristic
                cost = get_cost(eq_class, decomp_ops)
                op_to_decomps[repr_op].append((decomp_ops, decomposers[i], cost))

    node_to_edges = {}
    if debug_print:
        print(node._op_eq_classes)
        for key, value in op_to_decomps.items():
            print(key, ':', len(value))
    for i, decomps in enumerate(itertools.product(*op_to_decomps.values())):
        new_ops = set(ops_in_tgs)
        op_to_decomposer = {}
        edge_cost = 0
        for repr_op, (decomp_ops, decomp_fn, decomp_cost) in zip(op_to_decomps.keys(), decomps):
            op_to_decomposer[repr_op] = decomp_fn
            new_ops |= set(decomp_ops)
            edge_cost += decomp_cost
        new_ops = [*cirq.flatten_to_ops(new_ops)]
        new_node = Node(get_equivalence_classes(new_ops, decomposers))
        if new_node in node_to_edges:
            edge = node_to_edges[new_node]
            old_edge_cost = edge.cost
            if old_edge_cost > edge_cost:
                edge.op_to_decomposers = op_to_decomposer
                edge.cost = edge_cost
        else:
            node_to_edges[new_node] = Edge(node, new_node, op_to_decomposer, edge_cost)
    return node_to_edges.values()


def get_path(visited, node, debug_print):
    ret = []
    if debug_print:
        print("get_path")
    while visited[node]:
        par_edge = visited[node]
        if debug_print:
            print(par_edge)
        ret.append(par_edge)
        node = par_edge.from_node
    if debug_print:
        print("-----------------")
    return ret[::-1]


def cost_heuristic(node, target_gateset):
    ret = 0
    for eq_class in node._op_eq_classes:
        if not eq_class.repr_op in target_gateset:
            ret += len(eq_class.all_ops)
    return ret


def find_all_paths(ops, target_gateset, decomposers, debug_print):
    source_node = Node(get_equivalence_classes(ops, decomposers))
    heap = [(cost_heuristic(source_node, target_gateset), 0, source_node)]
    visited = {source_node: None}
    arrived = set()
    opt_cost = defaultdict(lambda: np.inf)
    opt_cost[source_node] = heap[0][0]
    num_steps, num_paths = 0, 0
    while len(heap) > 0:
        cost, dist, node = heappop(heap)
        if node in arrived:
            continue
        arrived.add(node)
        num_steps += 1
        if debug_print:
            print("Start", num_steps, len(heap), cost, dist, node)
        if is_target_node(node, target_gateset):
            num_paths += 1
            print(f"Found {num_paths} path in {num_steps} steps to {node._gateset}")
            return get_path(visited, node, debug_print)

        for edge in get_edges_for_node(node, target_gateset, decomposers, debug_print):
            if edge.to_node not in arrived:
                new_dist = dist + edge.cost
                new_cost = new_dist + cost_heuristic(edge.to_node, target_gateset)
                if new_cost < opt_cost[edge.to_node]:
                    opt_cost[edge.to_node] = new_cost
                    visited[edge.to_node] = edge
                    heappush(heap, (new_cost, new_dist, edge.to_node))
        if debug_print:
            print("End", num_steps, len(heap))
    print(f"Couldn't find a path in {num_steps} to {node}")
    return None


class TransformCircuit(cirq.PointOptimizer):
    def __init__(self, decomp_node, op_to_decomposer):
        super().__init__()
        self._decomp_node = decomp_node
        self._op_to_decomposer = op_to_decomposer

    def optimization_at(self, circuit, index, op):
        if cirq.trace_distance_bound(op) <= 1e-8:
            return cirq.PointOptimizationSummary(
                clear_span=1, new_operations=[], clear_qubits=op.qubits
            )
        repr_op = None
        for eq_class in self._decomp_node._op_eq_classes:
            if op.gate in eq_class.repr_gs:
                repr_op = eq_class.repr_op
        assert repr_op is not None
        if not repr_op in self._op_to_decomposer:
            return None

        converted = self._op_to_decomposer[repr_op](op)
        converted = [op for op in converted if not isinstance(op, cirq.GlobalPhaseOperation)]
        return cirq.PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )


def compile(
    circuit: 'cirq.Circuit', target_gateset, decomposers, debug_print=False
):  # , cost_function):
    circuit_ops = [*circuit.all_operations()]
    path = find_all_paths(circuit_ops, target_gateset, decomposers, debug_print)
    for edge in path:
        if debug_print:
            print(edge)
        TransformCircuit(edge.from_node, edge.op_to_decomposers).optimize_circuit(circuit)
    return circuit
