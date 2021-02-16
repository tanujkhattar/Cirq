from sys import path
from os.path import dirname as dir

path.append(dir(path[0]))

import examples.bernstein_vazirani
import examples.place_on_bristlecone
import examples.hello_qubit
import examples.bell_inequality
import examples.quantum_fourier_transform
import examples.bcs_mean_field
import examples.grover
import examples.phase_estimator
import examples.quantum_teleportation
import examples.superdense_coding


class ExamplesTest:
    def time_example_runs_bernstein_vazirani_perf(self):
        examples.bernstein_vazirani.main(qubit_count=3)

    def time_example_runs_hello_line_perf(self):
        examples.place_on_bristlecone.main()

    def time_example_runs_hello_qubit_perf(self):
        examples.hello_qubit.main()

    def time_example_runs_bell_inequality_perf(self):
        examples.bell_inequality.main()

    def time_example_runs_quantum_fourier_transform_perf(self):
        examples.quantum_fourier_transform.main()

    def time_example_runs_bcs_mean_field_perf(self):
        examples.bcs_mean_field.main()

    def time_example_runs_grover_perf(self):
        examples.grover.main()

    def time_example_runs_phase_estimator_perf(self):
        examples.phase_estimator.main(qnums=(2,), repetitions=2)

    def time_example_runs_quantum_teleportation(self):
        examples.quantum_teleportation.main()

    def time_example_runs_superdense_coding(self):
        examples.superdense_coding.main()
