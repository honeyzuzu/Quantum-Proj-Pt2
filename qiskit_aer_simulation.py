import datetime as dt
import numpy as np
import qiskit as qs
import qiskit_aer as aer
import qiskit_ibm_runtime as ibm
import matplotlib.pyplot as plt
import DataClass as dc

from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver, ElectronicStructureDriver

use_simulator = True #Set to True to use the simulator, False to use the hardware

#Setting up the backend
service = ibm.QiskitRuntimeService()
hardware = service.backend('ibm_rensselaer')

pm = generate_preset_pass_manager(backend=hardware, optimization_level=3)

simulator = aer.AerSimulator.from_backend(hardware)

if use_simulator:
    backend = simulator

else:
    backend = hardware

#Setting up the data
start = dt.datetime(2016, 1, 1)
end = dt.datetime(2021, 1, 1)
file_path = "historic_data.xlsx"

data = dc.DataClass(start, end, file_path)
data.load_data()
data.run()

#Setting up the quantum algorithm
initial_state = HartreeFock(num_spin_orbitals=data.num_spin_orbitals, num_particles=data.num_particles, qubit_mapping='parity')
ansatz = UCCSD(num_spin_orbitals=data.num_spin_orbitals, num_particles=data.num_particles, initial_state=initial_state, qubit_mapping='parity')
backend = aer.AerSimulator()
optimizer = COBYLA(maxiter=1000)
qalgo = VQE(ansatz, optimizer, quantum_instance=backend)

#Running the quantum algorithm
result = qalgo.compute_minimum_eigenvalue(data.qubit_op)
print(result)

#Plotting the results
plt.plot(result.history['optimal_value'])

