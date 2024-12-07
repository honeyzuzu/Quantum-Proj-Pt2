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
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 0.735", unit=DistanceUnit.ANGSTROM, basis='sto3g')
es_driver = ElectronicStructureDriver(driver)
qubit_op = es_driver.run()

qubit_op = FermionicOp(h1=qubit_op.one_body_integrals, h2=qubit_op.two_body_integrals).mapping(JordanWignerMapper())


#Running the quantum algorithm
result = qalgo.compute_minimum_eigenvalue(data.qubit_op)
print(result)

#Plotting the results
plt.plot(result.history['optimal_value'])

