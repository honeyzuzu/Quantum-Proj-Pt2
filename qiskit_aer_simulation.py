import datetime as dt
import numpy as np
import qiskit as qs
import qiskit_aer as aer
import qiskit_ibm_runtime as ibm
import matplotlib.pyplot as plt

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