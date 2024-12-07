import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from DataClass import DataClass


from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
import util
from StockDataProcessor import StockDataProcessor

from collections import Counter

def generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev) -> QuantumCircuit:
    bounds = [(monthly_expected_log_returns[i] - 3*stddev[i], monthly_expected_log_returns[i] + 3*stddev[i]) for i in range(len(monthly_expected_log_returns))]
    #mvnd = NormalDistribution(num_qubits,[3.5,3.5,3.5], cov_matrix, bounds=[(0,7),(0,7),(0,7)])
    mvnd = NormalDistribution(num_qubits[0],monthly_expected_log_returns[0], cov_matrix[0][0], bounds=bounds[0])
    print(mvnd.values)
    print(mvnd.probabilities)
    #qc = QuantumCircuit(sum(num_qubits))
    #qc.append(mvnd, range(sum(num_qubits)))
    qc = QuantumCircuit(3)
    qc.append(mvnd, range(3))
    qc.measure_all()
    return qc

data = DataClass(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="./data/historic_data.xlsx"
)
data.load_data()
data.run()