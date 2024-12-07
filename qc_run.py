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

monthly_expected_log_returns = data._mean_vector
cov_matrix = data._cov_matrix
num_qubits = [3,3,3]
stddev = data._stddev

qc = generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev)
qc.draw(output="mpl")
num_shots = 120
sampler = SamplerV2()
job = sampler.run([qc], shots=num_shots)
result = job.result()

counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()
print(counts)
print(len(counts))
binary_samples = [k for k, v in counts.items() for _ in range(int(v * num_shots))]
asset_values = util.binary_to_asset_values_qc(binary_samples[0], 3, monthly_expected_log_returns, cov_matrix)
print(asset_values)

def plot_data(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.show()

plot_data(data._data['^GSPC'], 'S&P 500 (^GSPC) Prices')
plot_data(data._data['^ACWX'], 'ACWI Ex-US (^ACWX) Prices')
plot_data(data._data['^GLAB.L'], 'GlaxoSmithKline (^GLAB.L) Prices')

def split_convert_dict(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        new_key = key.split()
        new_dict[new_key] = value
    return new_dict

def nearest_probability_distribution(quasi_probabilities):
    # Split the quasi-probabilities into three groups
    quasi_probabilities = split_dict_into_three(quasi_probabilities)
    
    # Initialize the new probabilities
    new_probabilities = {key: 0 for key in quasi_probabilities.keys()}
    
    # Iterate over the new probabilities
    for key in new_probabilities.keys():
        # Calculate the new probability
        new_probabilities[key] = sum([value for k, value in quasi_probabilities.items() if key[0] == k[0]])
    
    return new_probabilities

def split_dict_into_three(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = value
    return new_dict

def get_timestamp():
    """
    Returns the current UTC timestamp in ISO-8601 format.

    Returns
    -------
    str
    """
    return dt.datetime.now(dt.UTC).isoformat()


def transpile_and_apply(ansatz, hamiltonian, backend, optimization_level=3, passmanager=None):
    """
    Transpile the ansatz circuit and apply the Hamiltonian observable to the backend layout.
    """

    if not passmanager:
        passmanager = generate_preset_pass_manager(target=backend.target,
                                                   optimization_level=optimization_level)
    
    isa_ansatz = passmanager.run(ansatz)
    isa_hamiltonian = hamiltonian.apply_layout(isa_ansatz.layout)

    return isa_ansatz, isa_hamiltonian

def create(bond_length=0.735):
    atom1 = f"H 0 0 {-bond_length/2}"
    atom2 = f"H 0 0 {bond_length/2}"

    atom = "; ".join([atom1, atom2])
    
    driver = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    hamiltonian = problem.hamiltonian.second_q_op()

    return problem, hamiltonian

def run():
    # Create the problem and Hamiltonian
    problem, hamiltonian = create()

    # Transpile the ansatz and apply the Hamiltonian
    isa_ansatz, isa_hamiltonian = transpile_and_apply(ansatz, hamiltonian, backend) 

    # Run the simulation
    job = backend.run([isa_ansatz, isa_hamiltonian], shots=shots)
    result = job.result()
    
    return result


# Create the molecular model and Hamiltonian
mapped_hamiltonian, mapper = map_to_qubits(hamiltonian)

# Create the ansatze
uccsd_ansatz, twolocal_ansatz = create_ansatze(mapped_hamiltonian, mapper)

# Create the quantum circuits for each ansatz combination
isa_uccsd, isa_hamiltonian_uccsd = transpile_and_apply(uccsd_ansatz,
                                                       mapped_hamiltonian,
                                                       backend=backend,
                                                       optimization_level=3,
                                                       passmanager=pm)

isa_twolocal, isa_hamiltonian_twolocal = transpile_and_apply(twolocal_ansatz,
                                                             mapped_hamiltonian,
                                                             backend=backend,
                                                             optimization_level=3,
                                                             passmanager=pm)

twolocal_results = run_vqe_twolocal(backend, isa_twolocal, isa_hamiltonian_twolocal)

ham = mapped_hamiltonian.to_matrix()
reference_energy = np.linalg.eigh(ham)[0][0]

uccsd_energy, uccsd_err = uccsd_results['energy'][-1]
twolocal_energy, twolocal_err = twolocal_results['energy'][-1]

print(f'Reference Energy: {reference_energy} eV')
print(f'UCCSD Energy: {uccsd_energy} eV')
print(f'TwoLocal Energy: {twolocal_energy} eV')