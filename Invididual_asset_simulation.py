import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
import util
from DataClass import DataClass
from qiskit.circuit.library import Initialize, Isometry
from scipy.stats import multivariate_normal


def generate_quantum_normal_distribution_all_assets(expected_log_returns, variances, num_qubits, stddevs):
    # Create a list to hold the quantum circuits for each asset
    quantum_circuits = []
    
    # Iterate over each asset
    i=0
    for i in range(len(expected_log_returns)):
        expected_log_return = expected_log_returns[i]
        print(expected_log_return)
        variance = variances[i]
        stddev = stddevs[i]
        
        # Calculate the bounds for the normal distribution
        lower_bound = expected_log_return - 3 * stddev
        upper_bound = expected_log_return + 3 * stddev
        bounds = [(lower_bound, upper_bound)]
        
        
        # Create the normal distribution circuit for the given parameters
        #mvnd = NormalDistribution(num_qubits=3, mu=[expected_log_return], sigma=[[variance]], bounds=bounds)
        inner = QuantumCircuit(3)
        x = np.linspace(lower_bound, upper_bound, num=2**3)  # type: Any
        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, expected_log_return, variance)
        normalized_probabilities = probabilities / np.sum(probabilities)

        initialize = Initialize(np.sqrt(normalized_probabilities))
        circuit = initialize.gates_to_uncompute().inverse()
        inner.compose(circuit, inplace=True)
        qc = QuantumCircuit(3)

        qc.append(inner.to_gate(), inner.qubits)
        # Initialize a quantum circuit
                
        # Measure all qubits
        qc.measure_all()
        
        # Add the quantum circuit to the list
        quantum_circuits.append(qc)

    
    return quantum_circuits

data = DataClass(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="../data/historic_data.xlsx"
)
data.run()
data.print_stats()

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12
num_qubits = [3,3,3]

qc_array = generate_quantum_normal_distribution_all_assets(monthly_expected_log_returns, np.diag(data._cov_matrix), num_qubits, data._stddev)

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_KEY")
backend = service.backend("ibm_rensselaer")
pm = generate_preset_pass_manager(backend=backend,optimization_level=1) #transpilation readable for quantum computer

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

all_asset_samples = []
i = 0
for qc in qc_array:
    isa_circuit = pm.run(qc)
    isa_circuit.depth()

    sampler = SamplerV2(backend=backend)
    job = sampler.run([isa_circuit], shots=2000)
    print(job.job_id())
    counts = job.result()[0].data.meas.get_counts()

    total_counts = sum(counts.values())
    quasi_probabilities = {key: value / total_counts for key, value in counts.items()}
    nearest_pd = nearest_probability_distribution(quasi_probabilities)

    binary_samples = [k for k, v in nearest_pd.items() for _ in range(int(v * 2000))]
    asset_samples = np.array([util.binary_to_asset_values_qc(sample, 3, [monthly_expected_log_returns[i]], data._cov_matrix) for sample in binary_samples])
    all_asset_samples.append(asset_samples)
    i += 1

all_asset_samples = np.array(all_asset_samples)
util.create_new_xlsx_monthly_dates(all_asset_samples,filename="../data/output_qc.xlsx")

for i, asset_samples in enumerate(all_asset_samples):
    # Reshape or flatten the asset samples as needed
    flattened_samples = asset_samples.reshape(-1)  # Adjust this based on the actual shape you need
    plt.figure()
    sns.histplot(flattened_samples, bins=16, kde=False, color='blue')
    plt.xlabel(f'Asset {i+1} Returns')
    plt.ylabel('Frequency')
    plt.title(f'Asset {i+1} Returns Distribution')
    plt.savefig(f"asset_{i+1}_returns_dist.png")
    plt.close()


#creating data object for the generated data
generated_Data = DataClass( 
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2044, 11, 30),
    file_path="output_generation_aer.xlsx")
generated_Data.run()
print("[DATA STATS]")
generated_Data.print_stats()    