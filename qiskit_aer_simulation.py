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

qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)
qc.draw(output="mpl")
num_shots = 120
sampler = Sampler()
job = sampler.run([qc], shots=num_shots)
result = job.result()


counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()
print(counts)
print(len(counts))


binary_samples = [k for k, v in counts.items() for _ in range(int(v * num_shots))]
print(binary_samples)
print(len(binary_samples))
asset_samples = np.array([util.binary_to_asset_values_test(sample, num_qubits, monthly_expected_log_returns, data._cov_matrix) for sample in binary_samples])
util.create_new_xlsx_monthly_dates(asset_samples,filename="output.xlsx")


#Running the quantum algorithm
result = qalgo.compute_minimum_eigenvalue(data.qubit_op)
print(result)


"""
# Load the generated percent data
generated_percent_data = dc(
    start=dt.datetime(2024, 4, 30),
    end=dt.datetime(2044, 11, 30),
    file_path="../data/percentage_output.xlsx"
)
generated_percent_data.run_nonlog()

portfolio_returns = generated_percent_data._data.dot(annual_expected_returns)

annual_portfolio_return = (1 + portfolio_returns).prod() ** (12 / generated_Data._data.shape[0]) - 1
annual_portfolio_volatility = np.std(portfolio_returns) * np.sqrt(12)
risk_free_rate = 0.00
sharpe_ratio = (annual_portfolio_return - risk_free_rate) / annual_portfolio_volatility
# calculate the maximum drawdown
cumulative_returns = (1 + portfolio_returns).cumprod()
max_drawdown = np.min(
    cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
)
# calculate the Calmar ratio
calmar_ratio = annual_portfolio_return / max_drawdown

print("annual_portfolio_return: ",annual_portfolio_return)
print("annual_portfolio_volatility: ",annual_portfolio_volatility)
print("sharpe_ratio: ",sharpe_ratio)
print("max_drawdown: ", max_drawdown)
print("calmar_ratio: ",calmar_ratio)

"""

# Plot the data
def plot_data(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.show()

plot_data(data._data['^GSPC'], 'S&P 500 (^GSPC) Prices')
plot_data(data._data['^ACWX'], 'ACWI Ex-US (^ACWX) Prices')
plot_data(data._data['^GLAB.L'], 'GlaxoSmithKline (^GLAB.L) Prices')

def split_dict_into_three(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = value
    return new_dict

def split_convert_dict(original_dict):

    new_dict = {}
    for key, value in original_dict.items():
        new_key = key.split()
        new_dict[new_key] = value
    return new_dict


    


