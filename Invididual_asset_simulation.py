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

data = StockDataProcessor(
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