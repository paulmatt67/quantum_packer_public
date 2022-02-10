from qiskit import IBMQ
from qiskit_methods.config import API_TOKEN_DE, API_TOKEN_US, API_URL_DE
from qiskit_methods.backends import show_num_qubits_for_provider
from qiskit_methods.backends import find_least_busy_backend


# Load IBMQ account provider and select a backend

# GERMANY-EHNINGEN
#IBMQ.enable_account(API_TOKEN_DE, API_URL_DE)
#provider = IBMQ.get_provider(hub='fraunhofer-de', group='fhg-all', project='ticket')
# Show backends available and number of qubits they have
#print('Available backends:')
#show_num_qubits_for_provider(provider)
#backend = provider.get_backend('ibmq_qasm_simulator')
#backend = provider.get_backend('ibmq_ehningen')

# USA
IBMQ.enable_account(API_TOKEN_US)
provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', project='ticket')
# Show backends available and number of qubits they have
print('Available backends:')
show_num_qubits_for_provider(provider)
#selected_backend_name = find_least_busy_backend(provider, n_qubits=36)
selected_backend_name = provider.get_backend('ibmq_brooklyn')
print('Selected backend: ' + selected_backend_name)
backend = provider.get_backend(selected_backend_name)

# Show programs available
provider.runtime.pprint_programs()

# ID of the uploaded Qiskit runtime program to use
# GERMANY-EHNINGEN
#program_id = 'tsp-runtime-solver-rXjPDWKAb2'
# USA
program_id = 'tsp-runtime-solver-n20RRyBn1X'

# Program options
options = {'backend_name': backend.name()}

# Create a dictionary of TSP instances (distance matrices)
problem_instances = {}
D1 = [[0]]
D2 = [[0, 1], [1, 0]]
D3 = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
D4 = [[0, 1, 2, 1], [1, 0, 3, 1], [2, 3, 0, 0], [1, 1, 0, 0]]
D5 = [[0, 1, 2, 1, 3], [1, 0, 3, 1, 1], [2, 3, 0, 0, 0], [1, 1, 0, 0, 2], [3, 1, 0, 2, 0]]
problem_instances.update({'TSP1': D1})
problem_instances.update({'TSP2': D2})
problem_instances.update({'TSP2': D3})
problem_instances.update({'TSP2': D4})
problem_instances.update({'TSP2': D5})

# Program inputs
inputs = {}
inputs.update({'problem_instances': problem_instances})

# Simple callback function that just prints interim results
def interim_result_callback(job_id, interim_result):
    print(f"interim result: {interim_result}")

# Run the program
job = provider.runtime.run(program_id, options=options, inputs=inputs, callback=interim_result_callback)

# Retrieve the result of the job
result = job.result()
print(result)
