from qiskit import IBMQ
from qiskit_methods.config import API_TOKEN_DE, API_TOKEN_US, API_URL_DE
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Z, I
import numpy as np
from qiskit_methods.backends import find_least_busy_backend


# Load IBMQ account provider and select a backend
# GERMANY-EHNINGEN
#IBMQ.enable_account(API_TOKEN_DE, API_URL_DE)
#provider = IBMQ.get_provider(hub='fraunhofer-de', group='fhg-all', project='ticket')
#backend = provider.get_backend('ibmq_ehningen')
# USA
IBMQ.enable_account(API_TOKEN_US)
provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', project='ticket')
least_busy_backend_name = find_least_busy_backend(provider, n_qubits=4)
backend = provider.get_backend(least_busy_backend_name)

# TEST OF THE VQE RUNTIME PROGRAM

num_qubits = 4
hamiltonian = (Z ^ Z) ^ (I ^ (num_qubits - 2))

# the rotation gates are chosen randomly, so we set a seed for reproducibility
ansatz = EfficientSU2(num_qubits, reps=1, entanglement='linear', insert_barriers=True)
ansatz.draw('mpl', style='iqx')

initial_point = np.random.random(ansatz.num_parameters)

backend_options = {
    'backend_name': backend.name()
}

intermediate_info = {
    'nfev': [],
    'parameters': [],
    'energy': [],
    'stddev': []
}

def raw_callback(*args):
    job_id, (nfev, parameters, energy, stddev) = args
    intermediate_info['nfev'].append(nfev)
    intermediate_info['parameters'].append(parameters)
    intermediate_info['energy'].append(energy)
    intermediate_info['stddev'].append(stddev)

vqe_inputs = {
    'ansatz': ansatz,
    'operator': hamiltonian,
    'optimizer': {'name': 'SPSA', 'maxiter': 5},  # let's only do a few iterations!
    'initial_point': initial_point,
    'measurement_error_mitigation': True,
    'shots': 1024
}

job = provider.runtime.run(
    program_id='vqe',
    inputs=vqe_inputs,
    options=backend_options,
    callback=raw_callback
)

print('Job ID:', job.job_id())
result = job.result()
print(f'Reached {result["optimal_value"]} after {result["optimizer_evals"]} evaluations.')

