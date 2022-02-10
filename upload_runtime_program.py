from qiskit import IBMQ
from qiskit_methods.config import API_TOKEN_DE, API_TOKEN_US, API_URL_DE


# Metadata of the runtime program
meta = {
  "name": "TSP-RUNTIME-SOLVER",
  "description": "Solves a batch of Traveling Salesman Problems and ensures that returned solutions are Hamiltonian paths.",
  "max_execution_time": 100000,
  "spec": {}
}

meta["spec"]["parameters"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "properties": {
    "problem_instances": {
      "description": "",
      "type": "a dictionary of problem instances, where keys are the TSP instance name as string and values are "
              "distance matrices as lists of lists."
    },
  },
  "required": [
    "problem_instances"
  ]
}

meta["spec"]["return_values"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "description": "dictionary of valid solutions for each instance",
  "type": "dictionary"
}

meta["spec"]["interim_results"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "description": "dictionary of solutions (Hamiltonian path as list of vertices) for each instance",
  "type": "dictionary"
}

# Load IBMQ account provider
#IBMQ.enable_account(API_TOKEN_DE, API_URL_DE)
#provider = IBMQ.get_provider(hub='fraunhofer-de')
# USA
IBMQ.enable_account(API_TOKEN_US)
provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', project='ticket')


# Upload the program
program_id = provider.runtime.upload_program(data='TSP_RUNTIME_SOLVER.py', metadata=meta)

# Query the program for information
prog = provider.runtime.program(program_id)
print(prog)

# Delete a program if there is a mistake in it
#provider.runtime.delete_program('tsp-runtime-solver-8EgeDZ37qk')
