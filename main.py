from QUANTUM_PACKER import QUANTUM_PACKER
from PACKING_PROBLEM_DATA_LOADER import PACKING_PROBLEM_DATA_LOADER

# problem instance name
#problem_instance = "MINI_TEST"
# Board dimensions
#W = 300
#H = 200
#board_dimensions = (W, H)
# Pieces
#P1 = [(0, 0), (80, 0), (80, 40)]
#P2 = [(0, 0), (100, 0), (100, 100), (0, 100)]
#P3 = [(0, 0), (50, 0), (50, 50), (100, 50), (50, 100)]
#pieces = [P1, P2, P2, P3, P3, P3]

# problem instance name
#problem_instance = "BASIC_TEST_1"
# Board dimensions
#W = 100
#H = 600
#board_dimensions = (W, H)
# Pieces
#P1 = [(0, 0), (80, 0), (80, 40), (30, 40), (90, 170), (140, 170), (140, 200), (0, 200)]
#P2 = [(0, 0), (200, 0), (200, 100), (120, 100), (120, 40), (40, 40), (40, 100), (0, 100)]
#P3 = [(0, 50), (100, 50), (100, 0), (150, 0), (150, 50), (250, 50), (250, 200), (150, 200), (150, 150), (100, 150)]
#P4 = [(0, 0), (100, 0), (50, 50), (70, 100), (50, 150), (100, 200), (0, 200)]
#P5 = [(0, 0), (70, 0), (70, 100), (0, 100), (0, 80), (40, 80), (40, 20), (0, 20)]
#P6 = [(0, 0), (300, 0), (300, 200), (200, 200), (200, 170), (170, 170), (170, 30), (0, 30)]
#P7 = [(0, 0), (100, 0), (100, 30), (70, 30), (30, 100), (100, 170), (100, 200), (0, 200)]
#P8 = [(-40, -40), (40, -40), (60, 0), (0, 40), (-60, 0)]
#P9 = [(0, 0), (150, 0), (150, 200), (0, 200), (0, 180), (50, 100), (0, 20)]
#pieces = [P1, P2, P3, P4, P5, P6, P7, P8, P9]

# Choose an instance from SHAPES1, SHAPES2, SHIRTS, TROUSERS, SWIM
problem_instance = "SHAPES1"
packing_problem_data = PACKING_PROBLEM_DATA_LOADER(problem_instance)
pieces = packing_problem_data.pieces
# Board dimensions
W = 900
H = 400
board_dimensions = (W, H)

# specify number of qubits available and quantum machine (local_qasm_simulator, IBM-DE or IBM-US)
num_qubits = 25
#quantum_machine = 'local_qasm_simulator'
quantum_machine = 'IBM-DE'

# create a quantum packer for this packing problem
min_num_pieces_per_rectangle = 2
quantum_packer = QUANTUM_PACKER(problem_instance, board_dimensions, pieces, min_num_pieces_per_rectangle, num_qubits, quantum_machine)

# show all the pieces to pack
quantum_packer.show_pieces()

# solve the irregular packing problem
layout = quantum_packer.solve(num_trials=100)

# show the solution
if layout == None:
    print('WARNING: NO SOLUTION FOUND FOR IRREGULAR PACKING PROBLEM')
else:
    quantum_packer.show_layout(layout)
