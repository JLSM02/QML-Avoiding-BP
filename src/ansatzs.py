from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import NLocal, TwoLocal, CCXGate, CRZGate, RXGate
import numpy as np



# ====================================================================
#                 Function to build a deep ansatz
# ====================================================================
def build_deep_ansatz(num_qubits: int, layers_per_qubit: int = 10) -> tuple[QuantumCircuit, int]:
    """
    Builds a deep ansatz.
    -----------------------------------------
    Args:
        layers_per_qubit (int): Number of layers to add to the circuit per qubit.
        num_qubits (int): Number of qubits to be used in the circuit.
    -----------------------------------------
    Returns:
        tuple: (QuantumCircuit, number of parameters in the circuit)
    """
    L = layers_per_qubit * num_qubits  # Number of layers
    qc = QuantumCircuit(num_qubits)
    qc.ry(np.pi/4, range(num_qubits))
    qc.barrier()
    thetas = []

    def layer(qc, theta_list):
        # RX for each qubit
        for i in range(num_qubits):
            aux = np.random.random()
            if aux < 1/3:
                qc.rx(theta_list[i], i)
            elif aux < 2/3:
                qc.ry(theta_list[i], i)
            else:
                qc.rz(theta_list[i], i)
        # CZ between near qubits
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    for layer_index in range(L):
        theta_layer = [Parameter(f'θ_{layer_index}_{i}') for i in range(num_qubits)]
        thetas.append(theta_layer)
        layer(qc, theta_layer)
        qc.barrier()

    # Returns circuit and number of parameters
    num_params =  len(thetas)*num_qubits
    return qc, num_params




# ====================================================================
#                 Function to N local ansatz
# ====================================================================
def build_Nlocal_ansatz(num_qubits, layers = 2) -> tuple[QuantumCircuit, int]:
    """
    Creates an N-local ansatz with a given number of qubits and repetitions.
    Returns the circuit and the number of free parameters.
    -----------------------------------------
    Args:
        num_qubits (int): Number of qubits in the circuit.
        layers (int): Number of repetitions (layers) of the ansatz.
    -----------------------------------------
    Returns:
        tuple: (QuantumCircuit, number of parameters in the circuit)
    """
    theta = Parameter("θ")

    entanglement_list = []
    for i in range(num_qubits-2):
        entanglement_list.append([i, i+1, i+2])

    ansatz = NLocal(
        num_qubits = num_qubits,
        rotation_blocks=[RXGate(theta), CRZGate(theta)],  # Keep rotation blocks
        entanglement_blocks=CCXGate(),
        entanglement=entanglement_list,  # Define entanglement pattern
        reps=layers,
        insert_barriers=True,
    )
    
    return ansatz, ansatz.num_parameters






def build_twoLocal_ansatz(num_qubits: int, layers: int = 1) -> tuple[QuantumCircuit, int]:
    """
    Creates an Two-Local ansatz with a given number of qubits and repetitions.
    Returns the circuit and the number of free parameters.
    -----------------------------------------
    Args:
        num_qubits (int): Number of qubits in the circuit.
        layers (int): Number of repetitions (layers) of the ansatz.
    -----------------------------------------
    Returns:
        tuple: (QuantumCircuit, number of parameters in the circuit)
    """
    ansatz = TwoLocal(num_qubits,
                      rotation_blocks='ry',
                      entanglement_blocks='cz',
                      entanglement='linear',
                      reps=layers,
                      insert_barriers=True)
    
    return ansatz, ansatz.num_parameters





def build_Surf_ansatz(num_qubits: int, layers: int = 1) -> tuple[QuantumCircuit, int]:

    
    qc = QuantumCircuit(num_qubits)

    # Parameters list
    thetas = [[Parameter(f'θ_{0}'), Parameter(f'θ_{1}')]]

    # Add random gates
    def rand_gate(theta, qubit):
        r = np.random.random()

        if r < 1/3:
            qc.rx(theta, qubit)
        elif r < 2/3:
            qc.ry(theta, qubit)
        else:
            qc.rz(theta, qubit)

    for l in range(layers):
        
        for i in range(num_qubits - 1):

            rand_gate(thetas[0][0], i)
            rand_gate(thetas[0][1], i+1)
            qc.cz(i, i + 1)

        qc.barrier()
        
    return qc, 2

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (HighLevelSynthesis, InverseCancellation)
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (SwapStrategy, FindCommutingPauliEvolutions, Commuting2qGateRouter)
from qiskit.circuit.library import CXGate

def is_two_qubits(pauli_string):
    """Function that returns an operator if it acts on 2 qubits"""
    return pauli_string.count('I') == len(pauli_string) - 1

def optimize_qaoa(cost_operator):
    # Separate 2 qubit gates from other gates
    pauli_list = cost_operator.to_list()
    two_qubits = [(p, c) for p, c in pauli_list if not is_two_qubits(p)]
    others    = [(p, c) for p, c in pauli_list if is_two_qubits(p)]
    cost_2qubits = SparsePauliOp.from_list(two_qubits)
    cost_other = SparsePauliOp.from_list(others)

    # Choose swap strategy (in this case -> line)
    num_qubits=cost_operator.num_qubits
    swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
    edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}

    # Define pass manager
    init_cost_layer = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap_strategy, edge_coloring,), HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]), InverseCancellation(gates_to_cancel=[CXGate()])])

    # Create a circuit for the 2 qubit gates and optimize it with the cost layer pass manager
    qaoa_2qubits = QAOAAnsatz(cost_operator=cost_2qubits, reps=1, initial_state=QuantumCircuit(num_qubits), mixer_operator=QuantumCircuit(num_qubits))
    qaoa_2qubits_opt=init_cost_layer.run(qaoa_2qubits)

    return qaoa_2qubits_opt
