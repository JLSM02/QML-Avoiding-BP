from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate
import numpy as np



# ====================================================================
#                 Function to build a deep ansatz
# ====================================================================
def build_deep_ansatz(num_qubits: int, layers_per_qubit: int = 10):
    """
    Builds a deep ansatz.

    Args:
        layers_per_qubit (int): Number of layers to add to the circuit per qubit.
        num_qubits (int): Number of qubits to be used in the circuit.

    Returns:
        qc (QuantumCircuit): Resulting Qiskit circuit implementing the ansantz.
        num_params (int): Number of parameters used in the ansatz
    """
    L = layers_per_qubit * num_qubits  # número de capas
    qc = QuantumCircuit(num_qubits)
    qc.ry(np.pi/4, range(num_qubits))
    qc.barrier()
    thetas = []

    def layer(qc, theta_list):
        # RX en cada qubit
        for i in range(num_qubits):
            aux = np.random.random()
            if aux < 1/3:
                qc.rx(theta_list[i], i)
            elif aux < 2/3:
                qc.ry(theta_list[i], i)
            else:
                qc.rz(theta_list[i], i)
        # CZ entre qubits adyacentes
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    for layer_index in range(L):
        theta_layer = [Parameter(f'θ_{layer_index}_{i}') for i in range(num_qubits)]
        thetas.append(theta_layer)
        layer(qc, theta_layer)
        qc.barrier()

    # Devuelve el circuito y el numero de parametros
    num_params =  len(thetas)*num_qubits
    return qc, num_params




# ====================================================================
#                 Function to N local ansatz
# ====================================================================
def build_Nlocal_ansatz(num_qubits, num_layers = 2):
    theta = Parameter("θ")

    entanglement_list = []
    for i in range(num_qubits-2):
        entanglement_list.append([i, i+1, i+2])

    ansatz = NLocal(
        num_qubits = num_qubits,
        rotation_blocks=[RXGate(theta), CRZGate(theta)],  # Keep rotation blocks
        entanglement_blocks=CCXGate(),
        entanglement=entanglement_list,  # Define entanglement pattern
        reps=num_layers,
        insert_barriers=True,
    )
    
    return ansatz, ansatz.num_parameters



