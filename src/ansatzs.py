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
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (HighLevelSynthesis, InverseCancellation)
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (SwapStrategy, FindCommutingPauliEvolutions, Commuting2qGateRouter)
from qiskit.circuit.library import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag, dag_to_circuit

class QAOAPass(TransformationPass):

    def __init__(self, num_layers, num_qubits, cost_other, init_state = None, mixer_layer = None):

        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        
        if init_state is None:
            # Add default initial state -> equal superposition
            self.init_state = QuantumCircuit(num_qubits)
            self.init_state.h(range(num_qubits))
        else: 
            self.init_state = init_state
        
        if mixer_layer is None:
            # Define default mixer layer
            self.mixer_layer = QuantumCircuit(num_qubits)
            self.mixer_layer.rx(-2*ParameterVector("β", self.num_layers)[0], range(num_qubits))
        else:
            self.mixer_layer = mixer_layer

        self.cost_other = cost_other

    def run(self, cost_layer_dag):

        cost_layer = dag_to_circuit(cost_layer_dag)
        qaoa_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Re-parametrize the circuit
        gammas = ParameterVector("γ", self.num_layers)
        betas = ParameterVector("β", self.num_layers)

        # Add initial state
        qaoa_circuit.compose(self.init_state, inplace = True)

        # Iterate over number of qaoa layers and alternate cost/reversed cost and mixer
        for layer in range(self.num_layers): 
        
            bind_dict = {cost_layer.parameters[0]: gammas[layer]}
            bound_cost_layer = cost_layer.assign_parameters(bind_dict)
            
            bind_dict = {self.mixer_layer.parameters[0]: betas[layer]}
            bound_mixer_layer = self.mixer_layer.assign_parameters(bind_dict)

            if layer % 2 == 0:
                # Even layer -> append cost
                for pauli_str, coeff in self.cost_other.to_list():
                    # Identify qubits with Z operators
                    target_qubits = [i for i, char in enumerate(pauli_str[::-1]) if char == 'Z']   
                    if target_qubits:
                        # Apply Rz gates with their coefficient
                        for n in target_qubits:
                            qaoa_circuit.rz(2*coeff.real*gammas[layer], n)  # 2 factor for Qiskit sctructure
                qaoa_circuit.compose(bound_cost_layer, range(self.num_qubits), inplace=True)
            else:
                # Odd layer -> append reversed cost
                qaoa_circuit.compose(bound_cost_layer.reverse_ops(), range(self.num_qubits), inplace=True)
                for pauli_str, coeff in self.cost_other.to_list():
                    # Identify qubits with Z operators
                    target_qubits = [i for i, char in enumerate(pauli_str[::-1]) if char == 'Z']   
                    if target_qubits:
                        # Apply Rz gates with their coefficient
                        for n in target_qubits:
                            qaoa_circuit.rz(2*coeff.real*gammas[layer], n)  # 2 factor for Qiskit sctructure
        
            # The mixer layer is not reversed
            qaoa_circuit.compose(bound_mixer_layer, range(self.num_qubits), inplace=True)    
        return circuit_to_dag(qaoa_circuit)