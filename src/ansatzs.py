from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import NLocal, TwoLocal, CCXGate, CRZGate, RXGate
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (HighLevelSynthesis, InverseCancellation)
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (SwapStrategy, FindCommutingPauliEvolutions, Commuting2qGateRouter)
from qiskit.circuit.library import CXGate, RZGate, RXGate, XGate, HGate, SXGate, SXdgGate, UGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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
    
    theta1 = Parameter("θ1")
    theta2 = Parameter("θ2")
    n = 6  # número de qubits

    qc = QuantumCircuit(num_qubits)

    for _l in range(layers):
        # Bloque de RY
        for i in range(num_qubits):
            qc.ry(theta1, i)

        # Cadena de CNOTs
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        # Bloque de RZ
        for i in range(num_qubits):
            qc.rz(theta2, i)
        
    return qc, 2



def optimize_ansatz(ansatz_naive):
    # Choose swap strategy (in this case -> line)
    num_qubits=ansatz_naive.num_qubits
    swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
    edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}

    # Define pass manager
    init_cost_layer = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap_strategy, edge_coloring,), HighLevelSynthesis(basis_gates=["x", "u", "h", "cx", "sx", "rz", "rx"]), InverseCancellation(gates_to_cancel=[CXGate(), XGate(), HGate(), (RZGate(np.pi), RZGate(-np.pi)), (RZGate(np.pi/2), RZGate(-np.pi/2)), (SXGate(),SXdgGate())])])

    # Create a circuit for the 2 qubit gates and optimize it with the cost layer pass manager
    ansatz_opt=init_cost_layer.run(ansatz_naive)

    return ansatz_opt



def get_cx_count(ansatz, backend):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = pm.run(ansatz)
    ops = transpiled.count_ops()
    return transpiled, ops.get('cx', 0)



def iterate_ansatz_opt(ansatz_naive, backend):
    # Inicialization
    ansatz_opt = optimize_ansatz(ansatz_naive)
    transpiled_ansatz_opt, num_cx_prev = get_cx_count(ansatz_opt, backend)

    # Optimization loop
    while True:
        ansatz_opt_prev=ansatz_opt
        transpiled_ansatz_opt_prev=transpiled_ansatz_opt
        ansatz_opt = optimize_ansatz(ansatz_opt)
        transpiled_ansatz_opt, num_cx = get_cx_count(ansatz_opt, backend)
    
        if num_cx < num_cx_prev:
            num_cx_prev = num_cx
        else:
            break
    return ansatz_opt_prev, transpiled_ansatz_opt_prev, num_cx_prev


