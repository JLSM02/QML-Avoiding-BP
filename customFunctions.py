import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize


# ====================================================================
#                 Función para expandir el observable
# ====================================================================
def expand_observable(op, total_qubits):
    expanded_paulis = []
    for pauli, coeff in zip(op.paulis, op.coeffs):
        pauli_str = pauli.to_label()
        # Añadir identidades antes y después según la posición deseada
        new_pauli = (
            pauli_str + "I" * (total_qubits - len(pauli_str))
        )
        expanded_paulis.append((new_pauli, coeff))
    return SparsePauliOp.from_list(expanded_paulis)



# ====================================================================
#                 Función construir un deep ansatz
# ====================================================================
def build_deep_ansatz(num_qubits):
    """Crea un circuito con L = 10n capas de ansatz para n qubits."""
    L = 10 * num_qubits  # número de capas
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

    # Devuelve el cirvuito y el numero de parametros
    num_params =  len(thetas)*num_qubits
    return qc, num_params



# ====================================================================
#             Función para calcular el valor esperado
# ====================================================================
def evaluate_observable(params, ansatz, observable, estimator):
    job = estimator.run([ansatz], [observable], [params])
    result = job.result()
    expected_value = result.values[0]
    
    return expected_value



# ====================================================================
#            Función para la derivada del valor esperado
# ====================================================================
def evaluate_deriv(params, ansatz, observable, index, estimator):

    # Desplazamientos para parameter-shift
    shifted_plus = params.copy()
    shifted_plus[index] += np.pi / 2

    shifted_minus = params.copy()
    shifted_minus[index] -= np.pi / 2

    value_plus = evaluate_observable(shifted_plus, ansatz, observable, estimator)
    value_minus = evaluate_observable(shifted_minus, ansatz, observable, estimator)


    deriv = 0.5 * (value_plus - value_minus)
    
    return deriv



# ====================================================================
#            Función para la obtener las varianzas
# ====================================================================
def get_variances_data(observable, ansatz, num_params, index, num_shots=1000):
    estimator = Estimator()

    # Lista para guardar los valores esperados
    value_list = []

    # Lista para guardar las derivadas respecto a theta_index
    deriv_list = []

    for _ in range(num_shots):

        rand_param_vector = 2 * np.pi *np.random.random(num_params)

        value = evaluate_observable(rand_param_vector, ansatz, observable, estimator)
        deriv = evaluate_deriv(rand_param_vector, ansatz, observable, index, estimator)

        value_list.append(value)
        value_list.append(deriv)

    return np.var(value_list), np.var(deriv_list)



# ====================================================================
#            Función para minimización VQE
# ====================================================================
def VQE_minimization(ansantz_function, minQubits, maxQubits, base_observable, index):

    print_logo()

    for i in range(minQubits, maxQubits+1):

        estimator = Estimator()
        
        current_observable=expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansantz_function(i)

        # Parámetros iniciales
        initial_param_vector = np.zeros(num_params)

        # Información sobre la iteración actual
        print("\n=====================================================")
        print(f"Preparando ejecución para {i} qubits.")
        print(f"Se usarán {num_params} parámetros")

        # Diccionario para almacenar la evolución del costo
        cost_history_dict = {
            "iters": 0,
            "cost_history": [],
            "deriv_history": [],
        }

        # If index = None, the derivative is not calculated
        if index==None:
            def cost_func(params, ansatz, observable, index, estimator):

                cost = evaluate_observable(params, ansatz, observable, estimator)

                cost_history_dict["iters"] += 1
                cost_history_dict["cost_history"].append(cost)

                return cost
        else:
            def cost_func(params, ansatz, observable, index, estimator):

                cost = evaluate_observable(params, ansatz, observable, estimator)
                deriv = evaluate_deriv(params, ansatz, observable, index, estimator)

                cost_history_dict["iters"] += 1
                cost_history_dict["cost_history"].append(cost)
                cost_history_dict["deriv_history"].append(deriv)

                return cost

        # Ejecutamos la optimización
        res = minimize(
            cost_func,
            initial_param_vector,
            args=(ansatz_circuit, current_observable, index, estimator),
            method="COBYLA",
        )

        # Graficar evolución del costo
        fig, ax = plt.subplots()
        ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"], label="Funcion de costo")
        ax.plot(range(cost_history_dict["iters"]), cost_history_dict["deriv_history"], label="Derivada")
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Energía")
        ax.set_title(f"Minimización para {i} qubits")
        plt.legend()
        plt.show()

        print(f"Fin ejecución con {i} qubits. Mínimo encontrado: {res.fun}")
        print("=====================================================")


# ====================================================================
#            Función para varianza de gradientes
# ====================================================================
def variance_vs_nQubits(ansantz_function, minQubits, maxQubits, base_observable, index, shots):
    data = []

    print_logo()

    for i in range(minQubits, maxQubits+1):
        
        current_observable=expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansantz_function(i)

        # Información sobre la iteración actual
        print("\n=====================================================")
        print(f"Calculando varianzas con {i} qubits.\n")
        
        value, deriv = get_variances_data(current_observable, ansatz_circuit, num_params, index, shots)
        print(f"Varianza del valor esperado: {value}")
        print(f"Varianza de la derivada: {deriv}")

        data.append([value, deriv])

    data = np.array(data)
    # Grafica concentracion del resultado y su derivada
    fig, ax = plt.subplots()
    ax.scatter(data[:,2], data[:,1], label="Var(E)")
    ax.scatter(data[:,2], data[:,0], label=r"Var($\partial$E)")
    ax.set_xlabel("N qubits")
    ax.set_ylabel("var(E)")
    ax.set_title(f"BP en VQE")
    ax.set_yscale("log")
    ax.legend()
    plt.show()

    return data




logo = "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@+ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@ *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@  @@@@@=   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@:        @@@@@@@ @@@@@@@@@@@@@@@ @@@@  .    @    @@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@     @@@@@@@@@@@@@@@    @      =@@@@@@ *@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@  -   #@@@@@@    @ @@@@@   @@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@ @@@@@@     @@@@@@ .@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@   @@@:  -@@@  @@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@     @@@@@:@@@@@     @@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@   @@@@         @@@@   @@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@   @  @@@           @@@@ @#  @@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  =  @@@@ :@@            @@@@ @@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    :@@@@ :@@             @@@ @@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@  @@  @@@           @@@@ @@  @@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@   @@@@         @@@@.  %@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@     @@@@@ @@@@@@    @@@@@@@@@@@@@ =@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@  @@@   @@@@@   @@@  @@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@- @@@@@@@@@@@@@@@@ @@@@@@@    @@@@@@  @@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@ #@    @@@@@    @@ @@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@.    @@@@@@@@@@@@@@@     @@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@          @@@@@@ @@@@@@@@@@@@@@@ %@@@@@          #@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ *@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
def print_logo():
    print(logo)