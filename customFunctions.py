import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize
from scipy.stats import linregress


# ====================================================================
#                 Función para expandir el observable
# ====================================================================
def expand_observable(op: SparsePauliOp, total_qubits: int):
    """
    Expands the given observable, adding product with identity matrix in order to be measured using the given number of qubits.

    Args:
        op (SparsePauliOp): Operator to be expanded.
        total_qubits (int): Number of qubits to be used.

    Returns:
        (SparsePauliOp): Expanded operator.
    """
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
def build_deep_ansatz(layers_per_qubit: int, num_qubits: int):
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
#             Función para calcular el valor esperado
# ====================================================================
def evaluate_observable(params, ansatz, observable, estimator):
    """
    Calculates the expecter value of an observable, using Qiskit.

    Args:
        params (Numpy 1D array): The list of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        estimator (Estimator): Qiskit estimator to use in the calculations.

    Returns:
        (float): Expectation value of the observable.
    """
    job = estimator.run([ansatz], [observable], [params])
    result = job.result()
    expected_value = result.values[0]
    
    return expected_value



# ====================================================================
#            Función para la derivada del valor esperado
# ====================================================================
def evaluate_deriv(params, ansatz, observable, index, estimator):
    """
    Computes the partial derivative of an observable with respect to the given parameter.

    Args:
        params (Numpy 1D array): The list of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        index (int): With respect to which parameter the derivative will be taken.
        estimator (Estimator): Qiskit estimator to use in the calculations.

    Returns:
        (float): Expectation value of the derivative of the observable.
    """

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
def get_variances_data(num_params, ansatz, observable, index, num_shots=100):
    """
    Get the variances of the expectation value of an observable and its derivative.

    Args:
        num_params (int): The number of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        index (int): With respect to which parameter the derivative will be taken.
        num_shots (int): Number of samples taken to compute the variances.

    Returns:
        (float): Variance of the expectation value of the observable.
        (float): Variance of the expectation value of the derivative.
    """
    
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
        deriv_list.append(deriv)

    return np.var(value_list), np.var(deriv_list)



# ====================================================================
#            Función para minimización VQE
# ====================================================================
def VQE_minimization_BP(ansatz_function, minQubits: int, maxQubits: int, base_observable, index: list[int], initial_guess: str = "zero", minimizer: str = "COBYLA"):
    """
    Compute the VQE algorithm using different numbers of qubits, then plot the minimization progess and the derivatives information.

    Args:
        ansatz_function (method): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        minQubits (int): The smallest number of qubits used.
        maxQubits (int): The greatest number of qubits used.
        base_observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        index (list[int] or str): With respect to which parameters the derivative will be taken. If given "all", it calculates all the derivatives.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero"
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA"
    """
    for i in range(minQubits, maxQubits+1):

        estimator = Estimator()
        
        current_observable = expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansatz_function(i)

        # Parámetros iniciales
        if initial_guess == "rand":
            initial_param_vector = np.random.random(num_params)
        elif initial_guess == "zero":
            initial_param_vector = np.zeros(num_params)
        elif initial_guess is np.ndarray():
            initial_param_vector = initial_guess
        else:
            print("Invalid initial guess, using all parameters as zero")

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
        if index == "all":
            for j in range(num_params):
                cost_history_dict["deriv_history"].append([])
        else:
            for j in index:
                cost_history_dict["deriv_history"].append([])
        
        def cost_func(params, ansatz, observable, index, estimator):

            cost = evaluate_observable(params, ansatz, observable, estimator)
            cost_history_dict["iters"] += 1
            cost_history_dict["cost_history"].append(cost)

            if index == "all":
                for j in range(num_params):
                    deriv = evaluate_deriv(params, ansatz, observable, j, estimator)
                    cost_history_dict["deriv_history"][j].append(deriv)
            
            else:
                for j in index:
                    deriv = evaluate_deriv(params, ansatz, observable, j, estimator)
                    cost_history_dict["deriv_history"][j].append(deriv)

            return cost

        # Ejecutamos la optimización
        res = minimize(
            cost_func,
            initial_param_vector,
            args=(ansatz_circuit, current_observable, index, estimator),
            method=minimizer,
        )

        # Graficar evolución del costo
        fig, ax = plt.subplots()
        ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"], label=r"$\langle O\rangle$")

        if index == "all":
            for j in range(num_params):
                ax.plot(range(cost_history_dict["iters"]), cost_history_dict["deriv_history"][j], label=rf"$\partial_{{{j}}}\langle O\rangle$")

        else:
            for j in index:
                ax.plot(range(cost_history_dict["iters"]), cost_history_dict["deriv_history"][j], label=rf"$\partial_{{{j}}}\langle O\rangle$")

        ax.set_xlabel("Iteraciones")
        ax.set_ylabel(r"$\langle O\rangle$")
        ax.set_title(f"Minimización para {i} qubits")
        plt.legend()
        plt.show()

        print(f"Fin ejecución con {i} qubits. Mínimo encontrado: {res.fun}")
        print("=====================================================")



# ====================================================================
#            Función para varianza de gradientes
# ====================================================================
def variance_vs_nQubits(ansantz_function, minQubits: int, maxQubits: int, base_observable, index: int, num_shots=100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False):
    """
    Obtain the variances of the expectation value and the given derivative using different numbers of qubits.

    Args:
        ansatz_function (method): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        minQubits (int): The smallest number of qubits used.
        maxQubits (int): The greatest number of qubits used.
        base_observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        index (int): With respect to which parameter the derivative will be taken.
        num_shots (int): Number of samples taken to compute the variances.
        print_info (bool): If the results will be printed
        plot_info (bool): If the results will be plotted


    Returns:
        (Dictionary): 
            "n_qubits" : (list[int])
            "var_value" : (list[float])
            "var_deriv" : (list[float])"
            "value_slope" : (int)
            "value_ord" : (int)
            "value_rsquare" : (int)
            "deriv_slope" : (int)
            "deriv_ord" : (int)
            "deriv_rsquare" : (int)
    """
    data = {
        "n_qubits": [],
        "var_value": [],
        "var_deriv": [],
        "value_slope": 0,
        "value_ord": 0,
        "value_rsquare": 0,
        "deriv_slope": 0,
        "deriv_ord": 0,
        "deriv_rsquare": 0
    }

    for i in range(minQubits, maxQubits+1):
        
        current_observable=expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansantz_function(i)

        if print_info:
            print("\n=====================================================")
            print(f"Calculando varianzas con {i} qubits.\n")
        
        var_value, var_deriv = get_variances_data(num_params, ansatz_circuit, current_observable, index, num_shots)

        # Información sobre la iteración actual
        if print_info:
            print(f"Varianza del valor esperado: {var_value}")
            print(f"Varianza de la derivada: {var_deriv}")

        data["n_qubits"].append(i)
        data["var_value"].append(var_value)
        data["var_deriv"].append(var_deriv)

    # Regresiones
    if do_regress:
        value_regress = linregress(data["n_qubits"], np.log(data["var_value"]))
        deriv_regress = linregress(data["n_qubits"], np.log(data["var_deriv"]))

        data["value_slope"] = value_regress[0]
        data["value_ord"] = value_regress[1]
        data["value_rsquare"] = value_regress[2]**2

        data["deriv_slope"] = deriv_regress[0]
        data["deriv_ord"] = deriv_regress[1]
        data["deriv_rsquare"] = deriv_regress[2]**2

        if print_info:
            print("\n=====================================================")
            print(f"Pendiente para valor esperado: {data['value_slope']}.")
            print(f"R^2 para valor esperado: {data['value_rsquare']}.")

            print("\n=====================================================")
            print(f"Pendiente para derivada: {data['deriv_slope']}.")
            print(f"R^2 para valor esperado: {data['deriv_rsquare']}.")
    
    # Grafica concentracion del resultado y su derivada
    if plot_info:
        fig, ax = plt.subplots()
        
        # Scatter
        ax.scatter(data["n_qubits"], data["var_value"], label=r"Var($\langle O\rangle$)")
        ax.scatter(data["n_qubits"], data["var_deriv"], label=rf"Var($\partial_{index}\langle O\rangle$)")

        # Tendencias
        if do_regress:
            base = np.linspace(2, maxQubits, 100)
            ax.plot(base, np.exp(data["value_slope"]*base+data["value_ord"]), color="black", label=r"Tendencia: Var($\langle O\rangle$)")
            ax.plot(base, np.exp(data["deriv_slope"]*base+data["deriv_ord"]), color="red", label=rf"Tendencia: Var($\partial_{index}\langle O\rangle$)")

        # Ajustes
        ax.set_xlabel(r"$N$ qubits")
        ax.set_title(rf"BP en VQE, variando el parámetro $\theta_{index}$")
        ax.set_yscale("log")
        ax.legend()
        plt.show()


    return data




def print_logo():
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@+ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@ *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@  @@@@@=   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@:        @@@@@@@ @@@@@@@@@@@@@@@ @@@@  .    @    @@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@     @@@@@@@@@@@@@@@    @      =@@@@@@ *@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@  -   #@@@@@@    @ @@@@@   @@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@   @@@:  -@@@  @@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@     @@@@@:@@@@@     @@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@   @@@@         @@@@   @@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@   @  @@@           @@@@ @#  @@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  =  @@@@ :@@            @@@@ @@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    :@@@@ :@@             @@@ @@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@  @@  @@@           @@@@ @@  @@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@   @@@@         @@@@.  %@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@     @@@@@ @@@@@@    @@@@@@@@@@@@@ =@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@  @@@   @@@@@   @@@  @@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@- @@@@@@@@@@@@@@@@ @@@@@@@    @@@@@@  @@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@ #@    @@@@@    @@ @@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@.    @@@@@@@@@@@@@@@     @@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@          @@@@@@ @@@@@@@@@@@@@@@ %@@@@@          #@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ *@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")