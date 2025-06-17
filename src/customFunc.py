import numpy as np
import matplotlib.pyplot as plt


from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize




# ====================================================================
#                 Function to expand observables
# ====================================================================
def expand_observable(op: SparsePauliOp, total_qubits: int):
    """
    Expands the given observable, adding product with identity matrix 
    in order to be measured using the given number of qubits.
    -----------------------------------------
    Args:
        op (SparsePauliOp): Operator to be expanded.
        total_qubits (int): Number of qubits to be used.
    -----------------------------------------
    Returns:
        (SparsePauliOp): Expanded operator.
    """
    expanded_paulis = []
    for pauli, coeff in zip(op.paulis, op.coeffs):
        pauli_str = pauli.to_label()
        # Add identities before and after deppending on the desired position
        new_pauli = (
            pauli_str + "I" * (total_qubits - len(pauli_str))
        )
        expanded_paulis.append((new_pauli, coeff))
    return SparsePauliOp.from_list(expanded_paulis)




# ====================================================================
#             Function that calculates a expectation value
# ====================================================================
def evaluate_observable(params, ansatz, observable, estimator):
    """
    Calculates the expected value of an observable, using Qiskit.
    -----------------------------------------
    Args:
        params (Numpy 1D array): The list of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        estimator (Estimator): Qiskit estimator to use in the calculations.
    -----------------------------------------
    Returns:
        (float): Expectation value of the observable.
    """
    job = estimator.run([ansatz], [observable], [params])
    result = job.result()
    expected_value = result.values[0]
    
    return expected_value




# ====================================================================
#         Function to get the derivative of an expectation value
# ====================================================================
def evaluate_deriv(params, ansatz, observable, index, estimator):
    """
    Computes the partial derivative of an observable with respect to the given parameter.
    -----------------------------------------
    Args:
        params (Numpy 1D array): The list of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        index (int): With respect to which parameter the derivative will be taken.
        estimator (Estimator): Qiskit estimator to use in the calculations.
    -----------------------------------------
    Returns:
        (float): Expectation value of the derivative of the observable.
    """

    # Shifts for parameter-shift
    shifted_plus = params.copy()
    shifted_plus[index] += np.pi / 2

    shifted_minus = params.copy()
    shifted_minus[index] -= np.pi / 2

    value_plus = evaluate_observable(shifted_plus, ansatz, observable, estimator)
    value_minus = evaluate_observable(shifted_minus, ansatz, observable, estimator)


    deriv = 0.5 * (value_plus - value_minus)
    
    return deriv




# ====================================================================
#         Function to get the gradient of an expectation value
# ====================================================================
def evaluate_grad(params, ansatz, observable, estimator):
    """
    Computes the gradient of an observable.
    -----------------------------------------
    Args:
        params (Numpy 1D array): The list of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        estimator (Estimator): Qiskit estimator to use in the calculations.
    -----------------------------------------
    Returns:
        list(float): Gradient of the expectation valur of the observable.
    """
    grad = []

    for i in range(len(params)):
        grad.append(evaluate_deriv(params, ansatz, observable, i, estimator))
    
    return grad



# ====================================================================
#            Function that calvulates the variances
# ====================================================================
def get_variances_data(num_params, ansatz, observable, estimator, index, num_shots=100, print_progress : bool = False):
    """
    Get the variances of the expectation value of an observable and its derivative.
    -----------------------------------------
    Args:
        num_params (int): The number of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        index (int): With respect to which parameter the derivative will be taken.
        num_shots (int): Number of samples taken to compute the variances.
    -----------------------------------------
    Returns:
        (float): Variance of the expectation value of the observable.
        (float): Variance of the expectation value of the derivative.
    """

    # List to save the expected values
    value_list = []

    # List to save the partial derivatives with respect to theta_index
    deriv_list = []

    for _ in range(num_shots):

        if print_progress and (_ + 1) % (num_shots // 10) == 0:
            print(int((_ + 1) / num_shots * 100), r"\% completado.")

        rand_param_vector = 2 * np.pi *np.random.random(num_params)

        value = evaluate_observable(rand_param_vector, ansatz, observable, estimator)
        deriv = evaluate_deriv(rand_param_vector, ansatz, observable, index, estimator)

        value_list.append(value)
        deriv_list.append(deriv)

    return np.var(value_list), np.var(deriv_list)



# ====================================================================
#            VQE implementation for BP study
# ====================================================================
def VQE_minimization_BP(ansatz_function, minQubits: int, maxQubits: int, base_observable, index: list[int], initial_guess: str = "zero", minimizer: str = "COBYLA", print_info: bool = True, plot_info: bool = True):
    """
    Compute the VQE algorithm using different numbers of qubits, then plot the minimization progess and the derivatives information.
    -----------------------------------------
    Args:
        ansatz_function (method): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        minQubits (int): The smallest number of qubits used.
        maxQubits (int): The greatest number of qubits used.
        base_observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        index (list[int] or str): With respect to which parameters the derivative will be taken. If given "all", it calculates all the derivatives.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero".
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA".
    -----------------------------------------
    Returns:
        (Dictionary): 
            "minimum_values" : (list[float]): A list containing the minimum found for every number of qubits.
            "n_qubits" : (list[int]): A list containing the number of qubits used.
    """

    data = {
        "n_qubits": [],
        "minimum_values": []
    }

    for i in range(minQubits, maxQubits+1):

        estimator = Estimator()
        
        current_observable = expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansatz_function(i)

        # Initial parameters
        if initial_guess == "rand":
            initial_param_vector = np.random.random(num_params)
        elif initial_guess == "zero":
            initial_param_vector = np.zeros(num_params)
        elif initial_guess is np.ndarray():
            initial_param_vector = initial_guess
        else:
            print("Invalid initial guess, using all parameters as zero")

        # Current iteration information
        if print_info:
            print("\n=====================================================")
            print(f"Preparando ejecución para {i} qubits.")
            print(f"Se usarán {num_params} parámetros")

        # Dictionary to save the evolution of the cost function
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

        # Optimization
        res = minimize(
            cost_func,
            initial_param_vector,
            args=(ansatz_circuit, current_observable, index, estimator),
            method=minimizer,
        )

        # Save the results in the dictionary
        data["n_qubits"].append(i)
        data["minimum_values"].append(res.fun)

        # Show the evolution of the cost function
        if plot_info:
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

        if plot_info:
            print(f"Fin ejecución con {i} qubits. Mínimo encontrado: {res.fun}")
            print("=====================================================")

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