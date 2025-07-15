# imports
import numpy as np
import random
import matplotlib.pyplot as plt
import src.customFunc as cf
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from scipy.stats import linregress
#from deap import base, creator, tools


def VQE_minimization(ansatz_circuit, observable: SparsePauliOp, initial_guess: str = "zero", minimizer: str = "COBYLA", tol=None, maxiter: int=10000, fixseed: bool=False):
    """
    Compute the VQE minimization algorithm.
    -----------------------------------------
    Args:
        ansatz_circuit (method):  ansatz to optimize.
        observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero".
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA".
    -----------------------------------------
    Returns:
        cost_history_dict (dict): iterations and their cost value
    """
    if fixseed==True:
        random.seed(42)
    estimator = Estimator()
    num_params=ansatz_circuit.num_parameters

    # Initial parameters
    if initial_guess == "rand":
        initial_param_vector = np.random.random(num_params)
    elif initial_guess == "zero":
        initial_param_vector = np.zeros(num_params)
    elif isinstance(initial_guess, np.ndarray):
        initial_param_vector = initial_guess
    elif isinstance(initial_guess, list):
        initial_param_vector = initial_guess
    else:
        print("Invalid initial guess, using all parameters as zero")
        initial_param_vector = np.zeros(num_params)
  
    def cost_func(param_vector, ansatz_circuit, observable, estimator):
        cost = cf.evaluate_observable(param_vector, ansatz_circuit, observable, estimator)
        cost_history_dict["iters"] += 1
        cost_history_dict["cost_history"].append(cost)
        return cost

    # Dictionary to save the evolution of the cost function
    cost_history_dict = {"iters": 0, "cost_history": []}

    # Optimization in layers
    res = minimize(cost_func, initial_param_vector, args=(ansatz_circuit, observable, estimator), method=minimizer, options={'maxiter': maxiter}, tol=tol)
    return cost_history_dict





def VQE_minimization_layer_training(ansatz, observable: SparsePauliOp, num_layers: int, range_layers: int, direction: str = "forward", initial_guess: str = "zero", minimizer: str = "COBYLA"):
    """
    Compute the VQE minimization algorithm using layer training.
    -----------------------------------------
    Args:
        ansatz_function (method): ansatz to optimize.
        observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        num_layers (int): number of layers in the ansatz.
        direction (str): direction of layer training.
        range_layers (int): layers to be optimized individually starting from the initial layer optimized.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero".
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA".
    -----------------------------------------
    Returns:
        cost_history_dict (dict): iterations and their cost value
    """
    estimator = Estimator()
    num_params=ansatz.num_parameters

    # Initial parameters
    if initial_guess == "rand":
        initial_param_vector = np.random.random(num_params)
    elif initial_guess == "zero":
        initial_param_vector = np.zeros(num_params)
    elif initial_guess is np.ndarray():
        initial_param_vector = initial_guess
    else:
        print("Invalid initial guess, using all parameters as zero")
        
    def cost_func(param_layer, ansatz, observable, param_vector, start, end, estimator):
        full_param_vector = param_vector.copy()
        full_param_vector[start:end] = param_layer

        cost = cf.evaluate_observable(full_param_vector, ansatz, observable, estimator)
        cost_history_dict["iters"] += 1
        cost_history_dict["cost_history"].append(cost)
        return cost

    # Dictionary to save the evolution of the cost function
    cost_history_dict = {"iters": 0, "cost_history": []}

    # Optimization in layers
    param_vector=initial_param_vector
    params_per_layer = len(initial_param_vector) // num_layers
    if direction == "forward":
        layer_indices = range(range_layers)
    elif direction == "backward":
        layer_indices = reversed(range(num_layers - range_layers, num_layers))
    else:
        raise ValueError("El parámetro 'direction' debe ser 'forward' o 'backward'.")
    for layer in layer_indices:
        start = layer * params_per_layer
        end = start + params_per_layer
        initial_param_layer = param_vector[start:end]
        res = minimize(cost_func, initial_param_layer, args=(ansatz, observable, param_vector, start, end, estimator), method=minimizer, options={'maxiter': 1000})
        param_vector[start:end]=res.x

    if range_layers != num_layers:
        if direction=="forward":
            next_param_layer=param_vector[end:]
            res = minimize(cost_func, next_param_layer, args=(ansatz, observable, param_vector, end, len(param_vector), estimator), method=minimizer, options={'maxiter': 10000-cost_history_dict["iters"]})
            param_vector[end:]=res.x
        elif direction == "backward":
            next_param_layer = param_vector[:start]
            res = minimize(cost_func, next_param_layer, args=(ansatz, observable, param_vector, 0, start, estimator), method=minimizer, options={'maxiter': 10000-cost_history_dict["iters"]})
            param_vector[:start] = res.x
    return cost_history_dict







def VQE_minimization_layer_adding_training(ansatz_function, observable: SparsePauliOp, num_qubits:int, num_layers: int, direction: str = "forward", initial_guess: str = "zero", minimizer: str = "COBYLA"):
    """
    Compute the VQE minimization algorithm using layer training.
    -----------------------------------------
    Args:
        ansatz_function (method): ansatz to optimize.
        observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        num_qubits (int): number of qubits in the ansatz.
        num_layers (int): number of layers in the ansatz.
        direction (str): direction of layer training.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero".
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA".
    -----------------------------------------
    Returns:
        cost_history_dict (dict): iterations and their cost value
    """
    def cost_func(param_layer, ansatz, observable, param_vector, estimator):
        full_param_vector=np.concatenate((param_vector, param_layer))

        cost = cf.evaluate_observable(full_param_vector, ansatz, observable, estimator)
        cost_history_dict["iters"] += 1
        cost_history_dict["cost_history"].append(cost)
        return cost
    def cost_func_inv(param_layer, ansatz, observable, param_vector, estimator):
        full_param_vector=np.concatenate((param_vector[:num_params_1],param_layer, param_vector[num_params_1:]))

        cost = cf.evaluate_observable(full_param_vector, ansatz, observable, estimator)
        cost_history_dict["iters"] += 1
        cost_history_dict["cost_history"].append(cost)
        return cost
    
    estimator = Estimator()
    ansatz, num_params_1=ansatz_function(num_qubits,1)
    cost_history_dict = {"iters": 0, "cost_history": []}    # Dictionary to save the evolution of the cost function

    # Initial parameters
    if initial_guess == "rand":
        initial_param_vector = np.random.random(num_params_1)
    elif initial_guess == "zero":
        initial_param_vector = np.zeros(num_params_1)
    elif isinstance(initial_guess, np.ndarray):
        initial_param_vector = initial_guess
    else:
        print("Invalid initial guess, using all parameters as zero")

    res = minimize(cost_func, initial_param_vector, args=(ansatz, observable, np.array([]), estimator), method=minimizer, options={'maxiter': 1000})
    param_vector=res.x

    if num_layers>=2:
        if direction == "forward":
            for layer in range(2, num_layers+1):
                ansatz, num_params=ansatz_function(num_qubits,layer)
                param_layer=np.zeros(num_params-len(param_vector))
                res = minimize(cost_func, param_layer, args=(ansatz, observable, param_vector, estimator), method=minimizer, options={'maxiter': 1000})
                param_vector=np.concatenate((param_vector, res.x))
        elif direction == "backward":
            for layer in range(2, num_layers+1):
                ansatz, num_params=ansatz_function(num_qubits,layer)
                param_layer=np.zeros(num_params-len(param_vector))
                res = minimize(cost_func_inv, param_layer, args=(ansatz, observable, param_vector, estimator), method=minimizer, options={'maxiter': 1000})
                param_vector=np.concatenate((param_vector[:num_params_1], res.x, param_vector[num_params_1:]))
        else:
            raise ValueError("El parámetro 'direction' debe ser 'forward' o 'backward'.")
    return cost_history_dict






def VQE_minimization_AG(ansatz_circuit, observable : SparsePauliOp, tolerance : float, crossover_prob : float = 0.5, mutation_prob : float = 0.25, population_size : int = 100, max_gen : int = 100, print_info: bool = True, plot_info: bool = True):
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
        "minimum_value": None,
        "optimal_parameters": None,
        "n_evaluations" : None,
        "n_generations" : None,
        "cost_history" : None
    }

    estimator = Estimator()
    
    num_params = ansatz_circuit.num_parameters

    # Current iteration information
    if print_info:
        print("\n=====================================================")
        print(f"Preparando ejecución.")
        print(f"Se usarán {num_params} parámetros")
    
    # Dictionary to save the evolution of the cost function
    cost_history = []

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator 
    toolbox.register("attr_phase", np.random.uniform, 0, 2*np.pi)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_phase, num_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    def evalOneMax(individual):
        return [-cf.evaluate_observable(np.array(individual), ansatz_circuit, observable, estimator)]
    

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # From DEAP tutorial: long version
    def perform_AG():
        # Create population
        pop = toolbox.population(n=population_size)

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = crossover_prob, mutation_prob

        # Extracting all the fitnesses of 
        fits = [ind.fitness.values[0] for ind in pop]
        n_evaluations = population_size

        # Variable keeping track of the number of generations
        g = 0

        # Stop conditions vars
        avg_score = np.mean(fits)
        avg_score_prev = avg_score - tolerance -1

        # Begin the evolution
        while abs(avg_score-avg_score_prev) > tolerance and g < max_gen:
            # A new generation
            g = g + 1
        
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                n_evaluations += 1
            
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            cost_history.append(-max(fits))

            avg_score_prev = avg_score
            avg_score = np.mean(fits)

        best_fit = max(fits)
        best_params = pop[fits.index(max(fits))]

        return -best_fit, best_params, g, n_evaluations
    
    opt_value, opt_parametes, n_generations, n_evaluations= perform_AG()


    # Save the results in the dictionary
    data["minimum_value"] = opt_value
    data["optimal_parameters"] = opt_parametes
    data["n_evaluations"] = n_evaluations
    data["n_generations"] = n_generations
    data["cost_history"] = cost_history

    # Show the evolution of the cost function
    if plot_info:
        fig, ax = plt.subplots()
        ax.plot(range(1, n_generations+1), cost_history, label=r"$\langle O\rangle$")

        ax.set_xlabel("Generaciones")
        ax.set_ylabel(r"$\langle O\rangle$")
        ax.set_title(f"Minimización")
        plt.legend()
        plt.show()

    if print_info:
        print(f"Fin ejecución. Mínimo encontrado: {opt_value}")
        print(f"Número de generaciones: {n_generations}")
        print(f"Número de evaluaciones de la función de coste: {n_evaluations}")
        print("=====================================================")

    return data


def VQE_minimization_AG_plus(ansatz_circuit, observable : SparsePauliOp, stop_condition : float, crossover_prob : float = 0.5, mutation_prob : float = 0.25, population_size : int = 100, max_gen : int = 100, minimizer: str = "COBYLA", print_info: bool = True, plot_info: bool = True):

    converg = False
    total_AG_evaluations = 0
    while not converg:

        AG_result = VQE_minimization_AG(ansatz_circuit, observable, stop_condition, crossover_prob, mutation_prob, population_size, max_gen, print_info=False, plot_info=False)
        total_AG_evaluations += AG_result['n_evaluations']

        # Si el AG converge
        if AG_result["n_generations"] < max_gen:
            converg = True
        
    COBYLA_result = VQE_minimization(ansatz_circuit, observable, initial_guess=AG_result["optimal_parameters"], minimizer=minimizer)

        # Show the evolution of the cost function
    if plot_info:
        fig, ax = plt.subplots()
        ax.plot(range(1, AG_result["n_generations"]+1), AG_result["cost_history"], label="AG")
        ax.plot(range(AG_result["n_generations"]+1, AG_result["n_generations"]+COBYLA_result["iters"]+1), COBYLA_result["cost_history"], label=minimizer)

        ax.set_xlabel("Generaciones/iteraciones")
        ax.set_ylabel(r"$\langle O\rangle$")
        ax.set_title(f"Minimización")
        plt.legend()
        plt.show()

    if print_info:
        print(f"Fin ejecución. Mínimo encontrado: {COBYLA_result['cost_history'][-1]}")
        print(f"Número de generaciones usadas en AG: {AG_result['n_generations']}")
        print(f"Número total de evaluaciones de la función de coste: {total_AG_evaluations + COBYLA_result['iters']}")
        print(f"Evaluaciones AG: {total_AG_evaluations}")
        print(f"Evaluaciones {minimizer}: {COBYLA_result['iters']}")
        print("=====================================================")


        return {"AG_result" : AG_result, "minimizer_result" : COBYLA_result}
    
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
def evaluate_deriv(params, ansatz, observable, index, estimator, use_shift_rule : bool = True, delta : float = 1e-5):
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
    
    if use_shift_rule:
        # Shifts for parameter-shift
        shifted_plus = params.copy()
        shifted_plus[index] += np.pi / 2

        shifted_minus = params.copy()
        shifted_minus[index] -= np.pi / 2

        value_plus = evaluate_observable(shifted_plus, ansatz, observable, estimator)
        value_minus = evaluate_observable(shifted_minus, ansatz, observable, estimator)


        deriv = 0.5 * (value_plus - value_minus)
    
    else:
        # Forward and backward derivative
        shifted_plus = params.copy()
        shifted_plus[index] += delta

        shifted_minus = params.copy()
        shifted_minus[index] -= delta

        value_plus = evaluate_observable(shifted_plus, ansatz, observable, estimator)
        value_minus = evaluate_observable(shifted_minus, ansatz, observable, estimator)


        deriv = (value_plus - value_minus)/delta
    
    return deriv



# ====================================================================
#            Function that calculates the variances
# ====================================================================
def get_variances_data(num_params, layer_params, ansatz, observable, index, estimator=Estimator(), num_shots=100, print_progress : bool = False, use_shift_rule : bool = True, delta : float = 1e-5):
    """
    Get the variances of the expectation value of an observable and its derivative.
    -----------------------------------------
    Args:
        num_params (int): The number of parameters to be used in the calculation.
        ansatz (QuantumCircuit): The Qiskit circuit containing the ansatz, the parametrized quantum circuit.
        observable (SparsePauliOp): The observable to be measured.
        index (int): With respect to which parameter the derivative will be taken.
        num_shots (int): Number of samples taken to compute the variances.
        print_progress (bool): If the completation percentage of the current variances will be printed, useful for heavy calculations.
    -----------------------------------------
    Returns:
        (float): Variance of the expectation value of the observable.
        (float): Variance of the expectation value of the derivative.
    """

    # List to save the expected values
    value_list = []

    # List to save the partial derivatives with respect to theta_index
    deriv_list = []

    param_vector=np.zeros(num_params)

    for _ in range(num_shots):

        if print_progress and (_ + 1) % (num_shots // 10) == 0:
            print(int((_ + 1) / num_shots * 100), r"\% completado.")

        param_vector[:layer_params] = 2 * np.pi *np.random.random(layer_params)

        value = evaluate_observable(param_vector, ansatz, observable, estimator)
        deriv = evaluate_deriv(param_vector, ansatz, observable, index, estimator, use_shift_rule=use_shift_rule, delta=delta)

        value_list.append(value)
        deriv_list.append(deriv)

    return np.var(value_list), np.var(deriv_list)



def variance_vs_nQubits(ansantz_function, layers, layer_train: bool, minQubits: int, maxQubits: int, base_observable : SparsePauliOp, index: int, num_shots : int = 100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False, only_even_qubits : bool=False, print_progress : bool=False, use_shift_rule : bool = True, delta : float = 1e-5):
    """
    Obtain the variances of the expectation value and the given derivative using different numbers of qubits.
    -----------------------------------------
    Args:
        ansatz_function (function): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        minQubits (int): The smallest number of qubits used.
        maxQubits (int): The greatest number of qubits used.
        base_observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        index (int): With respect to which parameter the derivative will be taken.
        num_shots (int): Number of samples taken to compute the variances.
        print_info (bool): If the results will be printed.
        plot_info (bool): If the results will be plotted.
        do_regress (bool): If a linear regression will be performed.
        only_even_qubits (bool): To use only a even number of qubits, usefull for UCCSD.
        print_progress (bool): If the completation percentage of the current variances will be printed, useful for heavy calculations.
    -----------------------------------------
    Returns:
        (Dictionary): 
            "n_qubits" : (list[int])
            "var_value" : (list[float])
            "var_deriv" : (list[float])
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

    estimator = Estimator()

    for i in range(minQubits, maxQubits+1):

        if not (only_even_qubits and i%2!=0):
            
            if base_observable == "global_observable":
                current_observable = cf.global_observable(i)

            else:
                current_observable = cf.expand_observable(base_observable, i)
            
            if layer_train==True:
                ansatz_circuit, layer_params = ansantz_function(i, 1)
                ansatz_circuit, num_params = ansantz_function(i, layers)
            else:
                ansatz_circuit, num_params = ansantz_function(i, layers)
                layer_params=num_params

            if print_info:
                print("\n=====================================================")
                print(f"Calculando varianzas con {i} qubits.\n")
            
            var_value, var_deriv = get_variances_data(num_params, layer_params, ansatz_circuit, current_observable, index, estimator, num_shots, print_progress=print_progress, use_shift_rule=use_shift_rule, delta=delta)

            # Current iteration information
            if print_info:
                print(f"Varianza del valor esperado: {var_value}")
                print(f"Varianza de la derivada: {var_deriv}")

            data["n_qubits"].append(i)
            data["var_value"].append(var_value)
            data["var_deriv"].append(var_deriv)

    # Regression
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
            print(f"R^2 para derivada: {data['deriv_rsquare']}.")
    
    # Shows result's concentration and its derivative
    if plot_info:
        fig, ax = plt.subplots()
        
        # Scatter
        ax.scatter(data["n_qubits"], data["var_value"], label=r"Var($\langle O\rangle$)")
        ax.scatter(data["n_qubits"], data["var_deriv"], label=rf"Var($\partial_{index}\langle O\rangle$)")

        # Tendencies
        if do_regress:
            base = np.linspace(minQubits, maxQubits, 100)
            ax.plot(base, np.exp(data["value_slope"]*base+data["value_ord"]), color="black", label=r"Tendencia: Var($\langle O\rangle$)")
            ax.plot(base, np.exp(data["deriv_slope"]*base+data["deriv_ord"]), color="red", label=rf"Tendencia: Var($\partial_{index}\langle O\rangle$)")

        # Settings
        ax.set_xlabel(r"$N$ qubits")
        ax.set_title(rf"BP en VQE, variando el parámetro $\theta_{index}$")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    return data