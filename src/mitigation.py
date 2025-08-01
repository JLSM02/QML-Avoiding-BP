# imports
import numpy as np
import matplotlib.pyplot as plt
import src.customFunc as cf
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from deap import base, creator, tools


def VQE_minimization(ansatz_circuit, observable: SparsePauliOp, initial_guess: str = "zero", minimizer: str = "COBYLA", tol=None):
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
    res = minimize(cost_func, initial_param_vector, args=(ansatz_circuit, observable, estimator), method=minimizer, options={'maxiter': 10000}, tol=tol)
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