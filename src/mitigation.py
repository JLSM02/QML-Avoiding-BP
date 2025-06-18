# imports
import numpy as np
import matplotlib.pyplot as plt
import customFunc as cf
from scipy.optimize import minimize
from qiskit.primitives import Estimator
from deap import base, creator, tools


def VQE_minimization(ansatz, observable, initial_guess: str = "zero", minimizer: str = "COBYLA"):
    """
    Compute the VQE minimization algorithm.
    -----------------------------------------
    Args:
        ansatz (method):  ansatz to optimize.
        observable (SparsePauliOp): The observable to be measured in its minimal form, it should use minQubits number of qubits.
        initial_guess (str or NumPy 1D array): "zero" initial guess with all parameters equal to cero, "rand" -> random initial guess. 1D Array -> the initial guess. default="zero".
        minimizer (str): scipy.optimize.minimize possible optimization methods, default="COBYLA".
    -----------------------------------------
    Returns:
        res.fun: optimized cost value
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
  
    def cost_func(param_vector, ansatz, observable, estimator):
        cost = cf.evaluate_observable(param_vector, ansatz, observable, estimator)
        cost_history_dict["iters"] += 1
        cost_history_dict["cost_history"].append(cost)
        return cost

    # Dictionary to save the evolution of the cost function
    cost_history_dict = {"iters": 0, "cost_history": []}

    # Optimization in layers
    res = minimize(cost_func, initial_param_vector, args=(ansatz, observable, estimator), method=minimizer)
    return cost_history_dict





def VQE_minimization_layer_training(ansatz, observable, num_layers: int, range_layers: int, direction: str = "forward", initial_guess: str = "zero", minimizer: str = "COBYLA"):
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
        res.fun: optimized cost value
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
        layer_indices = reversed(range(range_layers))
    else:
        raise ValueError("El parámetro 'direction' debe ser 'forward' o 'backward'.")
    for layer in layer_indices:
        start = layer * params_per_layer
        end = start + params_per_layer
        initial_param_layer = param_vector[start:end]
        res = minimize(cost_func, initial_param_layer, args=(ansatz, observable, param_vector, start, end, estimator), method=minimizer)
        param_vector[start:end]=res.x

    if range_layers != num_layers:
        if direction=="forward":
            next_param_layer=param_vector[end:]
            res = minimize(cost_func, next_param_layer, args=(ansatz, observable, param_vector, end, len(param_vector), estimator), method=minimizer)
            param_vector[end:]=res.x
        elif direction == "backward":
            next_param_layer = param_vector[:start]
            res = minimize(cost_func, next_param_layer, args=(ansatz, observable, param_vector, 0, start, estimator), method=minimizer)
            param_vector[:start] = res.x
<<<<<<< HEAD
    return res.fun, cost_history_dict







def VQE_minimization_AG(ansatz_function, minQubits: int, maxQubits: int, base_observable, stop_condition : float, population_size : int = 100, max_iters : int = 100, print_info: bool = True, plot_info: bool = True):
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
        "minimum_values": [],
        "optimal_parameters": [],
        "n_evaluations" : [],
        "n_generations" : []
    }

    for i in range(minQubits, maxQubits+1):

        estimator = Estimator()
        
        current_observable = cf.expand_observable(base_observable, i)
        ansatz_circuit, num_params = ansatz_function(i)

        # Current iteration information
        if print_info:
            print("\n=====================================================")
            print(f"Preparando ejecución para {i} qubits.")
            print(f"Se usarán {num_params} parámetros")
        
        # Dictionary to save the evolution of the cost function
        cost_history_dict = {
            "cost_history": []
        }

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
            return [-cf.evaluate_observable(np.array(individual), ansatz_circuit, current_observable, estimator)]
        

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

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
            CXPB, MUTPB = 0.5, 0.2

            # Extracting all the fitnesses of 
            fits = [ind.fitness.values[0] for ind in pop]
            n_evaluations = population_size

            # Variable keeping track of the number of generations
            g = 0

            # Begin the evolution
            while max(fits) < -stop_condition and g < max_iters:
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

                cost_history_dict["cost_history"].append(-max(fits))

            best_fit = max(fits)
            best_params = pop[fits.index(max(fits))]

            return -best_fit, best_params, g, n_evaluations
        
        opt_value, opt_parametes, n_generations, n_evaluations= perform_AG()


        # Save the results in the dictionary
        data["n_qubits"].append(i)
        data["minimum_values"].append(opt_value)
        data["optimal_parameters"].append(opt_parametes)
        data["n_evaluations"].append(n_evaluations)
        data["n_generations"].append(n_generations)

        # Show the evolution of the cost function
        if plot_info:
            fig, ax = plt.subplots()
            ax.plot(range(1, n_generations+1), cost_history_dict["cost_history"], label=r"$\langle O\rangle$")

            ax.set_xlabel("Generaciones")
            ax.set_ylabel(r"$\langle O\rangle$")
            ax.set_title(f"Minimización para {i} qubits")
            plt.legend()
            plt.show()

        if print_info:
            print(f"Fin ejecución con {i} qubits. Mínimo encontrado: {opt_value}")
            print(f"Número de generaciones: {n_generations}")
            print(f"Número de evaluaciones de la función de coste: {n_evaluations}")
            print("=====================================================")

    return data
=======
    return cost_history_dict
>>>>>>> dfa3863520f1d4bb181687103a72822e3259bdaf
