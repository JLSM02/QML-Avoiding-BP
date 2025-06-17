import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import Estimator

import sys
sys.path.append('../../../')
from src import customFunc as cf

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
    return res.fun, cost_history_dict

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
    return res.fun, cost_history_dict