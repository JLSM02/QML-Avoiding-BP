# Imports
import src.customFunc as cf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import Estimator, BackendEstimator
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import fake_backend


# ====================================================================
#            Look for BP by studying variances concentration
# ====================================================================


def variance_vs_nQubits(ansantz_function, minQubits: int, maxQubits: int, base_observable : SparsePauliOp, index: int, num_shots : int = 100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False, only_even_qubits : bool=False, print_progress : bool=False, use_shift_rule : bool = True, delta : float = 1e-5):
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
            
            current_observable = cf.expand_observable(base_observable, i)
            ansatz_circuit, num_params = ansantz_function(i)

            if print_info:
                print("\n=====================================================")
                print(f"Calculando varianzas con {i} qubits.\n")
            
            var_value, var_deriv = cf.get_variances_data(num_params, ansatz_circuit, current_observable, estimator, index, num_shots, print_progress=print_progress, use_shift_rule=use_shift_rule, delta=delta)

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



def noisy_variance_vs_nQubits(ansantz_function, noise_backend : fake_backend, noise_scale : float, minQubits: int, maxQubits: int, base_observable, index: int, num_shots : int=100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False, only_even_qubits : bool=False, print_progress : bool=False, use_shift_rule : bool = True, delta : float = 1e-5):
    """
    Obtain the variances of the expectation value and the given derivative using different numbers of qubits.
    -----------------------------------------
    Args:
        ansatz_function (method): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        noise_backend (fake_backend) The backend from which the noise model will be extracted.
        noise_scale (float): The amount of noise from the given fake_backend to be used.
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

    # Si escalo o no el ruido
    if noise_scale != 1:
        original_noise = NoiseModel.from_backend(noise_backend)

        # Escalado del error (por ejemplo, aumentar 50%) -> 1.5
        # Usa <1.0 para reducir el ruido
        # Crear un nuevo modelo de ruido escalado
        scaled_noise = NoiseModel()

        # Copiar y escalar errores de compuertas
        for instr, qubits in original_noise._local_quantum_errors.items():
            for q, err in qubits.items():
                prob = err.to_dict()['probabilities'][0]  # Supone un solo error tipo
                new_prob = min(prob * noise_scale, 1.0)  # Asegúrate de no pasar de 1
                gate_name = instr
                qubit = q

                # Crear nuevo error con misma estructura
                if len(qubit) == 1:
                    new_error = depolarizing_error(new_prob, 1)
                elif len(qubit) == 2:
                    new_error = depolarizing_error(new_prob, 2)
                else:
                    continue  # No soportamos otros tamaños aquí

                scaled_noise.add_quantum_error(new_error, gate_name, qubit)

        # Ahora usas scaled_noise en el simulador
        noisy_simulator = AerSimulator(noise_model=scaled_noise)

        # Creo el estimator para el circuito ruidoso
        estimator = BackendEstimator(backend=noisy_simulator)
    
    # Si no escalo el ruido, uso el fakeBackend directamente
    else:
        estimator = BackendEstimator(fake_backend)

    for i in range(minQubits, maxQubits+1):

        if not (only_even_qubits and i%2!=0):
            
            current_observable = cf.expand_observable(base_observable, i)
            ansatz_circuit, num_params = ansantz_function(i)

            if print_info:
                print("\n=====================================================")
                print(f"Calculando varianzas con {i} qubits.\n")
            
            var_value, var_deriv = cf.get_variances_data(num_params, ansatz_circuit, current_observable, estimator, index, num_shots, print_progress=print_progress, use_shift_rule=use_shift_rule, delta=delta)

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
        ax.set_title(rf"BP en VQE, variando el parámetro $\theta_{index}$, con escala de ruido {noise_scale:.3f}")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    return data




def variance_vs_layers(ansantz_function, minLayers: int, maxLayers: int, n_qubits : int, base_observable : SparsePauliOp, index: int, num_shots : int=100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False, print_progress : bool=False, use_shift_rule : bool = True, delta : float = 1e-5):
    """
    Obtain the variances of the expectation value and the given derivative using different numbers of qubits.
    -----------------------------------------
    Args:
        ansatz_function (function): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        minLayers (int): The smallest number of layers used.
        maxlayers (int): The greatest number of layers used.
        n_qubits (int): The number of qubits to be used.
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
            "n_layers" : (list[int])
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
        "n_layers": [],
        "var_value": [],
        "var_deriv": [],
        "value_slope": 0,
        "value_ord": 0,
        "value_rsquare": 0,
        "deriv_slope": 0,
        "deriv_ord": 0,
        "deriv_rsquare": 0
    }

    # Creo el estimator para el circuito ruidoso
    estimator = Estimator()
    
    for layers in range(minLayers, maxLayers+1):

        current_observable = cf.expand_observable(base_observable, n_qubits)
        ansatz_circuit, num_params = ansantz_function(n_qubits, layers)

        if print_info:
            print("\n=====================================================")
            print(f"Calculando varianzas con nº capas: {layers}.\n")
        
        var_value, var_deriv = cf.get_variances_data(num_params, ansatz_circuit, current_observable, estimator, index, num_shots, print_progress=print_progress, use_shift_rule=use_shift_rule, delta=delta)

        # Current iteration information
        if print_info:
            print(f"Varianza del valor esperado: {var_value}")
            print(f"Varianza de la derivada: {var_deriv}")

        data["layers"].append(layers)
        data["var_value"].append(var_value)
        data["var_deriv"].append(var_deriv)
    

    # Regression
    if do_regress:
        value_regress = linregress(data["layers"], np.log(data["var_value"]))
        deriv_regress = linregress(data["layers"], np.log(data["var_deriv"]))

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
        ax.scatter(data["layers"], data["var_value"], label=r"Var($\langle O\rangle$)")
        ax.scatter(data["layers"], data["var_deriv"], label=rf"Var($\partial_{index}\langle O\rangle$)")

        # Tendencies
        if do_regress:
            base = np.linspace(minLayers, maxLayers, 100)
            ax.plot(base, np.exp(data["value_slope"]*base+data["value_ord"]), color="black", label=r"Tendencia: Var($\langle O\rangle$)")
            ax.plot(base, np.exp(data["deriv_slope"]*base+data["deriv_ord"]), color="red", label=rf"Tendencia: Var($\partial_{index}\langle O\rangle$)")

        # Settings
        ax.set_xlabel("N capas")
        ax.set_title(rf"BP en VQE, variando el parámetro $\theta_{index}$")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    return data





def noisy_variance_vs_layers(ansantz_function, noise_backend : fake_backend, noise_scale : float, minLayers: int, maxLayers: int, n_qubits : int, base_observable, index: int, num_shots : int=100, print_info: bool=True, plot_info: bool=True, do_regress : bool=False, print_progress : bool=False, use_shift_rule : bool = True, delta : float = 1e-5):
    """
    Obtain the variances of the expectation value and the given derivative using different numbers of qubits.
    -----------------------------------------
    Args:
        ansatz_function (method): A function defined as follows: ansatz_function(N_qubits (int)) -> qc (QuantumCircuit), num_params (int)
        noise_backend (fake_backend) The backend from which the noise model will be extracted.
        noise_scale (float): The amount of noise from the given fake_backend to be used.
        minLayers (int): The smallest number of layers used.
        maxLayers (int): The greatest number of layers used.
        n_qubits (int): The number of qubits to be used.
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
            "n_layers" : (list[int])
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
        "n_layers": [],
        "var_value": [],
        "var_deriv": [],
        "value_slope": 0,
        "value_ord": 0,
        "value_rsquare": 0,
        "deriv_slope": 0,
        "deriv_ord": 0,
        "deriv_rsquare": 0
    }

    # Si escalo o no el ruido
    if noise_scale != 1:
        original_noise = NoiseModel.from_backend(noise_backend)

        # Escalado del error (por ejemplo, aumentar 50%) -> 1.5
        # Usa <1.0 para reducir el ruido
        # Crear un nuevo modelo de ruido escalado
        scaled_noise = NoiseModel()

        # Copiar y escalar errores de compuertas
        for instr, qubits in original_noise._local_quantum_errors.items():
            for q, err in qubits.items():
                prob = err.to_dict()['probabilities'][0]  # Supone un solo error tipo
                new_prob = min(prob * noise_scale, 1.0)  # Asegúrate de no pasar de 1
                gate_name = instr
                qubit = q

                # Crear nuevo error con misma estructura
                if len(qubit) == 1:
                    new_error = depolarizing_error(new_prob, 1)
                elif len(qubit) == 2:
                    new_error = depolarizing_error(new_prob, 2)
                else:
                    continue  # No soportamos otros tamaños aquí

                scaled_noise.add_quantum_error(new_error, gate_name, qubit)

        # Ahora usas scaled_noise en el simulador
        noisy_simulator = AerSimulator(noise_model=scaled_noise)

        # Creo el estimator para el circuito ruidoso
        estimator = BackendEstimator(backend=noisy_simulator)
    
    # Si no escalo el ruido, uso el fakeBackend directamente
    else:
        estimator = BackendEstimator(fake_backend)
    
    for layers in range(minLayers, maxLayers+1):

        current_observable = cf.expand_observable(base_observable, n_qubits)
        ansatz_circuit, num_params = ansantz_function(n_qubits, layers)

        if print_info:
            print("\n=====================================================")
            print(f"Calculando varianzas con nº capas: {layers}.\n")
        
        var_value, var_deriv = cf.get_variances_data(num_params, ansatz_circuit, current_observable, estimator, index, num_shots, print_progress=print_progress, use_shift_rule=use_shift_rule, delta=delta)

        # Current iteration information
        if print_info:
            print(f"Varianza del valor esperado: {var_value}")
            print(f"Varianza de la derivada: {var_deriv}")

        data["layers"].append(layers)
        data["var_value"].append(var_value)
        data["var_deriv"].append(var_deriv)
    

    # Regression
    if do_regress:
        value_regress = linregress(data["layers"], np.log(data["var_value"]))
        deriv_regress = linregress(data["layers"], np.log(data["var_deriv"]))

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
        ax.scatter(data["layers"], data["var_value"], label=r"Var($\langle O\rangle$)")
        ax.scatter(data["layers"], data["var_deriv"], label=rf"Var($\partial_{index}\langle O\rangle$)")

        # Tendencies
        if do_regress:
            base = np.linspace(minLayers, maxLayers, 100)
            ax.plot(base, np.exp(data["value_slope"]*base+data["value_ord"]), color="black", label=r"Tendencia: Var($\langle O\rangle$)")
            ax.plot(base, np.exp(data["deriv_slope"]*base+data["deriv_ord"]), color="red", label=rf"Tendencia: Var($\partial_{index}\langle O\rangle$)")

        # Settings
        ax.set_xlabel("N capas")
        ax.set_title(rf"BP en VQE, variando el parámetro $\theta_{index}$")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    return data
