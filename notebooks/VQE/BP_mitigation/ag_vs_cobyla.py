# Imports
import time
import numpy as np
from qiskit.quantum_info import SparsePauliOp
import sys
sys.path.append('../../../')
from src import customFunc as cf
from src import ansatzs as anz
from src import mitigation as mit

# Base observable
Z1Z2_obs = SparsePauliOp.from_list([("ZZ", 1)])


# Expansion to more qubits
for n in range(2, 10):
	expanded_obs = cf.expand_observable(Z1Z2_obs, n)

	print(f"Begining experiments with {n} qubits:\n")


	###################### Exp AG low pop #####################


	# Numero de experimentos (optimizaciones) a realizar
	n_experiments = 1000

	# Población a usar en este experimento
	pop = 5

	# Guardare los datos de los experimentos
	evaluations_exp = np.zeros(n_experiments)
	generations_exp = np.zeros(n_experiments)
	time_exp = np.zeros(n_experiments)
	minimums_exp = np.zeros(n_experiments)

	print(f"AG with pop={pop}:\n")

	for iter in range(n_experiments):

		t0 = time.time()
		data = mit.VQE_minimization_AG(anz.build_deep_ansatz(n,2)[0], expanded_obs, 0.0001, crossover_prob=0.8, mutation_prob=0.6, population_size=pop, max_gen=1000, print_info=False, plot_info=False)
		t1 = time.time()

		generations_exp[iter] = data["n_generations"]
		evaluations_exp[iter] = data["n_evaluations"]
		time_exp[iter] = t1-t0
		minimums_exp[iter] = data["minimum_value"]

		print(f"Experiment {iter} terminated with {evaluations_exp[iter]} evaluations.")

	# Junto todos los datos en un array y los guardo en txt
	array_to_save = np.column_stack((generations_exp, evaluations_exp, time_exp, minimums_exp))
	np.savetxt(f"data/data_GA_{n}qubits_{pop}.csv", array_to_save, fmt="%f", delimiter=",")


	###################### Exp AG high pop #####################


	# Numero de experimentos (optimizaciones) a realizar
	n_experiments = 1000

	# Población a usar en este experimento
	pop = 10

	# Guardare los datos de los experimentos
	evaluations_exp = np.zeros(n_experiments)
	generations_exp = np.zeros(n_experiments)
	time_exp = np.zeros(n_experiments)
	minimums_exp = np.zeros(n_experiments)

	print(f"AG with pop={pop}:\n")

	for iter in range(n_experiments):

		t0 = time.time()
		data = mit.VQE_minimization_AG(anz.build_deep_ansatz(n,2)[0], expanded_obs, 0.0001, crossover_prob=0.8, mutation_prob=0.6, population_size=pop, max_gen=1000, print_info=False, plot_info=False)
		t1 = time.time()

		generations_exp[iter] = data["n_generations"]
		evaluations_exp[iter] = data["n_evaluations"]
		time_exp[iter] = t1-t0
		minimums_exp[iter] = data["minimum_value"]

		print(f"Experiment {iter} terminated with {evaluations_exp[iter]} evaluations.")

	# Junto todos los datos en un array y los guardo en txt
	array_to_save = np.column_stack((generations_exp, evaluations_exp, time_exp, minimums_exp))
	np.savetxt(f"data/data_GA_{n}qubits_{pop}.csv", array_to_save, fmt="%f", delimiter=",")


	###################### Exp COBYLA #####################

	# Numero de experimentos (optimizaciones) a realizar
	n_experiments = 1000

	# Guardare los datos de los experimentos
	generations_exp = np.zeros(n_experiments)
	evaluations_exp = np.zeros(n_experiments)
	time_exp = np.zeros(n_experiments)
	minimums_exp = np.zeros(n_experiments)

	print(f"COBYLA:\n")

	for iter in range(n_experiments):

		t0 = time.time()

		data = mit.VQE_minimization(anz.build_deep_ansatz(n, 2)[0], expanded_obs, initial_guess="rand", tol=0.0001)

		t1 = time.time()
		evaluations_exp[iter] = data["iters"]
		time_exp[iter] = t1-t0
		minimums_exp[iter] = data["cost_history"][-1]

		print(f"Experiment {iter} terminated with {evaluations_exp[iter]} evaluations.")

	# Junto todos los datos en un array y los guardo en txt
	array_to_save = np.column_stack((generations_exp, evaluations_exp, time_exp, minimums_exp))
	np.savetxt(f"data/data_COBYLA_{n}qubits.csv", array_to_save, fmt="%f", delimiter=",")
