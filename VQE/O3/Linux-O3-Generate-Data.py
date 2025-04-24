import numpy as np
import pickle

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


# Geometría de la molecula
geometry = "O 0.000000 0.000000 0.000000; O 1.205 0.000000 0.587; O -1.205 0.000000 0.587"

# Configuramos el driver PySCF
driver = PySCFDriver(atom=geometry, basis='sto3g')

# Ejecutamos el driver para obtener el problema de la estructura electrónica
es_problem = driver.run()


# ============================= Hamiltoniano =================================

# Construimos el Hamiltoniano después de la segunda cuantización
hamiltonian = es_problem.second_q_ops()[0]

# Aplicamos las transformaciones de Jordan-Wigner
mapper = JordanWignerMapper()
hamiltonian = mapper.map(hamiltonian)

# Mostramos el Hamiltoniano
print(f"Hamiltoniano para geometría: {geometry} \n{hamiltonian}")

# Guardamos en archivo
with open(f"VQE/O3/data/hamiltonian.pkl", "wb") as f:
    pickle.dump(hamiltonian, f)


# ============================= Repulsión nuclear =================================

nuclear_repulsion = es_problem.nuclear_repulsion_energy

# Guardamos en archivo
print(f"Energía nuclear: {nuclear_repulsion}")
with open(f"VQE/O3/data/nuclear_repulsion.pkl", "wb") as f:
    pickle.dump(nuclear_repulsion, f)


# ============================= Ansatz =================================

#Definir parámetros
num_spatial_orbitals = es_problem.num_spin_orbitals // 2  # 14 // 2 = 7
num_particles = es_problem.num_particles  # (5, 5)

#Crear el estado de Hartree-Fock
hf_initial_state = HartreeFock(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper
)

#Crear el ansatz UCCSD
ansatz = UCCSD(
    num_spatial_orbitals=num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=hf_initial_state
)

# Guardamos en archivo
with open("VQE/O3/data/ansatz.pkl", "wb") as f:
    pickle.dump(ansatz, f)