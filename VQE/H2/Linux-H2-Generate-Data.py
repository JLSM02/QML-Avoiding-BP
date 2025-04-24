import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

import pickle


geometry =  f"H 0.0 0.0 0.725/2; H 0.0 0.0 -0.725/2"


# Configuramos el driver PySCF
driver = PySCFDriver(atom=geometry, basis='sto3g')

# Ejecutamos el driver para obtener el problema de la estructura electrónica
es_problem = driver.run()

# Construimos el Hamiltoniano después de la segunda cuantización
hamiltonian = es_problem.second_q_ops()[0]

# Aplicamos las transformaciones de Jordan-Wigner
mapper = JordanWignerMapper()

hamiltonian = mapper.map(hamiltonian)


print(f"Hamiltoniano: \n{hamiltonian}")
with open(f"VQE/H2/data/hamiltonian.pkl", "wb") as f:
    pickle.dump(hamiltonian, f)


nuclear_repulsion = es_problem.nuclear_repulsion_energy


print(f"Energía nuclear: {nuclear_repulsion}")
with open(f"VQE/H2/data/nuclear_repulsion.pkl", "wb") as f:
    pickle.dump(nuclear_repulsion, f)