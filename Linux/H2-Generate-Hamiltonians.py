import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

import pickle


dist = 0.725
geometry =  f"H 0.0 0.0 {dist}/2; H 0.0 0.0 -{dist}/2"



# Configuramos el driver PySCF
driver = PySCFDriver(atom=geometry, basis='sto3g')

# Ejecutamos el driver para obtener el problema de la estructura electrónica
es_problem = driver.run()

# Construimos el Hamiltoniano después de la segunda cuantización
hamiltonian = es_problem.second_q_ops()[0]

# Aplicamos las transformaciones de Jordan-Wigner
mapper = JordanWignerMapper()
hamiltonian = mapper.map(hamiltonian)

print(f"Hamiltoniano para geometría: {geometry} \n{hamiltonian}")

with open(f"../Hydrogen/hamiltonian_{dist}.pkl", "wb") as f:
    pickle.dump(hamiltonian, f)