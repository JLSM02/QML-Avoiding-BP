import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

import pickle

def hamiltonians(geometry,dist):
    # Configuramos el driver PySCF
    driver = PySCFDriver(atom=geometry, basis='sto3g')

    # Ejecutamos el driver para obtener el problema de la estructura electrónica
    es_problem = driver.run()

    # Construimos el Hamiltoniano después de la segunda cuantización
    hamiltonian = es_problem.second_q_ops()[0]

    # Aplicamos las transformaciones de Jordan-Wigner
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian)

    with open(f"VQE/H2/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Repulsión nuclear
    nuclear_repulsion = es_problem.nuclear_repulsion_energy

    with open(f"VQE/H2/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

distances = np.linspace(0.25, 4, 25)
for dist in distances:
    geometry = f"H 0.0 0.0 {-dist/2}; H 0.0 0.0 {dist/2}"
    hamiltonians(geometry,dist)