import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

import pickle

def hamiltonians(geometry,dist):
    # Configure PySCF driver
    driver = PySCFDriver(atom=geometry, basis='sto3g')

    # Execute driver to obtain electronic structure's problem
    es_problem = driver.run()

    # Build the Hamiltonian after second cuantization
    hamiltonian = es_problem.second_q_ops()[0]

    # Apply Jordan-Wigner transformations
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian)

    with open(f"notebooks/VQE/H2/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Nuclear repulsion
    nuclear_repulsion = es_problem.nuclear_repulsion_energy

    with open(f"notebooks/VQE/H2/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

# For different distances
distances = np.linspace(0.25, 4, 25)
for dist in distances:
    # Molecule geometry
    geometry = f"H 0.0 0.0 {-dist/2}; H 0.0 0.0 {dist/2}"
    hamiltonians(geometry,dist)