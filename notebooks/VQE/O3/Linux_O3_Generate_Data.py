import numpy as np
import pickle

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


import numpy as np
import pickle

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


def hamiltonians(geometry, dist):
    # Configure PySCF driver
    driver = PySCFDriver(atom=geometry, basis='sto3g')

    # Execute driver to obtain electronic structure's problem
    es_problem = driver.run()

    # Build the Hamiltonian after second cuantization
    hamiltonian = es_problem.second_q_ops()[0]

    # Apply Jordan-Wigner transformations
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian)

    with open(f"notebooks/VQE/O3/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Nuclear repulsion
    nuclear_repulsion = es_problem.nuclear_repulsion_energy

    with open(f"notebooks/VQE/O3/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

# For different distances
distances = np.linspace(0.25, 4, 16)
sen = np.sin(116.8/2 *360/2/np.pi)
cos = np.cos(116.8/2 *360/2/np.pi)
for dist in distances:
    # Molecule geometry
    geometry = f"O 0.000000 0.000000 0.000000; O {sen*dist} 0.000000 {cos*dist}; O -1.205 0.000000 0.587"
    hamiltonians(geometry,dist)