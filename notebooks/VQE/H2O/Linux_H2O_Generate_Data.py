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

    # Definir parámetros
    num_spatial_orbitals = es_problem.num_spin_orbitals // 2  
    num_particles = es_problem.num_particles  

    # Crear el estado de Hartree-Fock
    hf_initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    # Crear el ansatz UCCSD
    ansatz_H2O = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_initial_state
    ) 

    with open(f"notebooks/VQE/H2O/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Nuclear repulsion
    nuclear_repulsion = es_problem.nuclear_repulsion_energy

    with open(f"notebooks/VQE/H2O/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

    with open(f"notebooks/VQE/H2O/data/ansatz{dist:.3f}.pkl", "wb") as f:
        pickle.dump(ansatz_H2O, f)


# For different distances
distances = np.linspace(0.25, 4, 16)
sen = np.sin(104.5/2 *360/2/np.pi)
cos = np.cos(104.5/2 *360/2/np.pi)
for dist in distances:
    # Molecule geometry
    geometry = f"O 0.0 0.0 0.0; H {sen*dist} {cos*dist} 0.0; H -0.757 0.586 0.0"
    hamiltonians(geometry,dist)