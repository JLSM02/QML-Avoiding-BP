import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

import pickle

def generate_linear_chain_geometry(n_atoms, dist):
    # Genera una cadena lineal en el eje z
    positions = [f"H 0.0 0.0 {i * dist}" for i in range(n_atoms)]
    return "; ".join(positions)

def hamiltonians(n_atoms, dist):
    geometry = generate_linear_chain_geometry(n_atoms, dist)
    
    # Driver de PySCF
    driver = PySCFDriver(atom=geometry, basis='sto3g')
    es_problem = driver.run()

    # Hamiltoniano mapeado
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(es_problem.second_q_ops()[0])

    # Parámetros del sistema
    num_spatial_orbitals = es_problem.num_spin_orbitals // 2
    num_particles = es_problem.num_particles

    # Estado Hartree-Fock
    hf_initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    # Ansatz UCCSD
    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_initial_state
    )

    # Guardar archivos
    base_path = f"notebooks/VQE/H_strings (UCCSD)/data/"

    molecule_id = f"H{n_atoms}"

    with open(f"{base_path}hamiltonian_{molecule_id}_{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    with open(f"{base_path}nuclear_repulsion_{molecule_id}_{dist:.3f}.pkl", "wb") as f:
        pickle.dump(es_problem.nuclear_repulsion_energy, f)

    with open(f"{base_path}ansatz_{molecule_id}_{dist:.3f}.pkl", "wb") as f:
        pickle.dump(ansatz, f)

# Generamos los datos
dist = 0.7
for n_atoms in [2, 4, 6, 8]:
    hamiltonians(n_atoms=n_atoms, dist=dist)