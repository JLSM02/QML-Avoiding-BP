import numpy as np
import pickle

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


def hamiltonians(geometry, dist):
    # Configuramos el driver PySCF
    driver = PySCFDriver(atom=geometry, basis='sto3g')

    # Ejecutamos el driver para obtener el problema de la estructura electrónica
    es_problem = driver.run()

    # Construimos el Hamiltoniano después de la segunda cuantización
    hamiltonian = es_problem.second_q_ops()[0]

    # Aplicamos las transformaciones de Jordan-Wigner
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian)

    with open(f"notebooks/VQE/H2O/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Repulsión nuclear
    nuclear_repulsion = es_problem.nuclear_repulsion_energy

    with open(f"notebooks/VQE/H2O/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

sen = np.sin(104.5/2 *360/2/np.pi)
cos = np.cos(104.5/2 *360/2/np.pi)

distances = np.linspace(0.25, 4, 25)
for dist in distances:
    # Geometría de la molecula
    
    geometry = f"O 0.0 0.0 0.0; H {sen*dist} {cos*dist} 0.0; H -0.757 0.586 0.0"
    hamiltonians(geometry,dist)