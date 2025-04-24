from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


# Hydrogen geometry

# Definimos la geometría molecular
geometry = "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"

# Configuramos el driver PySCF
driver = PySCFDriver(atom=geometry, basis='sto3g')

# Ejecutamos el driver para obtener el problema de la estructura electrónica
es_problem = driver.run()

# Construimos el Hamiltoniano después de la segunda cuantización
hamiltonian = es_problem.second_q_ops()[0]

# Aplicamos las transformaciones de Jordan-Wigner
mapper = JordanWignerMapper()
hamiltonian = mapper.map(hamiltonian)

print(hamiltonian)