# QML-Avoiding-BP

> Investigating barren plateaus in quantum machine learning models through gradient analysis and optimization strategies.

---

## 🧩 Project Overview

This project explores the **barren plateau phenomenon** in the training of **variational quantum circuits (VQCs)**. It provides tools for simulating and analyzing gradient variance as a function of the number of qubits using Qiskit.

---

## 📁 Project Structure
```
project/
│
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies required to run the project
├── LICENSE                      # License information for the project
│
├── src/                         # Main project module
│   ├── __init__.py              # Marks this directory as a Python package
│   ├── customFuncs.py           # Utility functions
│   └── ansatzs.py               # Ansatz building functions
│
├── notebooks/                   # Main experiments
│   ├── VQE/                     # BP in study of molecules via VQE
│   └── Z1Z2/                    # BP in Z1Z2 observable
│
└── tests/                       # Test scripts and notebooks
    └── test-customFunc.py
    └── test-ansatzs.py
```
---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/JLSM02/QML-Avoiding-BP.git
cd QML-Avoiding-BP
```
2. Create and activate a virtual environment:
```bash
conda create -n qml-env python=3.10
conda activate qml-env
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use
Run the notebooks in the notebooks/ folder to reproduce key experiments!

---

## 📚 Dependencies
* Python ≥ 3.10
* Qiskit
* NumPy
* Matplotlib
* SciPy
* . . .

All listed (and their specific version) in requirements.txt.

---

## Authors

This project was developed as part of the Master's Thesis in Quantum Computing at Universidad Internacional de La Rioja (UNIR), 2025.

- **Juan Luis Salas Montoro** – [@JLSM02](https://github.com/JLSM02)
- **Martín Ruiz Fernández2** – [@mruifer](https://github.com/mruifer)
- **Daniel Perez García** – [@danieelpg02](https://github.com/danieelpg02)

### Supervisor

- **Dr. David Pérez de Lara** – Supervisor at UNIR

---

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
