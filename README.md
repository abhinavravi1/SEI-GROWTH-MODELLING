SEI Growth and Irreversible Capacity Loss Modeling in Lithium-Ion Batteries

This repository contains a Python-based simulation framework for analyzing Solid Electrolyte Interphase (SEI) growth and irreversible capacity loss in lithium-ion batteries using quantum tunneling models, fracture mechanics, and stress-assisted crack growth.

The code models:

Inner SEI layer tunneling (Q<sub>II</sub>)
Crack propagation and associated degradation (Q<sub>III</sub>)
Crack-induced SEI formation (Q<sub>IV</sub>)
Analytical and numerical integration of SEI-related capacity loss over charge–discharge cycles
📘 Features

Analytical and numerical modeling of Q<sub>II</sub>, Q<sub>III</sub>, and Q<sub>IV</sub>
Parametric modeling of:
Stress amplitude
Electrochemical tunneling
SEI thickness growth
Visualization of capacity loss per mechanism and total capacity degradation
Least-squares estimation for model parameter fitting
Support for reinforcement learning-based model refinement (optional extension)
📁 Files

sei_model.py – Core script for SEI growth modeling, numerical solving, and plotting.
README.md – Description of the model, setup, and usage.
requirements.txt – List of required Python packages (e.g., numpy, scipy, matplotlib).
🧮 Mathematical Background

This model is based on coupled electrochemical–mechanical formulations, including:

Tunneling current estimation from quantum mechanics
Stress-driven fracture propagation (Paris' Law)
Multilayer SEI growth kinetics
Capacity loss due to both transport and mechanical degradation
📊 Output

Analytical vs numerical comparisons for each component (Q<sub>II</sub>, Q<sub>III</sub>, Q<sub>IV</sub>)
Total capacity loss curve across N cycles
SEI thickness growth rate over time
