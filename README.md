# Homological Analysis of Dynamical Stability in Toroidal Neural ODEs

This repository contains the official implementation for the paper "Homological Analysis of Dynamical Stability in Toroidal Neural ODEs
". 
We propose a novel method to diagnose the dynamical stability of Neural ODEs trained on toroidal manifolds by applying classical theory from algebraic topology.

Our method provides a crucial tool for verifying the safety and reliability of learned dynamical models, moving beyond standard performance metrics to analyze the model's internal, qualitative properties.

## Setup

To set up the environment and install the required dependencies, please use the provided Conda environment file:

```bash
# Clone the repository
git clone [https://github.com/SHlee-TDA/homological_analysis_of_neural_ode.git](https://github.com/SHlee-TDA/homological_analysis_of_neural_ode.git)
cd Homological-Analysis-of-Toroidal-Neural-ODEs

# Create and activate the conda environment
conda env create -f environment.yml
conda activate toroidal-ode
```

## Running Experiments

All experiments can be run from the `src/` directory.

```bash
cd src
python main.py --experiment=all
```

## Experiment Checklist

This project is structured around four key experiments designed to validate our proposed methodology.

| Experiment Name                                       | Objective                                                                                                         | System Under Test                                                                    | Expected Contribution & Claim                                                                                                                                                             |
| :---------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Foundational Case Study: Stability Verification** | To verify that our method can diagnose the stability of a successfully trained model and that the result is robust. | A `ToroidalODE` trained on a 2-DOF robot arm's circular trajectory, run with multiple random seeds. | **Claim 1:** Our method can verify the hidden stability of a model, providing a deeper level of trust than performance metrics alone. The result is robust to initialization randomness.       |
| **2. Diagnostic Power: Detecting Instability** | To demonstrate that our tool can detect instability, whether from insufficient training or from learning chaotic dynamics.  | A) An undertrained model from Exp 1. <br> B) A model trained on data from a chaotic system (Anosov map). | **Claim 2:** Our method is a complete diagnostic tool that not only certifies stability but also successfully detects and flags unstable learned dynamics.                                       |
| **3. Benchmarking: Comparison with Lyapunov Exponents** | To validate our diagnoses against the state-of-the-art stability metric.                                          | The stable model from Exp 1 and the chaotic model from Exp 2B.                       | **Claim 3:** Our method's diagnoses are consistent with standard stability metrics (Lyapunov Exponents), while providing unique, complementary insights into the system's global topological structure. |
| **4. Practical Implications: Stability vs. Generalization** | To provide initial evidence for the hypothesis that our stability metric correlates with generalization performance.     | The stable model from Exp 1 and the chaotic model from Exp 2B, evaluated on an unseen test set. | **Claim 4:** The diagnosed stability is a potential proxy for good generalization. Models identified as STABLE generalize better than those identified as CHAOTIC.                           |