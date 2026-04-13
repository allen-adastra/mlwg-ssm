
## Project Description
This is a repository containing code and notebooks for a tutorial introducing concepts like state, action, and observations. The tutorial is a one hour session for the Machine Learning Working Group at MIT PSFC. This includes classical state-space models, POMDPs, and eventually action-conditioned world models.

The advertised session is:

Tutorial Title: "Making Real Machines Work: Modelling with States, Actions and Observations"

Abstract: It is important for us to get better at operating and controlling our tokamaks to not break SPARC and make everyone sad. Modelling for control and operations requires a somewhat different perspective: one focused on understanding how actions taken (e.g. voltages sent to power supplies) in the present map to future observations (e.g. voltages observed on magnetic probes). Fortunately, highly overlapping modelling paradigms have been developed by several communities to address the relationship between actions, system state, and observations. This tutorial will review classical perspectives developed by the control systems and reinforcement learning (RL) communities and touch upon recent developments such as action-conditioned world models with some hands on examples.

## Project Structure

```
mlwg-state-space/
├── CLAUDE.md               # This file
└── notebooks/                # Jupyter notebooks for students.
    ├── intro_to_state_space_models.ipynb               # An introduction to state-space models using the cartpole as an example.

└── src/                     # Source code for the project.
    ├── system.py               # Toy dynamical system definition
    └── systems/                     # implementations of various dynamical systems.
        ├── cartpole.py               # Cartpole system definition

```


## Coding Conventions

- Use uv for dependency management and virtual environments
- Python 3.10+
- Use JAX for modelling and simulation.
- Prefer matplotlib for plotting.
- Reproducible: always set random seeds
