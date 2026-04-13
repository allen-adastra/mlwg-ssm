"""Base protocol for dynamical systems used throughout the tutorial.

A system is defined by three functions:
  - dynamics:  (state, action, params) -> state_dot   (continuous-time ODE RHS)
  - step:      (state, action, params, dt) -> next_state  (discrete-time transition)
  - observe:   (state, params) -> observation

All functions must be JAX-compatible (jit/vmap/grad safe).
"""

from typing import Protocol, Any
from jaxtyping import Array, Float


State = Float[Array, "state_dim"]
Action = Float[Array, "action_dim"]
Observation = Float[Array, "obs_dim"]


class DynamicalSystem(Protocol):
    """Protocol that all system implementations should satisfy."""

    def dynamics(self, state: State, action: Action, params: Any) -> State:
        """Continuous-time dynamics: returns dx/dt."""
        ...

    def step(self, state: State, action: Action, params: Any, dt: float) -> State:
        """Discrete-time step via numerical integration."""
        ...

    def observe(self, state: State, params: Any) -> Observation:
        """Observation function: maps state to observable quantities."""
        ...
