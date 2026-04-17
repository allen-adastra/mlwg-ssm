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
