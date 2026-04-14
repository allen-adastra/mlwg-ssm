"""Cartpole (inverted pendulum on a cart) dynamical system.

State vector:  [x, x_dot, theta, theta_dot]
  x         - cart position (m)
  x_dot     - cart velocity (m/s)
  theta     - pole angle from vertical, positive counter-clockwise (rad)
  theta_dot - pole angular velocity (rad/s)

Action:  [F]  - horizontal force applied to the cart (N)

Observation (default): full state [x, x_dot, theta, theta_dot]

Equations of motion derived from the Lagrangian of the cart-pole system.
See Florian (2007), "Correct equations for the dynamics of the cart-pole system".
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class CartPoleParams(NamedTuple):
    """Physical parameters of the cartpole system."""

    m_c: float = 1.0  # mass of cart (kg)
    m_p: float = 0.1  # mass of pole (kg)
    pole_len: float = 0.5  # half-length of pole (m)
    g: float = 9.81  # gravitational acceleration (m/s^2)
    b: float = 0.0  # cart friction coefficient (N·s/m)


State = Float[Array, "4"]
Action = Float[Array, "1"]


def dynamics(state: State, action: Action, params: CartPoleParams) -> State:
    """Continuous-time cartpole dynamics: returns dx/dt.

    Args:
        state:  [x, x_dot, theta, theta_dot]
        action: [F]  force applied to cart (N)
        params: CartPoleParams

    Returns:
        state_dot: [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    x, x_dot, theta, theta_dot = state
    F = action[0]

    m_c, m_p, pole_len, g, b = (
        params.m_c,
        params.m_p,
        params.pole_len,
        params.g,
        params.b,
    )
    l = pole_len  # noqa: E741 — short name matches standard physics notation
    M = m_c + m_p  # total mass

    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)

    # Denominator shared by both acceleration expressions
    denom = M - m_p * cos_th**2

    # Angular acceleration of the pole
    theta_ddot = (
        (M * g * sin_th) - cos_th * (F - b * x_dot + m_p * l * theta_dot**2 * sin_th)
    ) / (l * denom)

    # Linear acceleration of the cart
    x_ddot = (
        F - b * x_dot + m_p * l * (theta_dot**2 * sin_th - theta_ddot * cos_th)
    ) / M

    return jnp.array([x_dot, x_ddot, theta_dot, theta_ddot])


def step(
    state: State,
    action: Action,
    params: CartPoleParams,
    dt: float = 0.02,
) -> State:
    """Advance the cartpole by dt seconds using 4th-order Runge-Kutta.

    Args:
        state:  [x, x_dot, theta, theta_dot]
        action: [F]  force applied to cart (N), held constant over dt
        params: CartPoleParams
        dt:     integration timestep (s)

    Returns:
        next_state: [x, x_dot, theta, theta_dot] at t + dt
    """
    k1 = dynamics(state, action, params)
    k2 = dynamics(state + dt / 2 * k1, action, params)
    k3 = dynamics(state + dt / 2 * k2, action, params)
    k4 = dynamics(state + dt * k3, action, params)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def observe(state: State, params: CartPoleParams) -> State:
    """Observation function.

    Default: full-state observation (no hidden state).
    Returns [x, x_dot, theta, theta_dot].
    """
    return state


def rollout(
    initial_state: State,
    actions: Float[Array, "T 1"],  # noqa: F722
    params: CartPoleParams,
    dt: float = 0.02,
) -> Float[Array, "T+1 4"]:  # noqa: F722
    """Simulate a trajectory given a sequence of actions.

    Args:
        initial_state: starting state [x, x_dot, theta, theta_dot]
        actions:       array of shape (T, 1), one action per timestep
        params:        CartPoleParams
        dt:            integration timestep (s)

    Returns:
        states: array of shape (T+1, 4), including the initial state
    """

    def scan_fn(state, action):
        next_state = step(state, action, params, dt)
        return next_state, next_state

    _, subsequent_states = jax.lax.scan(scan_fn, initial_state, actions)
    return jnp.concatenate([initial_state[None], subsequent_states], axis=0)


class CartPole:
    """Stateless cartpole system.

    Wraps the module-level functions so the system satisfies the
    DynamicalSystem protocol and can be passed around as an object.

    Example::

        params = CartPoleParams()
        system = CartPole()

        state = jnp.array([0.0, 0.0, 0.1, 0.0])   # slight initial tilt
        action = jnp.array([0.0])

        next_state = system.step(state, action, params)
    """

    def dynamics(self, state: State, action: Action, params: CartPoleParams) -> State:
        return dynamics(state, action, params)

    def step(
        self,
        state: State,
        action: Action,
        params: CartPoleParams,
        dt: float = 0.02,
    ) -> State:
        return step(state, action, params, dt)

    def observe(self, state: State, params: CartPoleParams) -> State:
        return observe(state, params)

    def rollout(
        self,
        initial_state: State,
        actions: Float[Array, "T 1"],  # noqa: F722
        params: CartPoleParams,
        dt: float = 0.02,
    ) -> Float[Array, "T+1 4"]:  # noqa: F722
        return rollout(initial_state, actions, params, dt)
