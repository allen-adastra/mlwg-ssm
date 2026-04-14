"""CLI commands for cartpole rendering.

Commands
--------
cartpole-frame  Render a single static frame.
cartpole-sim    Simulate with zero force and animate the result.
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .dynamics import CartPole, CartPoleParams
from .rendering import animate, render_frame

jax.config.update("jax_enable_x64", True)


def _default_params() -> CartPoleParams:
    return CartPoleParams()


def frame() -> None:
    """Render a single cartpole frame."""
    parser = argparse.ArgumentParser(description="Render a single cartpole frame.")
    parser.add_argument(
        "--theta",
        type=float,
        default=10.0,
        metavar="DEG",
        help="Initial pole angle in degrees (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save to file instead of displaying (e.g. frame.png)",
    )
    args = parser.parse_args()

    params = _default_params()
    state = jnp.array([0.0, 0.0, jnp.deg2rad(args.theta), 0.0])

    render_frame(state, params)

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


def sim() -> None:
    """Simulate the cartpole with zero force and animate."""
    parser = argparse.ArgumentParser(
        description="Simulate cartpole under zero force and animate."
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=10.0,
        metavar="DEG",
        help="Initial pole angle in degrees (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        metavar="SEC",
        help="Simulation duration in seconds (default: 4)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        metavar="SEC",
        help="Integration timestep in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save as GIF to this path (e.g. sim.gif)",
    )
    args = parser.parse_args()

    params = _default_params()
    system = CartPole()

    s0 = jnp.array([0.0, 0.0, jnp.deg2rad(args.theta), 0.0])
    T = int(args.duration / args.dt)
    actions = jnp.zeros((T, 1))
    trajectory = system.rollout(s0, actions, params, dt=args.dt)

    anim = animate(trajectory, params, dt=args.dt)

    if args.output:
        fps = int(round(1.0 / args.dt))
        anim.save(args.output, writer="pillow", fps=fps)
        print(f"Saved {args.duration}s gif ({T} frames at {fps} fps) → {args.output}")
    else:
        plt.show()
