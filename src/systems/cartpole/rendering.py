"""Matplotlib renderer for the cartpole system.

Visual style inspired by the Gymnasium CartPole-v1 renderer:
  - cart: dark-grey filled rectangle on a track
  - pole: a thin rod pivoting from the cart centre
  - axle: small filled circle at the pivot
  - track: horizontal line with left/right limit markers

Coordinate convention (matches dynamics.py):
  theta = 0   →  pole pointing straight up
  theta > 0   →  pole tilted counter-clockwise (to the left)
"""

import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from .dynamics import CartPoleParams

# ── Trajectory plot constants ──────────────────────────────────────────────────
_STATE_LABELS = [
    (r"$x$ (m)", "Cart position"),
    (r"$\dot{x}$ (m/s)", "Cart velocity"),
    (r"$\theta$ (rad)", "Pole angle"),
    (r"$\dot{\theta}$ (rad/s)", "Pole angular velocity"),
]
_ACTION_LABELS = [
    (r"$F$ (N)", "Applied force"),
]
_BLOCK_BG_STATES = "#F0F4FA"
_BLOCK_BG_ACTIONS = "#F5F0FA"
_OBS_COLOR = "#E07030"  # orange — noisy observations

# ── Visual geometry constants (metres) ────────────────────────────────────────
_CART_WIDTH = 0.5
_CART_HEIGHT = 0.3
_POLE_WIDTH = 0.06
_AXLE_RADIUS = 0.06
_TRACK_Y = 0.0
_CART_Y = _TRACK_Y + _CART_HEIGHT / 2  # centre of cart above track

# Colours matching the Gymnasium palette
_CART_COLOR = "#404040"
_POLE_COLOR = "#8B4513"  # saddlebrown
_AXLE_COLOR = "#A9A9A9"  # darkgrey
_TRACK_COLOR = "#000000"
_WHEEL_COLOR = "#303030"
_BG_COLOR = "#FFFFFF"

_WHEEL_RADIUS = 0.07
_WHEEL_OFFSETS = [-_CART_WIDTH / 4, _CART_WIDTH / 4]


def _y_max(params: CartPoleParams) -> float:
    """Top of the visible area, with room for labels above the pole tip."""
    return 2 * params.pole_len + _CART_HEIGHT + 0.7


def _draw_cartpole(
    ax: plt.Axes, state, params: CartPoleParams, x_range: float = 3.0
) -> None:
    """Draw a single cartpole frame onto *ax* (clears existing artists first)."""
    ax.cla()

    x_cart = float(state[0])
    theta = float(state[2])
    pole_len = params.pole_len  # half-length; full visual pole = 2 * pole_len

    # Re-apply limits after cla()
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-0.6, _y_max(params))
    ax.set_facecolor(_BG_COLOR)

    # ── Track ─────────────────────────────────────────────────────────────────
    ax.axhline(_TRACK_Y, color=_TRACK_COLOR, linewidth=2, zorder=1)
    for xm in (-x_range + 0.05, x_range - 0.05):
        ax.axvline(
            xm, color=_TRACK_COLOR, linewidth=1.5, linestyle="--", alpha=0.4, zorder=1
        )

    # ── Cart ──────────────────────────────────────────────────────────────────
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x_cart - _CART_WIDTH / 2, _TRACK_Y),
            _CART_WIDTH,
            _CART_HEIGHT,
            boxstyle="round,pad=0.02",
            linewidth=1,
            edgecolor="black",
            facecolor=_CART_COLOR,
            zorder=3,
        )
    )

    # ── Wheels ────────────────────────────────────────────────────────────────
    for dx in _WHEEL_OFFSETS:
        ax.add_patch(
            mpatches.Circle(
                (x_cart + dx, _TRACK_Y), _WHEEL_RADIUS, color=_WHEEL_COLOR, zorder=4
            )
        )

    # ── Pole ──────────────────────────────────────────────────────────────────
    pivot_x, pivot_y = x_cart, _CART_Y
    # theta=0 → pole points up; positive theta → tilts left (counter-clockwise)
    tip_x = pivot_x + 2 * pole_len * jnp.sin(theta)
    tip_y = pivot_y + 2 * pole_len * jnp.cos(theta)
    ax.add_line(
        Line2D(
            [pivot_x, float(tip_x)],
            [pivot_y, float(tip_y)],
            linewidth=_POLE_WIDTH * 100,
            color=_POLE_COLOR,
            solid_capstyle="round",
            zorder=5,
        )
    )

    # ── Axle ──────────────────────────────────────────────────────────────────
    ax.add_patch(
        mpatches.Circle((pivot_x, pivot_y), _AXLE_RADIUS, color=_AXLE_COLOR, zorder=6)
    )

    # ── Physics labels ────────────────────────────────────────────────────────
    # Pole unit direction and its rightward perpendicular
    sin_th, cos_th = float(jnp.sin(theta)), float(jnp.cos(theta))
    perp_x, perp_y = cos_th, -sin_th  # 90° clockwise from pole direction

    label_offset = 0.18  # metres, perpendicular offset for labels

    # m_c — centred on the cart body
    ax.text(
        x_cart,
        _CART_Y,
        r"$m_c$",
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=10,
    )

    # m_p — circle at pole tip + callout label (always offset to upper-right)
    ax.add_patch(
        mpatches.Circle((float(tip_x), float(tip_y)), 0.07, color=_POLE_COLOR, zorder=7)
    )
    ax.annotate(
        r"$m_p$",
        xy=(float(tip_x), float(tip_y)),
        xytext=(float(tip_x) + 0.35, float(tip_y) + 0.25),
        fontsize=11,
        fontweight="bold",
        color=_POLE_COLOR,
        ha="center",
        va="center",
        zorder=10,
        arrowprops=dict(arrowstyle="-", color=_POLE_COLOR, lw=1.0),
    )

    # l — dimension line from pivot to pole midpoint, offset to the side
    mid_x = pivot_x + pole_len * sin_th
    mid_y = pivot_y + pole_len * cos_th
    dim_offset = label_offset * 0.9
    ax.annotate(
        "",
        xy=(mid_x, mid_y),
        xytext=(pivot_x, pivot_y),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        zorder=7,
    )
    ax.text(
        mid_x + dim_offset * perp_x,
        mid_y + dim_offset * perp_y,
        r"$l$",
        fontsize=11,
        fontweight="bold",
        color="red",
        ha="center",
        va="center",
        zorder=10,
    )

    # ── State annotation ──────────────────────────────────────────────────────
    state_str = rf"$x={float(state[0]):+.2f}$ m    $\theta={jnp.rad2deg(theta):+.1f}°$"
    params_str = rf"$m_c={params.m_c:.2f}$ kg    $m_p={params.m_p:.3f}$ kg    $l={params.pole_len:.2f}$ m"
    ax.text(
        0.02,
        0.97,
        state_str,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    ax.text(
        0.02,
        0.87,
        params_str,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        color="#555555",
    )


def render_frame(
    state,
    params: CartPoleParams,
    ax: plt.Axes | None = None,
    x_range: float = 2.0,
) -> plt.Axes:
    """Render a single cartpole state.

    Args:
        state:   array [x, x_dot, theta, theta_dot]
        params:  CartPoleParams
        ax:      matplotlib Axes to draw on (creates a new figure if None)
        x_range: half-width of the track shown (metres)

    Returns:
        The Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.subplots_adjust(bottom=0.18)

    _draw_cartpole(ax, state, params, x_range=x_range)
    ax.set_aspect("equal")
    ax.set_xlabel("cart position (m)")
    ax.set_yticks([])
    return ax


def animate(
    trajectory,
    params: CartPoleParams,
    dt: float = 0.02,
    x_range: float = 2.0,
    figsize: tuple[float, float] = (8, 4),
    playback_speed: float = 1.0,
) -> FuncAnimation:
    """Animate a cartpole trajectory.

    Args:
        trajectory:      array of shape (T, 4) from CartPole.rollout()
        params:          CartPoleParams
        dt:              simulation timestep (s) — sets real-time playback rate
        x_range:         half-width of the track shown (metres)
        figsize:         figure size in inches
        playback_speed:  >1 speeds up, <1 slows down

    Returns:
        matplotlib FuncAnimation — call plt.show() or display in a notebook.

    Example::

        anim = animate(trajectory, params)
        from IPython.display import HTML
        HTML(anim.to_jshtml())
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.18)
    ax.set_aspect("equal")
    ax.set_xlabel("cart position (m)")
    ax.set_yticks([])

    interval_ms = (dt * 1000) / playback_speed

    def update(frame: int):
        _draw_cartpole(ax, trajectory[frame], params, x_range=x_range)
        ax.set_aspect("equal")
        ax.set_xlabel("cart position (m)")
        ax.set_yticks([])
        return (ax,)

    return FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=interval_ms,
        blit=False,
    )


def plot_trajectory(
    trajectory,
    actions,
    params: CartPoleParams,
    dt: float = 0.02,
    figsize: tuple[float, float] = (11, 7),
    observations=None,
) -> plt.Figure:
    """Plot state and action trajectories in two visually distinct blocks.

    Args:
        trajectory:   array of shape (T+1, 4) from CartPole.rollout()
        actions:      array of shape (T, 1), one action per timestep
        params:       CartPoleParams (used for axis labels only)
        dt:           simulation timestep (s)
        figsize:      figure size in inches
        observations: optional array of shape (T+1, 4) from rollout_with_obs();
                      plotted as orange scatter dots on top of the true-state lines
                      to illustrate sensor noise

    Returns:
        The matplotlib Figure.

    Example::

        fig = plot_trajectory(trajectory, actions, params, dt=0.02)
        plt.show()

        # With noisy observations:
        states, obs = system.rollout_with_obs(s0, actions, params, key)
        fig = plot_trajectory(states, actions, params, observations=obs)
    """
    T = len(actions)
    t_states = jnp.linspace(0.0, T * dt, T + 1)
    t_actions = t_states[:-1]  # actions are applied at each step start

    fig = plt.figure(figsize=figsize, layout="constrained")

    # Two vertically stacked subfigures — states on top, actions on bottom
    subfigs = fig.subfigures(2, 1, height_ratios=[3, 1], hspace=0.08)

    # ── States block ──────────────────────────────────────────────────────────
    subfigs[0].set_facecolor(_BLOCK_BG_STATES)

    title_suffix = " + Observations" if observations is not None else ""
    subfigs[0].suptitle(
        f"States{title_suffix}",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        color="#185FA5",
    )

    state_axes = subfigs[0].subplots(2, 2, sharex=True)
    for i, (ylabel, title) in enumerate(_STATE_LABELS):
        ax = state_axes.flat[i]
        ax.plot(
            t_states,
            trajectory[:, i],
            color="#185FA5",
            linewidth=1.4,
            label="True state",
            zorder=3,
        )
        if observations is not None:
            ax.scatter(
                t_states,
                observations[:, i],
                color=_OBS_COLOR,
                s=6,
                alpha=0.6,
                label="Observation",
                zorder=4,
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(_BLOCK_BG_STATES)
        if i >= 2:  # bottom row
            ax.set_xlabel("Time (s)", fontsize=9)

    if observations is not None:
        # Single shared legend at the top-right of the states block
        state_axes.flat[1].legend(
            fontsize=8,
            framealpha=0.8,
            loc="upper right",
            markerscale=2,
        )

    # ── Actions block ─────────────────────────────────────────────────────────
    subfigs[1].set_facecolor(_BLOCK_BG_ACTIONS)
    subfigs[1].suptitle(
        "Actions",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        color="#6B3FA0",
    )

    n_actions = actions.shape[1]
    action_axes = subfigs[1].subplots(1, n_actions, sharey=False)
    if n_actions == 1:
        action_axes = [action_axes]

    for j, ax in enumerate(action_axes):
        ylabel, title = (
            _ACTION_LABELS[j]
            if j < len(_ACTION_LABELS)
            else (f"$a_{j}$", f"Action {j}")
        )
        ax.step(t_actions, actions[:, j], where="post", color="#6B3FA0", linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(_BLOCK_BG_ACTIONS)

    return fig


def animate_trajectory(
    trajectory,
    actions,
    params: CartPoleParams,
    dt: float = 0.02,
    figsize: tuple[float, float] = (16, 7),
    playback_speed: float = 1.0,
    x_range: float = 2.0,
    observations=None,
) -> FuncAnimation:
    """Animate a cartpole trajectory with synchronised state/action time series.

    The figure has three visual sections:
      - Left:         cartpole animation (physics view)
      - Right-top:    States block — 2×2 grid of state variables
      - Right-bottom: Actions block — one subplot per action channel

    A red vertical line in every time-series plot tracks the current frame.
    If *observations* is provided, noisy observation dots are drawn on the
    state plots in orange alongside the true-state lines.

    Args:
        trajectory:     array of shape (T+1, 4) from CartPole.rollout()
        actions:        array of shape (T, 1), one action per timestep
        params:         CartPoleParams
        dt:             simulation timestep (s)
        figsize:        figure size in inches
        playback_speed: >1 speeds up, <1 slows down
        x_range:        half-width of the cartpole track (metres)
        observations:   optional array of shape (T+1, 4) from rollout_with_obs()

    Returns:
        matplotlib FuncAnimation — display with ``HTML(anim.to_jshtml())``
        in a notebook or call ``plt.show()`` in a script.

    Example::

        from IPython.display import HTML
        anim = animate_trajectory(trajectory, actions, params, dt=0.02)
        HTML(anim.to_jshtml())
    """
    T = len(actions)
    t_states = jnp.linspace(0.0, T * dt, T + 1)
    t_actions = t_states[:-1]

    fig = plt.figure(figsize=figsize, facecolor="white")

    # ── Layout: 3 rows × 3 cols ────────────────────────────────────────────────
    # Col 0 (cartpole) is wider; rows 0-1 are state plots; row 2 is actions.
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.4, 1, 1],
        height_ratios=[1, 1, 0.6],
        hspace=0.55,
        wspace=0.42,
        left=0.05,
        right=0.97,
        top=0.91,
        bottom=0.09,
    )

    ax_cart = fig.add_subplot(gs[:, 0])
    ax_cart.set_aspect("equal")
    ax_cart.set_xlabel("cart position (m)")
    ax_cart.set_yticks([])

    state_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    ax_action = fig.add_subplot(gs[2, 1:])

    # ── Block header labels (placed via figure coordinates) ───────────────────
    states_x = state_axes[0].get_position().x0
    action_y = ax_action.get_position().y1 + 0.012

    fig.text(
        states_x,
        0.95,
        "States",
        fontsize=11,
        fontweight="bold",
        color="#185FA5",
        va="bottom",
        ha="left",
    )
    fig.text(
        states_x,
        action_y,
        "Actions",
        fontsize=11,
        fontweight="bold",
        color="#6B3FA0",
        va="bottom",
        ha="left",
    )

    # ── State time-series plots ────────────────────────────────────────────────
    vlines: list = []
    for i, (ylabel, title) in enumerate(_STATE_LABELS):
        ax = state_axes[i]
        ax.plot(
            t_states,
            trajectory[:, i],
            color="#185FA5",
            linewidth=1.4,
            label="True state",
            zorder=3,
        )
        if observations is not None:
            ax.scatter(
                t_states,
                observations[:, i],
                color=_OBS_COLOR,
                s=5,
                alpha=0.55,
                label="Observation",
                zorder=4,
            )
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(_BLOCK_BG_STATES)
        if i >= 2:
            ax.set_xlabel("Time (s)", fontsize=8)
        vl = ax.axvline(0.0, color="#E03030", linewidth=1.2, alpha=0.85, zorder=10)
        vlines.append(vl)

    if observations is not None:
        state_axes[1].legend(
            fontsize=7, framealpha=0.8, loc="upper right", markerscale=2
        )

    # ── Action time-series plot ────────────────────────────────────────────────
    n_actions = actions.shape[1]
    for j in range(n_actions):
        ylabel_a, title_a = (
            _ACTION_LABELS[j]
            if j < len(_ACTION_LABELS)
            else (f"$a_{j}$", f"Action {j}")
        )
        ax_action.step(
            t_actions, actions[:, j], where="post", color="#6B3FA0", linewidth=1.4
        )
    ax_action.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_action.set_title(title_a, fontsize=9)
    ax_action.set_ylabel(ylabel_a, fontsize=8)
    ax_action.set_xlabel("Time (s)", fontsize=8)
    ax_action.grid(True, alpha=0.3)
    ax_action.set_facecolor(_BLOCK_BG_ACTIONS)
    vl_action = ax_action.axvline(
        0.0, color="#E03030", linewidth=1.2, alpha=0.85, zorder=10
    )
    vlines.append(vl_action)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame: int):
        t = frame * dt
        _draw_cartpole(ax_cart, trajectory[frame], params, x_range=x_range)
        ax_cart.set_aspect("equal")
        ax_cart.set_xlabel("cart position (m)")
        ax_cart.set_yticks([])
        for vl in vlines:
            vl.set_xdata([t, t])
        return [ax_cart, *vlines]

    interval_ms = (dt * 1000) / playback_speed
    return FuncAnimation(fig, update, frames=T + 1, interval=interval_ms, blit=False)
