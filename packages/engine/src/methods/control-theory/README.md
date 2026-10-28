# Control theory (part 2, signals, systems and information)

Systems controlled and estimated: state-space and transfer-function models, stability, PID and frequency-domain design, pole placement and observers, LQR and LQG, model predictive control, system identification and robust control, as sessions over `systems`, `dynamics` and `optim`.

## Capabilities

- Modelling: named plants (mass-spring-damper, pendulum, cart-pole, tanks) in state space and as transfer functions, linearised around a point; options: the plant and its parameters.
- Stability: eigenvalues and poles moving as a gain changes, root locus, Nyquist and Bode.
- Classical control: PID tuned live, the step response and its measures; lead-lag compensation.
- State-space control: controllability and observability, pole placement, a Luenberger observer tracking the state, the Kalman filter as the optimal observer (`dynamics/kalmanTracking`).
- Optimal control: LQR by the Riccati iteration, LQG, dynamic programming on a grid, Pontryagin on a small problem.
- Model predictive control: the receding horizon solved each step, constraints active.
- System identification: ARX by least squares, subspace methods, as data arrive.

## Examples to make

- PID gains as knobs with the step response and the control effort.
- Root locus as the gain sweeps, with the closed-loop poles crossing the axis.
- The Kalman filter following a point it sees through noise (exists), with the observer gain as the knob.
- MPC steering the cart-pole with the horizon drawn ahead of the cart each step.
- LQR's Riccati iteration converging, the gain and the cost.

## Rendering

- Curves, pole-zero and Nyquist plots, episodes of an environment under a controller (`EpisodePlayer` with `environments`), a horizon overlay.
- Missing: the complex plane as a panel kind (poles moving at the viewed step).

## Shared

- `shared/`: the plant slot (shared with `reinforcement-learning` environments: cart-pole is both), the controller slot (PID, state feedback, MPC, a policy).

## Problems

- New chapter; `dynamics/kalmanTracking` moves here or to `time-series/state-space-models` (decide: here, as an observer).
