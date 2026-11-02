# Control theory (part 2, signals, systems and information)

Systems controlled and estimated: linear time-invariant models in their representations (transfer function, zero-pole-gain, state space, discretised), their time and frequency responses, stability and margins (root locus, Routh and Jury, Bode, Nyquist), classical control (PID with its tuning rules, lead-lag), state-space control (controllability and observability, pole placement, observers), optimal control (LQR by the Riccati recursion, LQG, trajectory optimisation, Pontryagin), model predictive control (linear, nonlinear, sampling), state estimation (the Kalman filter as the optimal observer, EKF and UKF), system identification (ARX, ARMAX, N4SID, DMD, SINDy) and robust control (norms, margins, model reduction), as sessions over `systems`, `dynamics` (control, filtering, identification), `environments` (the plants and their differentiable models) and `agents` (the classical controllers as agents). The Kalman tracking page exists (`methods/dynamics/kalmanTracking.ts`).

## Structure

Two kinds of page, filed under two folders of `methods/README.md` (section 3). Analysing a linear system (its poles, its step response, its Bode plot) is a numerical procedure over a model, so it lives under `simulate/dynamics/linear-systems/`, beside the named nonlinear systems of `complex-systems` and the solvers of `mathematics`. Closing a loop is an agent acting on a plant, so the controllers live under `act/control/`, the folder `reinforcement-learning` points at: the plant slot is the environment slot (cart-pole is both), the controller slot admits a trained policy, and the loop base is one sample per step.

```
simulate/dynamics/linear-systems/
  representations/   transfer function, zero-pole-gain, state space, interconnection, discretisation, the differential equation
  responses/         step, impulse, free response, lsim one sample per step, the response measures
  stability/         poles and zeros, the root locus as the gain sweeps, Routh and Jury
  frequency/         Bode, Nyquist with its encirclements, Nichols, the margins, a delay's cost
act/control/
  classical/         PID with the tuning rules and relay autotuning, anti-windup, lead-lag compensation
  state-space/       controllability and observability, pole placement, Luenberger observers, LQR and LQG
  predictive/        model predictive control (QP, iLQR, CEM, MPPI), trajectory optimisation, Pontryagin against dynamic programming
  estimation/        the Kalman filter as the optimal observer on the plant slot, EKF and UKF
  identification/    ARX and ARMAX, N4SID, DMD and Koopman, SINDy, recursive least squares, as data arrive
  robust/            H2 and H-infinity norms, the worst case over an uncertain parameter, balanced truncation, adaptive control
  shared/            the plant and controller slots, the loop base, the complex-plane record, the discretiser (below)
```

The built `methods/dynamics/kalmanTracking.ts` goes to `infer/filters/` (`probabilistic-inference` and `time-series` both claim it as the filter's page); `estimation/` here reuses its filter on the plant slot as the observer of a loop.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a response or a loop is one sample time; of a locus one gain; of a Riccati recursion one backward step; of a planner one re-plan; of an identification one arriving sample. Status is about the method alone; what the library has is in the needs column.

### simulate/dynamics/linear-systems/representations/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `representations` | One plant seen every way: transfer function, zero-pole-gain, state space (and second-order sections), the conversions between them, the minimal realisation after a pole-zero cancellation (the knob that moves a zero onto a pole), the differential equation with its TeX, the inverse Laplace transform term by term giving the response; discretisation as the mode (zero-order hold, first-order hold, Tustin, matched poles and zeros) with the sample time as the knob: the discrete poles moving in the unit circle and the sampled response against the continuous one; series, parallel and feedback interconnection of two plants. | `system/poles` t!, `system/zeros` t!, `system/matrices` t, `system/order` s, `discrete/poles` t!, `discrete/dt` s, `system/equation` p (TeX), `response/continuous` p, `response/discrete` p, `plane/poles` p | `systems/representations.ts`: `tf`, `zpk`, `ss`, `sos`, `toSs`, `toTf`, `toZpk`, `toSos`, `minreal`, `dimensions`; `systems/conversions.ts`: `tf2ss`, `ss2tf`, `tf2zpk`, `zpk2ss`; `systems/transforms.ts`: `tfToDifferential`, `tfToDifference`, `inverseLaplace`, `inverseZ`; `systems/discretise.ts`: `cont2discrete`, `c2dMatrices`; `systems/interconnect.ts`: `series`, `parallel`, `feedback`, `connect`; `systems/pairs.ts`: `laplacePairs`; all modes ready | |

### simulate/dynamics/linear-systems/responses/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `time-response` | The step, impulse and free responses, and `lsim` to any input one sample per step (the state and output on the tape; a square wave, a ramp, noise as the input slot): the second-order standard with ζ and ωn as knobs (underdamped ringing, critical, overdamped), the measures read off the response (rise time, overshoot, settling time, steady-state error) as a probe, the time grid from the slowest pole; a non-minimum-phase zero's initial undershoot. | `response/t` s, `response/y` s, `response/x` t!, `response/u` s, `response/measures` p, `response/picture` p | `systems/responses.ts`: `lsimAlgorithm`, `lsim`, `step`, `impulse`, `initial`, `defaultResponseTimes`; the measures TODO (`systems/responses.ts`: `stepInfo`) | |

### simulate/dynamics/linear-systems/stability/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `root-locus` | The closed-loop poles as the gain sweeps (one gain per step): the branches leaving the open-loop poles for the zeros and the asymptotes, the crossing of the imaginary axis (the gain at which the loop goes unstable, an event), the damping lines; the Routh array (continuous) or the Jury table (discrete) as the algebraic check of the same gain, the Routh array's sign changes counting unstable poles. | `gain/value` s, `gain/poles` t!, `gain/stable` s, `gain/crossed` e, `locus/plane` p (the branches so far with the current poles), `routh/array` p | `systems/stability.ts`: `rootLocus`, `routh`, `jury`, `poles`, `zeros`, `isStable`; `systems/interconnect.ts`: `feedback`; ready | |

### simulate/dynamics/linear-systems/frequency/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `frequency-response` | Bode magnitude and phase (the asymptotes of each pole and zero summing: one factor per step builds the plot), Nyquist with the encirclements of −1 counted and the stability decision, Nichols; the gain and phase margins with their crossover frequencies; a transport delay (the knob) eating the phase margin while the magnitude is untouched, the Padé approximation as the mode that keeps it rational. | `factor/added` m, `response/magnitude` t!, `response/phase` t!, `nyquist/curve` t!, `nyquist/encirclements` s, `margins/gain` s, `margins/phase` s, `margins/crossovers` t, `bode/picture` p, `nyquist/picture` p | `systems/frequency.ts`: `bode`, `nyquist`, `nichols`, `freqresp`, `evalfr`, `dcgain`; `systems/margins.ts`: `stabilityMargins`, `margin`, `allMargins`, `phaseCrossoverFrequencies`; the Padé delay TODO (`systems/representations.ts`: `pade`) | |

### act/control/classical/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `pid` | The gains as knobs, or set by the tuning mode (Ziegler-Nichols from the ultimate gain, Cohen-Coon from the step response, relay autotuning watched as the relay oscillation settles) on the plant slot: the closed loop one sample per step, the setpoint, the output and the error, the control effort and its three terms, the integral winding up under saturation (an event) and anti-windup fixing it, setpoint weighting and the filtered derivative, measurement noise amplified by D; the measures of the response as a probe; P alone (the offset), PI (gone), PID (faster). | `loop/setpoint` s, `loop/output` s, `loop/error` s, `loop/control` s, `pid/terms` t (P, I, D), `loop/saturated` e, `tuning/gains` t, `tuning/relay` t! (the relay oscillation), `response/measures` p | `dynamics/control/pid.ts`: `pid`, `simulatePid` (traced), `zieglerNichols`, `cohenCoon`, `relayAutotune`; the plant from `shared/plants.ts`; all modes ready | |
| `lead-lag` | A lead compensator placed at the crossover for phase margin and a lag for steady-state error (their corner frequencies as knobs): the open-loop Bode before and after, the margins moving, the step response following; the design rule as a probe (where the lead's centre must sit). | `design/margins` t, `design/crossover` s, `loop/output` s, `loop/error` s, `bode/picture` p | `systems/interconnect.ts`: `series`, `feedback`; `systems/margins.ts`: `stabilityMargins`; `systems/responses.ts`: `lsimAlgorithm`; the design rules TODO (`dynamics/control/compensators.ts`: `leadCompensator`, `lagCompensator`) | |

### act/control/state-space/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `controllability-observability` | A plant with a mode that the input cannot reach or the output cannot see (the coupling as the knob): the controllability and observability matrices and their ranks, the staircase tests, the Gramians as ellipsoids of reachable states for unit energy, the Kalman decomposition's four parts drawn as a diagram; what pole placement can and cannot move. | `analysis/rankCtrb` s, `analysis/rankObsv` s, `analysis/gramian` t!, `analysis/ellipsoid` p, `analysis/decomposition` p (a `diagram` record) | `systems/analysis.ts`: `ctrb`, `obsv`, `isControllable`, `isObservable`, `gram`, `kalmanDecomposition`; ready (a probe page over the plant slot) | |
| `pole-placement` | The closed-loop poles dragged (knobs) and the gain K recomputed by the mode (Kautsky-Nichols-Van Dooren, Ackermann for one input): the response following the poles, the gain's size growing as the poles are pushed left (the cost of speed), the plane with the open- and closed-loop poles. | `design/poles` t!, `design/gain` t!, `design/gainNorm` s, `loop/state` t!, `loop/control` s, `plane/poles` p | `dynamics/control/placement.ts`: `place`, `acker`; `systems/responses.ts`: `lsimAlgorithm`; both modes ready | |
| `observer` | A Luenberger observer tracking the state through the output, its poles placed faster than the plant's by a ratio (the knob): the estimation error decaying per sample, the separation principle (the state feedback acting on the estimate, the closed-loop poles the union of both sets), the observer's gain against the Kalman gain of `lqe` on the same plant under noise (the optimal observer); the dual problem drawn beside the controller's. | `loop/state` t!, `observer/estimate` t!, `observer/error` s, `observer/gain` t, `observer/poles` t!, `loop/control` s, `plane/poles` p | `dynamics/control/placement.ts`: `place` (on the dual pair); `dynamics/control/optimal.ts`: `lqe`, `dlqe`; `systems/responses.ts`: `lsimAlgorithm`; the observer loop in `shared/loop.ts` | |
| `lqr` | The backward Riccati recursion one step per step (finite horizon: the time-varying gains converging to the steady-state gain, the horizon as the knob), Q and R as knobs trading state error against effort (the response and the cost under each), the closed-loop poles the recursion lands on, LQG with the estimator's gain beside it; on the double integrator (the textbook benchmark), the cart-pole linearised at upright and the pendulum's swing-up (energy shaping handing over to LQR near the top, an event). | `riccati/P` t!, `riccati/K` t!, `riccati/change` s, `riccati/step` s, `loop/state` t!, `loop/control` s, `loop/cost` s, `loop/handover` e (the swing-up), `plane/poles` p, `eval/episode` p | `dynamics/control/optimal.ts`: `lqr`, `dlqr`, `riccatiRecursion`, `riccatiOde`, `lqg`, `lqe`; `environments/models.ts`: `linearize`, `cartPoleModel`, `pendulumModel`; `environments/continuous-control.ts`: `doubleIntegratorMatrices`, `doubleIntegrator`; `agents/control.ts`: `lqr`, `lqrAgent`, `swingUpAgent`; all modes ready | |

### act/control/predictive/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `mpc` | The receding horizon watched: at every step the plan over the horizon drawn ahead of the plant, only its first input applied, the plan again from the new state; the horizon length and the constraints (input limits, state bounds) as knobs, the constraints going active as events; the planner as the mode (linear MPC as a condensed quadratic programme, nonlinear MPC by warm-started iLQR, the cross-entropy method and MPPI sampling over the differentiable model); the cart-pole swing-up under an input limit that LQR cannot respect. | `loop/state` t!, `loop/control` s, `plan/horizon` t! (the planned states and inputs), `plan/cost` s, `plan/iterations` s, `plan/constraintActive` e, `plan/samples` t (CEM, MPPI), `eval/episode` p | `dynamics/control/predictive.ts`: `linearMpc`, `recedingHorizon` (traced), `nonlinearMpc`; `agents/control.ts`: `mpcAgent` (modes: iLQR, CEM, MPPI), `cemAgent`; `rl/planning.ts`: `cem`, `mppi`; `environments/models.ts`: `cartPoleModel`, `rolloutModel`, `sequenceCosts`; all modes ready | |
| `trajectory-optimisation` | Finding the whole trajectory offline: iLQR and DDP one iteration per step (the trajectory improving, the cost falling, the regularisation and line search as events) against the direct transcriptions (single shooting, multiple shooting with its defects closing, trapezoidal collocation) handed to a general optimiser as modes; the swing-up trajectory found and then tracked by `lqr`'s time-varying gains. | `iteration/cost` s, `iteration/trajectory` t!, `iteration/controls` t!, `iteration/regularisation` s, `iteration/lineSearch` e, `iteration/defects` s (multiple shooting), `trajectory/picture` p | `dynamics/control/trajectory.ts`: `ilqrAlgorithm`, `ilqr`, `ddp`, `singleShooting`, `multipleShooting`, `directCollocation`; `environments/models.ts`: `trajectoryDerivatives`; `optim/minimize` for the transcriptions; all modes ready | |
| `pontryagin` | The maximum principle on a small problem (the double integrator to rest in minimum time, a quadratic cost): the costate integrated backwards, the Hamiltonian minimised pointwise (the bang-bang switch as an event), the two-point boundary value problem solved by shooting one shot per step (the miss at the far boundary shrinking); against dynamic programming on a grid (the value function by value iteration over a discretised state space, the switching curve emerging). | `shot/initialCostate` t, `shot/miss` s, `trajectory/state` t!, `trajectory/costate` t!, `trajectory/control` s, `trajectory/switch` e, `dp/valueGrid` p, `dp/sweep` s | `dynamics/ode/boundary.ts`: `shooting`, `solveBvp`; `rl/mdp.ts`: `valueIterationAlgorithm`, `tabularMdp`; the discretiser of a `ControlModel` into a `TabularMdp` in `shared/discretise.ts`; TODO the Hamiltonian and costate setup (`dynamics/control/indirect.ts`: `hamiltonianSystem`, `bangBang`) | |

### act/control/estimation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `kalman-observer` | The Kalman filter as the observer of a loop on the plant slot: the estimate's covariance settling to the steady state (the gain converging on `lqe`'s), the innovations' whiteness as the check of the assumed noises (the knobs), LQG closing the loop on the estimate; EKF and UKF as modes on the nonlinear plant (the pendulum seen through its angle only, the sigma points through the dynamics drawn), their error against the truth. The moving-point world is `infer/filters/kalman-tracking`. | `filter/state` t!, `filter/covariance` t, `filter/gain` t, `filter/innovation` s, `filter/nis` s, `filter/error` s, `filter/sigmaPoints` t (UKF), `loop/control` s, `plane/poles` p (the estimator's) | `dynamics/filtering/kalman.ts`: `kalmanFilterAlgorithm`, `kalmanPredict`, `kalmanUpdate`, `steadyStateKalman`, `nisTest`; `dynamics/filtering/nonlinear.ts`: `extendedKalmanFilter`, `unscentedKalmanFilter`, `merweSigmaPoints` (whole-series; TODO the stepped forms as `kalmanFilterAlgorithm` has); `dynamics/control/optimal.ts`: `lqe`, `lqg` | built `methods/dynamics/kalmanTracking.ts` (the tracking world; TODO the plant-slot form with the loop closed) |

### act/control/identification/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `system-identification` | The method as the mode on input-output data from the plant slot (the excitation as a knob: a step, a PRBS, a chirp; the noise level as another): ARX by least squares and recursively one sample per step (the parameters converging, the forgetting factor as the knob following a plant that changes), ARMAX by the prediction-error method, N4SID with its singular values choosing the order (the step adds an order), DMD's modes and the Koopman view, SINDy's sparse coefficients (the threshold as the knob, terms dropping out as events) recovering the equations of a named nonlinear system; the identified model's step response against the plant's, the residuals' whiteness. | `data/n` s, `fit/parameters` t!, `fit/error` s, `fit/order` s, `fit/singularValues` t, `fit/termDropped` e (SINDy), `model/equations` p (TeX), `model/response` p (against the plant's), `model/residualAcf` p | `dynamics/identification.ts`: `arx`, `armax`, `predictionError`, `n4sid`, `dmd`, `edmd`, `polynomialLibrary`, `sindy`, `stlsq`; `dynamics/named-systems.ts`: `lorenzField`, `vanDerPolField` (the systems SINDy recovers); `symbolic`: `toLatex`; recursive least squares TODO (`dynamics/identification.ts`: `rlsAlgorithm`) | |

### act/control/robust/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `norms-and-reduction` | The H2 and H∞ norms of a loop's sensitivity (the peak on the Bode plot marked, the margins it implies), an uncertain parameter swept (the knob, one value per step) with the worst-case response and the gain that stays stable over the whole range, balanced truncation dropping the weakest states (the Hankel singular values as a bar chart, the reduced model's response error against the order kept). | `sweep/parameter` s, `sweep/hinfNorm` s, `sweep/stable` s, `sweep/worstResponse` t!, `reduction/hankel` t!, `reduction/order` s, `reduction/error` s, `bode/sensitivity` p | `systems/norms.ts`: `h2Norm`, `hinfNorm`, `systemNorm`; `systems/margins.ts`: `stabilityMargins`; `systems/reduction.ts`: `hsvd`, `balancedRealization`, `balred`; ready; H∞ synthesis and μ TODO (`dynamics/control/robust.ts`: `hinfSynthesis` by the two Riccati equations) | |
| `adaptive-control` | Model-reference adaptive control on a first-order plant with an unknown gain: the MIT rule's parameter adapting one sample per step (the adaptation gain as the knob, too large and the loop oscillates), the plant's output against the reference model's, the parameter error; gain scheduling as the other mode. | `loop/output` s, `loop/reference` s, `adaptation/parameter` s, `adaptation/error` s, `adaptation/gain` s | the loop base; TODO `dynamics/control/adaptive.ts`: `mrac` (the MIT rule and its Lyapunov form), `gainSchedule` | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `plants.ts` | The plant slot: textbook LTI plants (first and second order with ζ and ωn, integrator chains, the DC motor, mass-spring-damper, two tanks, a plant with a transport delay) as `systems` objects with their parameters as knobs, and the nonlinear plants of `environments` (cart-pole, pendulum, acrobot, the double integrator, the quadrotor) through `environments/models.ts`: `ControlModel`, `linearize` at an operating point and `controlModelEnv`; shared with `reinforcement-learning`'s environment slot | every group |
| `controllers.ts` | The controller slot as a mode table: PID, lead-lag, state feedback, observer-based, LQR, MPC (with its planner), a trained policy from `act/` (`agents/control.ts`: `lqrAgent`, `swingUpAgent`, `mpcAgent` are agents already) | `act/control/*` |
| `loop.ts` | The closed-loop base: one sample per step over a plant and a controller (the `simulatePid` and `recedingHorizon` shape), the input slot (setpoint profiles, disturbances, measurement noise), saturation, the observer in the loop, the standard paths (`loop/setpoint`, `loop/output`, `loop/error`, `loop/control`, `loop/state`) and the response measures as a probe | `act/control/*` |
| `plane.ts` | The complex-plane record: poles and zeros with the imaginary axis or the unit circle, damping and natural-frequency lines, the open- and closed-loop sets, a locus's branches so far | `linear-systems/*`, `state-space/`, `estimation/` |
| `discretise.ts` | A `ControlModel` on a grid as a `TabularMdp` (states binned, actions quantised, the model stepped), so dynamic programming of `reinforcement-learning` runs on a control plant | `predictive/pontryagin`, `reinforcement-learning` |

## Examples to make

- PID gains as knobs with the step response and the control effort; the integral winding up under saturation, then anti-windup.
- Root locus as the gain sweeps, with the closed-loop poles crossing the axis and the Routh array agreeing on the gain.
- The Kalman filter following a point it sees through noise (exists), with the assumed noise as the knob; then the same filter as the observer of a loop.
- MPC steering the cart-pole with the horizon drawn ahead of the cart each step and the input limit going active.
- LQR's Riccati recursion converging, the gain and the cost, Q and R as knobs.
- SINDy recovering the Lorenz equations as the threshold rises, terms dropping out one by one.

## Rendering

- Curves, pole-zero and Nyquist plots, episodes of an environment under a controller (`filmstrip`, `worldView`), a horizon overlay, Bode pairs.
- Missing: the complex plane as a panel kind (poles moving at the viewed step, a locus's branches); a Bode panel with the margins marked and a cursor on the frequency; a horizon overlay on the plant's scene.

## Problems

- The library covers the chapter; the gaps are small and named per row: the step-response measures, the Padé delay, lead-lag design rules, the indirect (Pontryagin) setup, recursive least squares, H∞ synthesis, adaptive control, and the stepped forms of the nonlinear filters.
- `methods/dynamics/kalmanTracking.ts` is claimed by three chapters (here as the observer, `time-series` for the structural models, `probabilistic-inference` for the filter); it goes to `infer/filters/` once, and this chapter's `estimation/` wraps the same filter on the plant slot.
- The plant slot and the environment slot are one registry seen twice: `environments/registry.ts` tags the control models as a `model` family already, and the LTI textbook plants need entries of their own (an `lti` family, or a registry of `systems` objects beside it).
- The split between `simulate/dynamics/linear-systems/` (analysis) and `act/control/` (loops) is the one place the chapter crosses two folders; one `act/control/` root holding `analysis/` as a group is the alternative, and a lookup either way.
