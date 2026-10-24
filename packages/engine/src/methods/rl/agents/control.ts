import { argminOf, clamp, type Json, maxOf, WhiteboxError } from '../../../lib/base/index.ts'
import { cholesky, daxpy, dgemv, solve, solveDare } from '../../../lib/numerics/linalg/index.ts'
import { type Key, key as makeKey } from '../../../lib/random/index.ts'
import { type Agent, cem, type Env, mppi, type Transition } from '../../../lib/rl/index.ts'
import {
  add,
  matmul,
  mul,
  neg,
  type TensorLike,
  tanh,
  toArray,
  toFlat,
  transpose,
  type Value,
} from '../../../lib/tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../../lib/trace/index.ts'
import {
  type ControlModel,
  controlModelEnv,
  linearize,
  pendulumModel,
  rolloutModel,
  sequenceCosts,
  trajectoryDerivatives,
} from '../environments/models.ts'
import { rowsTensor, vectorTensor } from './_common.ts'

type Matrix = readonly (readonly number[])[]
const identity = (n: number, scale = 1): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? scale : 0)))
const asRows = (x: Value): number[][] => toArray(x) as number[][]
const sym = (a: Value): Value => mul(add(a, transpose(a)), 0.5)

// ------------------------------------------------------------------------------------------------------------ LQR

/** The solution of an infinite-horizon discrete-time LQR problem. */
export interface LqrSolution {
  /** The gain: the optimal control is `u = -K x`. */
  readonly K: Value
  /** The cost-to-go matrix: the optimal cost from `x` is `x^T P x`. */
  readonly P: Value
}

/**
 * The infinite-horizon discrete-time linear-quadratic regulator: minimise `sum x^T Q x + u^T R u` subject to
 * `x' = A x + B u`. `P` solves the discrete algebraic Riccati equation (lib/numerics/linalg `solveDare`, as
 * scipy.linalg.solve_discrete_are) and `K = (R + B^T P B)^{-1} B^T P A`.
 *
 * @param a - `A`, `[n, n]`.
 * @param b - `B`, `[n, m]`.
 * @param q - `Q`, `[n, n]` (positive semi-definite).
 * @param r - `R`, `[m, m]` (positive definite).
 * @returns The gain and the cost-to-go matrix.
 * @concept optimal-control
 * @example
 * ```ts
 * import { lqr } from 'whitebox-engine/methods/rl/agents'
 * import { item } from 'whitebox-engine/tensor'
 * Number(item(lqr([[1]], [[1]], [[1]], [[1]]).K).toFixed(12)) // => 0.618033988750
 * ```
 */
export function lqr(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike): LqrSolution {
  const P = solveDare(a, b, q, r)
  const bt = transpose(b)
  const K = solve(add(r, matmul(bt, matmul(P, b))), matmul(bt, matmul(P, a)))
  return { K, P }
}

// The environment action for a control: the nearest of a discrete environment's control values, else the control
// clipped to the model's bounds.
function toEnvAction<A>(model: ControlModel, u: readonly number[]): A {
  const choices = model.discreteActions
  if (choices !== undefined) {
    const d = choices.map((c) => c.reduce((s, v, i) => s + (v - (u[i] ?? 0)) ** 2, 0))
    return argminOf(d) as A
  }
  return u.map((v, i) => clamp(v, model.actionLow?.[i] ?? -Infinity, model.actionHigh?.[i] ?? Infinity)) as A
}

/** Options of {@link lqrAgent}. */
export interface LqrAgentOptions {
  /** State cost (default the identity). */
  readonly Q?: Matrix
  /** Control cost (default the identity). */
  readonly R?: Matrix
  /** The operating point to linearise at and regulate to (default the model's equilibrium). */
  readonly state?: readonly number[]
  readonly action?: readonly number[]
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: readonly number[]) => readonly number[]
}

/** The state of an LQR agent: its gain and operating point (fixed: the agent does not learn). */
export interface LqrAgentState {
  readonly K: readonly (readonly number[])[]
  readonly state: readonly number[]
  readonly action: readonly number[]
}

/**
 * A linear-quadratic regulator as an agent: linearise the model at its operating point (by autodiff), solve the
 * Riccati equation, and act with `u = u0 - K (x - x0)`, clipped to the action bounds or, for a discrete environment
 * such as CartPole, mapped to the nearest of its actions (bang-bang LQR). The agent's type parameter is the
 * environment's action type (`number` for discrete environments).
 *
 * @param model - The model (for example `cartPoleModel()` or `doubleIntegratorModel()`).
 * @param options - Cost matrices, operating point and the state of an observation.
 * @returns The agent.
 * @throws WhiteboxError with code `value` when no operating point is given and the model has no equilibrium.
 * @concept optimal-control
 * @example
 * ```ts
 * import { lqrAgent } from 'whitebox-engine/methods/rl/agents'
 * import { cartPoleModel } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = lqrAgent<number>(cartPoleModel())
 * agent.greedy(agent.init(key(0)), [0, 0, 0.05, 0]) // => 1
 * ```
 */
export function lqrAgent<A = readonly number[]>(
  model: ControlModel,
  options: LqrAgentOptions = {},
): Agent<LqrAgentState, readonly number[], A> {
  const x0 = options.state ?? model.equilibrium?.state
  const u0 = options.action ?? model.equilibrium?.action
  if (x0 === undefined || u0 === undefined) throw new WhiteboxError('value', 'lqrAgent: give an operating point')
  const toState = options.toState ?? ((o: readonly number[]) => o)
  const Q = options.Q ?? identity(model.stateSize)
  const R = options.R ?? identity(model.actionSize)
  const policy = (s: LqrAgentState, obs: readonly number[]): A => {
    const x = toState(obs)
    const dx = s.state.map((v, i) => (x[i] ?? 0) - v)
    const u = s.action.map((v, i) => v - (s.K[i] ?? []).reduce((acc, k, j) => acc + k * (dx[j] ?? 0), 0))
    return toEnvAction<A>(model, u)
  }
  return {
    name: 'lqr',
    config: { model: model.id, Q: Q.map((r) => [...r]), R: R.map((r) => [...r]) } as Json,
    init: () => {
      const { A, B } = linearize(model, x0, u0)
      return {
        K: asRows(lqr(A, B, rowsTensor(Q, model.stateSize), rowsTensor(R, model.actionSize)).K),
        state: x0,
        action: u0,
      }
    },
    act: (s, obs) => policy(s, obs),
    greedy: (s, obs) => policy(s, obs),
    learn: (s) => s,
  }
}

// ------------------------------------------------------------------------------------------------------- swing-up

/** Options of {@link swingUpAgent}. */
export interface SwingUpOptions {
  /** Gravity of the pendulum (default 10, gymnasium's). */
  readonly g?: number
  /** Gain of the energy controller (default 1; with the torque limit the pump is nearly bang-bang). */
  readonly energyGain?: number
  /** Hand over to LQR within this angle of upright (default 0.35 rad). */
  readonly switchAngle?: number
  /** LQR costs at upright (default `diag(1, 0.1)` and 0.1). */
  readonly Q?: Matrix
  readonly R?: Matrix
  /** The torque limit (default 2). */
  readonly maxTorque?: number
}

/** The state of a swing-up agent: its LQR gain at upright. */
export interface SwingUpState {
  readonly K: readonly number[]
}

/**
 * Swing up and balance gymnasium's Pendulum (Astrom and Furuta 2000): far from upright, an energy controller pumps the
 * pendulum's energy `E = thetad^2 / 2 + (3 g / 2)(cos theta - 1)` (zero at upright, where `dE/dt = 3 u thetad`)
 * towards zero with `u = sat(k (0 - E) sign(thetad))`; within `switchAngle` of upright an LQR controller from the
 * linearised {@link pendulumModel} catches and holds it. Observations are gymnasium's `[cos, sin, thetad]`. (A cart
 * double pendulum is not covered: energy shaping does not extend to it simply, and swinging it up needs a staged
 * controller or trajectory optimisation from a good initial guess.)
 *
 * @param options - Gravity, energy gain, switching angle, LQR costs and the torque limit.
 * @returns The agent.
 * @concept optimal-control
 * @example
 * ```ts
 * import { swingUpAgent } from 'whitebox-engine/methods/rl/agents'
 * import { key } from 'whitebox-engine/random'
 * const agent = swingUpAgent()
 * agent.greedy(agent.init(key(0)), [-1, 0, 1]) // => [2]
 * ```
 */
export function swingUpAgent(options: SwingUpOptions = {}): Agent<SwingUpState, readonly number[], readonly number[]> {
  const g = options.g ?? 10
  const k = options.energyGain ?? 1
  const switchAngle = options.switchAngle ?? 0.35
  const umax = options.maxTorque ?? 2
  const model = pendulumModel({ g })
  const Q = options.Q ?? [
    [1, 0],
    [0, 0.1],
  ]
  const R = options.R ?? [[0.1]]
  const policy = (s: SwingUpState, obs: readonly number[]): number[] => {
    const theta = Math.atan2(obs[1] ?? 0, obs[0] ?? 1)
    const w = obs[2] ?? 0
    if (Math.abs(theta) < switchAngle) {
      const u = -((s.K[0] ?? 0) * theta + (s.K[1] ?? 0) * w)
      return [clamp(u, -umax, umax)]
    }
    const energy = 0.5 * w * w + ((3 * g) / 2) * (Math.cos(theta) - 1)
    // Push along the motion to add energy (and against it to remove any excess); at rest, kick in either direction.
    const direction = w >= 0 ? 1 : -1
    return [clamp(k * -energy * direction, -umax, umax)]
  }
  return {
    name: 'swing-up',
    config: { g, energyGain: k, switchAngle, maxTorque: umax } as Json,
    init: () => {
      const { A, B } = linearize(model, [0, 0], [0])
      return { K: toFlat(lqr(A, B, rowsTensor(Q, 2), rowsTensor(R, 1)).K) }
    },
    act: (s, obs) => policy(s, obs),
    greedy: (s, obs) => policy(s, obs),
    learn: (s) => s,
  }
}

// ----------------------------------------------------------------------------------------------------------- iLQR

/** Options of {@link ilqrAlgorithm} and {@link ilqr}. */
export interface IlqrOptions {
  /** The horizon (default the length of `initial`, else 50). */
  readonly horizon?: number
  /** Initial controls, one per step (default the equilibrium action, else zeros). */
  readonly initial?: readonly (readonly number[])[]
  /** Stop when the relative cost improvement falls below this (default 1e-6). */
  readonly tolerance?: number
  /** Initial regularisation `mu` (default 1) and its bounds (default 1e-6 and 1e10). */
  readonly mu?: number
  readonly muMin?: number
  readonly muMax?: number
  /** The factor by which `mu` grows or shrinks (Tassa et al.'s delta_0, default 2). */
  readonly muFactor?: number
  /** Line-search step sizes, largest first (default 1, 1/2, ..., 1/512). */
  readonly alphas?: readonly number[]
  /**
   * Optimise unconstrained controls `v` with `u = c + h tanh(v)` inside the model's action bounds (default true when
   * the model has bounds).
   */
  readonly squash?: boolean
  /** A cost on the final state, replacing the model's (for example a large cost on not being upright). */
  readonly terminalCost?: (state: Value) => Value
}

/** The state of an iLQR run. */
export interface IlqrState {
  /** The current controls (in the model's action space) and the states they visit. */
  readonly controls: readonly (readonly number[])[]
  readonly states: readonly (readonly number[])[]
  readonly cost: number
  /** The optimised controls (before squashing). */
  readonly raw: readonly (readonly number[])[]
  readonly mu: number
  readonly delta: number
  /** The last accepted step size (0 when the line search failed). */
  readonly alpha: number
  /** The cost reduction the quadratic model predicted for the last full step. */
  readonly expected: number
  readonly iterations: number
  readonly converged: boolean
  /** The feedback gains `K_t` (`[m][n]`) and feedforward terms `k_t` of the last backward pass. */
  readonly gains: readonly (readonly (readonly number[])[])[]
  readonly feedforward: readonly (readonly number[])[]
  /** The cost after each accepted iteration (the first entry is the initial cost). */
  readonly history: readonly number[]
}

interface Squash {
  readonly model: ControlModel
  toRaw(u: readonly number[]): number[]
  toControl(v: readonly number[]): number[]
}

function squashModel(model: ControlModel, on: boolean): Squash {
  const low = model.actionLow
  const high = model.actionHigh
  if (!on || low === undefined || high === undefined) return { model, toRaw: (u) => [...u], toControl: (v) => [...v] }
  const c = low.map((l, i) => (l + (high[i] ?? l)) / 2)
  const h = low.map((l, i) => ((high[i] ?? l) - l) / 2)
  const toU = (v: Value): Value => add(mul(tanh(v), vectorTensor(h)), vectorTensor(c))
  const inner = model
  return {
    model: {
      ...inner,
      dynamics: (x, v) => inner.dynamics(x, toU(v)),
      cost: (x, v) => inner.cost(x, toU(v)),
    },
    toRaw: (u) => u.map((x, i) => Math.atanh(clamp((x - (c[i] ?? 0)) / (h[i] ?? 1), -1 + 1e-9, 1 - 1e-9))),
    toControl: (v) => v.map((x, i) => (c[i] ?? 0) + (h[i] ?? 1) * Math.tanh(x)),
  }
}

interface BackwardResult {
  readonly K: Float64Array[]
  readonly k: number[][]
  readonly dV1: number
  readonly dV2: number
}

// The backward pass of Tassa, Erez and Todorov (2012) with the Levenberg-Marquardt regularisation `Quu + mu I` (their
// control-space variant, which stays effective where squashed controls saturate); undefined when a regularised `Quu`
// is not positive definite.
function backward(
  model: ControlModel,
  states: readonly (readonly number[])[],
  raw: readonly (readonly number[])[],
  mu: number,
): BackwardResult | undefined {
  const n = model.stateSize
  const m = model.actionSize
  const T = raw.length
  const d = trajectoryDerivatives(model, states.slice(0, T), raw)
  let Vx: Value = vectorTensor(new Array<number>(n).fill(0))
  let Vxx: Value = rowsTensor(identity(n, 0), n)
  if (model.terminalCost !== undefined) {
    const last = trajectoryDerivatives(
      { ...model, cost: (x) => (model.terminalCost as (s: Value) => Value)(x) },
      [states[T] ?? []],
      [new Array<number>(m).fill(0)],
    )
    Vx = vectorTensor(last.lx[0] ?? [])
    Vxx = rowsTensor(last.lxx[0] ?? [], n)
  }
  const reg = rowsTensor(identity(m, mu), m)
  const K: Float64Array[] = new Array(T)
  const k: number[][] = new Array(T)
  let dV1 = 0
  let dV2 = 0
  for (let t = T - 1; t >= 0; t--) {
    const fx = rowsTensor(d.fx[t] ?? [], n)
    const fu = rowsTensor(d.fu[t] ?? [], m)
    const fxT = transpose(fx)
    const fuT = transpose(fu)
    const Qx = add(vectorTensor(d.lx[t] ?? []), matmul(fxT, Vx))
    const Qu = add(vectorTensor(d.lu[t] ?? []), matmul(fuT, Vx))
    const Qxx = add(rowsTensor(d.lxx[t] ?? [], n), matmul(fxT, matmul(Vxx, fx)))
    const Quu = add(rowsTensor(d.luu[t] ?? [], m), matmul(fuT, matmul(Vxx, fu)))
    const Qux = add(rowsTensor(d.lux[t] ?? [], n), matmul(fuT, matmul(Vxx, fx)))
    const QuuR = sym(add(Quu, reg))
    try {
      cholesky(QuuR)
    } catch {
      return undefined
    }
    const kt = neg(solve(QuuR, Qu))
    const Kt = neg(solve(QuuR, Qux))
    const KtT = transpose(Kt)
    Vx = add(add(add(Qx, matmul(KtT, matmul(Quu, kt))), matmul(KtT, Qu)), matmul(transpose(Qux), kt))
    Vxx = sym(add(add(add(Qxx, matmul(KtT, matmul(Quu, Kt))), matmul(KtT, Qux)), matmul(transpose(Qux), Kt)))
    const kv = toFlat(kt)
    const qu = toFlat(Qu)
    const quuK = toFlat(matmul(Quu, kt))
    for (let i = 0; i < m; i++) {
      dV1 += (kv[i] ?? 0) * (qu[i] ?? 0)
      dV2 += 0.5 * (kv[i] ?? 0) * (quuK[i] ?? 0)
    }
    K[t] = Float64Array.from(toFlat(Kt))
    k[t] = kv
  }
  return { K, k, dV1, dV2 }
}

// All line-search candidates rolled out as one batch: x_hat for every alpha, controls u + alpha k + K (x_hat - x).
function lineSearch(
  model: ControlModel,
  states: readonly (readonly number[])[],
  raw: readonly (readonly number[])[],
  b: BackwardResult,
  alphas: readonly number[],
): { readonly states: number[][][]; readonly raw: number[][][]; readonly costs: number[] } {
  const n = model.stateSize
  const m = model.actionSize
  const A = alphas.length
  const T = raw.length
  let X: number[][] = alphas.map(() => [...(states[0] ?? [])])
  const xs: number[][][] = alphas.map((_, a) => [X[a] as number[]])
  const us: number[][][] = alphas.map(() => [])
  let total: Value = vectorTensor(new Array<number>(A).fill(0))
  for (let t = 0; t < T; t++) {
    const U = X.map((x, a) => {
      const dx = daxpy(-1, states[t] ?? [], x)
      const fb = dgemv(b.K[t] ?? [], m, n, dx)
      return Array.from(daxpy(alphas[a] ?? 0, b.k[t] ?? [], daxpy(1, fb, raw[t] ?? [])))
    })
    for (const [a, u] of U.entries()) us[a]?.push(u)
    const xb = rowsTensor(X, n)
    const ub = rowsTensor(U, m)
    total = add(total, model.cost(xb, ub))
    const next = asRows(model.dynamics(xb, ub))
    X = next
    for (const [a, x] of next.entries()) xs[a]?.push(x)
  }
  if (model.terminalCost !== undefined) total = add(total, model.terminalCost(rowsTensor(X, n)))
  return { states: xs, raw: us, costs: toFlat(total) }
}

const DEFAULT_ALPHAS = Array.from({ length: 10 }, (_, i) => 2 ** -i)

/**
 * Iterative LQR (Li and Todorov 2004) with the regularisation and line search of Tassa, Erez and Todorov (2012), as
 * an {@link Algorithm}: each step linearises the dynamics and quadratises the cost along the current trajectory (by
 * autodiff, all steps at once), runs the backward Riccati pass with `Quu + mu I` regularisation (raising `mu` until
 * every regularised `Quu` is positive definite), and rolls out the new controls `u + alpha k + K (x - x_bar)` for every step size
 * in one batch, accepting the largest that lowers the cost by a fair share of the predicted reduction (then lowering
 * `mu`; otherwise raising it). The cost never increases. Control bounds are handled by optimising through
 * `u = c + h tanh(v)`. Records the cost, `mu`, the accepted step size and the predicted reduction.
 *
 * @param model - The differentiable model.
 * @param state - The start state.
 * @param options - Horizon, initial controls, tolerance, regularisation, step sizes and squashing.
 * @returns The algorithm.
 * @concept optimal-control
 * @example
 * ```ts
 * import { ilqrAlgorithm } from 'whitebox-engine/methods/rl/agents'
 * import { pendulumModel } from 'whitebox-engine/methods/rl'
 * ilqrAlgorithm(pendulumModel(), [Math.PI, 0], { horizon: 10 }).name // => 'ilqr'
 * ```
 */
export function ilqrAlgorithm(
  model: ControlModel,
  state: readonly number[],
  options: IlqrOptions = {},
): Algorithm<IlqrState> {
  const m = model.actionSize
  const withEnd = options.terminalCost === undefined ? model : { ...model, terminalCost: options.terminalCost }
  const sq = squashModel(withEnd, options.squash ?? true)
  const horizon = options.horizon ?? options.initial?.length ?? 50
  const tol = options.tolerance ?? 1e-6
  const muMin = options.muMin ?? 1e-6
  const muMax = options.muMax ?? 1e10
  const delta0 = options.muFactor ?? 2
  const alphas = options.alphas ?? DEFAULT_ALPHAS
  const rest = model.equilibrium?.action ?? new Array<number>(m).fill(0)
  const increase = (s: { mu: number; delta: number }): { mu: number; delta: number } => {
    const delta = Math.max(delta0, s.delta * delta0)
    return { delta, mu: Math.max(muMin, s.mu * delta) }
  }
  return {
    name: 'ilqr',
    config: { model: model.id, horizon, tolerance: tol, alphas: [...alphas] } as Json,
    schema: channels({
      metrics: {
        cost: { description: 'total cost of the current trajectory', direction: 'lower' },
        mu: { description: 'regularisation of the backward pass' },
        alpha: { description: 'accepted line-search step (0 when rejected)' },
        expected: { description: 'cost reduction predicted for a full step' },
      },
    }),
    init: () => {
      const controls = Array.from({ length: horizon }, (_, t) => [...(options.initial?.[t] ?? rest)])
      const raw = controls.map((u) => sq.toRaw(u))
      const roll = rolloutModel(sq.model, state, raw)
      return {
        controls: raw.map((v) => sq.toControl(v)),
        states: roll.states,
        cost: roll.cost,
        raw,
        mu: options.mu ?? 1,
        delta: delta0,
        alpha: 0,
        expected: 0,
        iterations: 0,
        converged: false,
        gains: [],
        feedforward: [],
        history: [roll.cost],
      }
    },
    step: (s) => {
      let reg = { mu: s.mu, delta: s.delta }
      let b = backward(sq.model, s.states, s.raw, reg.mu)
      while (b === undefined && reg.mu < muMax) {
        reg = increase(reg)
        b = backward(sq.model, s.states, s.raw, reg.mu)
      }
      if (b === undefined) return { ...s, mu: reg.mu, delta: reg.delta, alpha: 0, iterations: s.iterations + 1 }
      const found = lineSearch(sq.model, s.states, s.raw, b, alphas)
      const expectedFull = -(b.dV1 + b.dV2)
      let accepted = -1
      for (const [i, a] of alphas.entries()) {
        const expected = -(a * b.dV1 + a * a * b.dV2)
        const actual = s.cost - (found.costs[i] ?? Infinity)
        if (actual > 0 && (expected <= 0 || actual / expected > 1e-4)) {
          accepted = i
          break
        }
      }
      const gains = b.K.map((K) =>
        Array.from({ length: m }, (_, i) => Array.from(K.slice(i * model.stateSize, (i + 1) * model.stateSize))),
      )
      if (accepted < 0) {
        const up = increase(reg)
        return {
          ...s,
          mu: up.mu,
          delta: up.delta,
          alpha: 0,
          expected: expectedFull,
          iterations: s.iterations + 1,
          converged: expectedFull < tol * Math.abs(s.cost),
          gains,
          feedforward: b.k,
        }
      }
      const cost = found.costs[accepted] ?? s.cost
      const raw = found.raw[accepted] ?? []
      const delta = Math.min(1 / delta0, reg.delta / delta0)
      const mu = reg.mu * delta > muMin ? reg.mu * delta : 0
      return {
        controls: raw.map((v) => sq.toControl(v)),
        states: found.states[accepted] ?? [],
        cost,
        raw,
        mu,
        delta,
        alpha: alphas[accepted] ?? 0,
        expected: expectedFull,
        iterations: s.iterations + 1,
        converged: (s.cost - cost) / Math.max(Math.abs(s.cost), 1e-12) < tol,
        gains,
        feedforward: b.k,
        history: [...s.history, cost],
      }
    },
    done: (s) => (s.converged ? 'converged' : s.mu >= muMax ? 'stopped' : undefined),
    observe: (s) => ({ metrics: { cost: s.cost, mu: s.mu, alpha: s.alpha, expected: s.expected } }),
  }
}

/** The result of {@link ilqr}. */
export interface IlqrResult {
  /** The optimised controls and the states they visit (`horizon + 1`, the start first). */
  readonly controls: number[][]
  readonly states: number[][]
  readonly cost: number
  /** The cost after each accepted iteration, starting from the initial controls'. */
  readonly history: readonly number[]
  /** The time-varying feedback gains `K_t` (in the optimised control coordinates). */
  readonly gains: readonly (readonly (readonly number[])[])[]
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Optimise a trajectory by iterative LQR (see {@link ilqrAlgorithm}).
 *
 * @param model - The differentiable model.
 * @param state - The start state.
 * @param options - iLQR options, the iteration budget (default 100) and the run options (tracer, policy).
 * @returns The controls, states, cost and its history.
 * @concept optimal-control
 * @example
 * ```ts
 * import { ilqr } from 'whitebox-engine/methods/rl/agents'
 * import { doubleIntegratorModel } from 'whitebox-engine/methods/rl'
 * const r = ilqr(doubleIntegratorModel(), [1, 0], { horizon: 30, squash: false })
 * r.cost < r.history[0] // => true
 * ```
 */
export function ilqr(
  model: ControlModel,
  state: readonly number[],
  options: IlqrOptions & { readonly maxIterations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): IlqrResult {
  const {
    maxIterations,
    horizon,
    initial,
    tolerance,
    mu,
    muMin,
    muMax,
    muFactor,
    alphas,
    squash,
    terminalCost,
    ...run
  } = options
  const opts: IlqrOptions = {
    ...(horizon === undefined ? {} : { horizon }),
    ...(initial === undefined ? {} : { initial }),
    ...(tolerance === undefined ? {} : { tolerance }),
    ...(mu === undefined ? {} : { mu }),
    ...(muMin === undefined ? {} : { muMin }),
    ...(muMax === undefined ? {} : { muMax }),
    ...(muFactor === undefined ? {} : { muFactor }),
    ...(alphas === undefined ? {} : { alphas }),
    ...(squash === undefined ? {} : { squash }),
    ...(terminalCost === undefined ? {} : { terminalCost }),
  }
  const result = runAlgorithm(ilqrAlgorithm(model, state, opts), { ...run, maxSteps: maxIterations ?? 100 })
  const s = result.state
  return {
    controls: s.controls.map((u) => [...u]),
    states: s.states.map((x) => [...x]),
    cost: s.cost,
    history: s.history,
    gains: s.gains,
    iterations: s.iterations,
    converged: s.converged,
  }
}

// ------------------------------------------------------------------------------------------------- receding horizon

/** The state of a receding-horizon agent: the current plan, the observation it was made for and its value. */
export interface PlanState {
  readonly plan?: readonly (readonly number[])[]
  /** The observation the plan starts from (as a key), so `act` can reuse a plan made in `learn`. */
  readonly plannedFor?: string
  /** The planner's estimate of the plan's return. */
  readonly value: number
  readonly replans: number
}

const shift = (plan: readonly (readonly number[])[] | undefined): number[][] | undefined =>
  plan === undefined || plan.length === 0
    ? undefined
    : [...plan.slice(1).map((u) => [...u]), [...(plan[plan.length - 1] ?? [])]]

// The agent protocol around a planner: plans are made in `learn` for the next observation, warm-started by shifting
// the previous plan one step, and reused by `act`; a fresh episode plans in `act` from the stale plan.
function recedingHorizon<O, A>(
  name: string,
  config: Json,
  plan: (obs: O, warm: readonly (readonly number[])[] | undefined, key: Key) => { plan: number[][]; value: number },
  toAction: (u: readonly number[]) => A,
): Agent<PlanState, O, A> {
  const id = (o: O): string => JSON.stringify(o)
  const first = (s: PlanState, obs: O, k: Key): A => {
    if (s.plan !== undefined && s.plannedFor === id(obs)) return toAction(s.plan[0] ?? [])
    return toAction(plan(obs, s.plan, k).plan[0] ?? [])
  }
  return {
    name,
    config,
    metrics: { planValue: { description: "the planner's estimate of the current plan's return", direction: 'higher' } },
    init: () => ({ value: 0, replans: 0 }),
    act: first,
    greedy: (s, obs) => first(s, obs, makeKey(0)),
    learn: (s, t: Transition<O, A>, k) => {
      if (t.terminated || t.truncated) return { value: s.value, replans: s.replans }
      const warm = s.plannedFor === id(t.observation) ? shift(s.plan) : s.plan
      const r = plan(t.nextObservation, warm, k)
      return { plan: r.plan, plannedFor: id(t.nextObservation), value: r.value, replans: s.replans + 1 }
    },
    stats: (s) => ({ planValue: s.value }),
  }
}

/** Options of {@link mpcAgent}. */
export interface MpcAgentOptions {
  /** The planner: iLQR on the model (default), or the cross-entropy method or MPPI with batched model rollouts. */
  readonly planner?: 'ilqr' | 'cem' | 'mppi'
  readonly horizon?: number
  /** Planner iterations per decision (default 3 for iLQR and CEM, 1 for MPPI). */
  readonly iterations?: number
  /** Sampling planners: candidates, elites (CEM), sampling standard deviation and temperature (MPPI). */
  readonly population?: number
  readonly elites?: number
  readonly sigma?: number
  readonly lambda?: number
  /** Further iLQR options. */
  readonly ilqr?: Omit<IlqrOptions, 'horizon' | 'initial'>
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: readonly number[]) => readonly number[]
}

/**
 * Model predictive control: at every step, plan over a receding horizon in the differentiable model, apply the first
 * action, and warm-start the next plan with the rest of this one shifted by a step. The planner is iLQR, or CEM or
 * MPPI scoring whole populations of action sequences in one batched model rollout. Plans live in the agent state:
 * `learn` plans for the next observation (so run the agent with `trainAgent`), `act` reuses that plan. The type
 * parameter is the environment's action type (discrete environments get the nearest of their actions).
 *
 * @param model - The differentiable model.
 * @param options - Planner, horizon, iterations, sampling options and the state of an observation.
 * @returns The agent.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { mpcAgent } from 'whitebox-engine/methods/rl/agents'
 * import { pendulumModel } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const toState = (o: readonly number[]) => [Math.atan2(o[1] ?? 0, o[0] ?? 1), o[2] ?? 0]
 * const agent = mpcAgent(pendulumModel(), { horizon: 10, toState })
 * Math.abs(agent.greedy(agent.init(key(0)), [1, 0.1, 0])[0] ?? 0) <= 2 // => true
 * ```
 */
export function mpcAgent<A = readonly number[]>(
  model: ControlModel,
  options: MpcAgentOptions = {},
): Agent<PlanState, readonly number[], A> {
  const planner = options.planner ?? 'ilqr'
  const horizon = options.horizon ?? 20
  const iterations = options.iterations ?? (planner === 'mppi' ? 1 : 3)
  const toState = options.toState ?? ((o: readonly number[]) => o)
  const env = controlModelEnv(model)
  // Default sampling spread: a quarter of the widest action range (1 for unbounded models).
  const ranges = model.actionHigh?.map((h, i) => (h - (model.actionLow?.[i] ?? -h)) / 4) ?? [1]
  const sigma = options.sigma ?? maxOf(ranges)
  const plan = (obs: readonly number[], warm: readonly (readonly number[])[] | undefined, k: Key) => {
    const x = toState(obs)
    if (planner === 'ilqr') {
      const r = ilqr(model, x, {
        ...options.ilqr,
        horizon,
        ...(warm === undefined ? {} : { initial: warm }),
        maxIterations: iterations,
      })
      return { plan: r.controls, value: -r.cost }
    }
    const score = (plans: readonly (readonly (readonly number[])[])[]): number[] =>
      sequenceCosts(model, x, plans).map((c) => -c)
    const common = {
      horizon,
      population: options.population ?? 64,
      sigma,
      score,
      ...(warm === undefined ? {} : { initial: warm }),
    }
    const r =
      planner === 'cem'
        ? cem(env, x, {
            ...common,
            iterations,
            key: k,
            ...(options.elites === undefined ? {} : { elites: options.elites }),
          })
        : mppi(env, x, { ...common, iterations, key: k, lambda: options.lambda ?? 1 })
    return { plan: r.plan, value: r.value }
  }
  return recedingHorizon<readonly number[], A>(
    `mpc-${planner}`,
    { model: model.id, planner, horizon, iterations } as Json,
    plan,
    (u) => toEnvAction<A>(model, u),
  )
}

/** Options of {@link cemAgent}. */
export interface CemAgentOptions<S, O> {
  /** The environment state of an observation (default the observation itself). */
  readonly toState?: (observation: O) => S
  readonly horizon?: number
  readonly population?: number
  readonly elites?: number
  /** CEM iterations per decision (default 3). */
  readonly iterations?: number
  readonly sigma?: number
  readonly alpha?: number
  readonly gamma?: number
}

/**
 * A cross-entropy-method planning agent that uses the true environment as its model (lib/rl `cem`): at every step it
 * recovers the environment state from the observation, refits a Gaussian over action sequences by simulating
 * candidates in the environment, and applies the first action; the shifted plan warm-starts the next decision. Plans
 * live in the agent state, made in `learn` for the next observation (run it with `trainAgent`).
 *
 * @param env - The environment to plan in (unwrapped, with a box action space).
 * @param options - The state of an observation, and the CEM options.
 * @returns The agent.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { cemAgent } from 'whitebox-engine/methods/rl/agents'
 * import { pendulum } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = cemAgent(pendulum.create(), { toState: (o) => [Math.atan2(o[1] ?? 0, o[0] ?? 1), o[2] ?? 0] as const, horizon: 5, population: 16 })
 * agent.act(agent.init(key(0)), [1, 0, 0], key(1)).length // => 1
 * ```
 */
export function cemAgent<S, O>(
  env: Env<S, O, readonly number[]>,
  options: CemAgentOptions<S, O> = {},
): Agent<PlanState, O, readonly number[]> {
  const toState = options.toState ?? ((o: O) => o as unknown as S)
  const horizon = options.horizon ?? 15
  const iterations = options.iterations ?? 3
  const plan = (obs: O, warm: readonly (readonly number[])[] | undefined, k: Key) => {
    const r = cem(env, toState(obs), {
      horizon,
      iterations,
      key: k,
      ...(options.population === undefined ? {} : { population: options.population }),
      ...(options.elites === undefined ? {} : { elites: options.elites }),
      ...(options.sigma === undefined ? {} : { sigma: options.sigma }),
      ...(options.alpha === undefined ? {} : { alpha: options.alpha }),
      ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
      ...(warm === undefined ? {} : { initial: warm }),
    })
    return { plan: r.plan, value: r.value }
  }
  return recedingHorizon<O, readonly number[]>(
    'cem',
    { env: env.spec.id, horizon, iterations, population: options.population ?? 64 } as Json,
    plan,
    (u) => [...u],
  )
}
