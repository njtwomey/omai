/**
 * Trajectory optimisation: finding the sequence of controls (and states) that minimises a cost under a dynamics model.
 * Iterative LQR and differential dynamic programming as traced algorithms with regularisation and line search, and the
 * direct transcriptions (single shooting, multiple shooting, trapezoidal direct collocation) that hand the problem to a
 * general optimiser.
 */

import { clamp, type Json, WhiteboxError } from '../../base/index.ts'
import { cholesky, daxpy, dgemv, solve, symmetricPart } from '../../numerics/linalg/index.ts'
import { lbfgsb, slsqp } from '../../optim/index.ts'
import {
  add,
  concat,
  eye,
  matmul,
  mul,
  neg,
  reshape,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  tanh,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { f64, rollout, stepDerivatives, weightedDynamicsHessian } from './_model.ts'

/**
 * A discrete-time model x' = f(x, u) with a stage cost l(x, u) (and an optional terminal cost), written with
 * operations on batches: `dynamics` and `cost` take states `[B, n]` and actions `[B, m]` (or single vectors) and
 * return `[B, n]` and `[B]`. The environment models of `methods/rl` have this shape.
 */
export interface DiscreteModel {
  readonly stateSize: number
  readonly actionSize: number
  dynamics(state: Value, action: Value): Value
  cost(state: Value, action: Value): Value
  terminalCost?(state: Value): Value
  readonly actionLow?: readonly number[]
  readonly actionHigh?: readonly number[]
  readonly equilibrium?: { readonly state: readonly number[]; readonly action: readonly number[] }
  readonly id?: string
}

/** Options of {@link ilqrAlgorithm} and {@link ilqr}. */
export interface IlqrOptions {
  /** The horizon (default the length of `initial`, else 50). */
  readonly horizon?: number
  /** Initial controls, one per step (default the equilibrium action, else zeros). */
  readonly initial?: readonly (readonly number[])[]
  /** Stop when the relative cost improvement falls below this (default 1e-6). */
  readonly tolerance?: number
  /** Initial regularisation mu (default 1) and its bounds (default 1e-6 and 1e10). */
  readonly mu?: number
  readonly muMin?: number
  readonly muMax?: number
  /** The factor by which mu grows or shrinks (Tassa et al.'s delta_0, default 2). */
  readonly muFactor?: number
  /** Line-search step sizes, largest first (default 1, 1/2, ..., 1/512). */
  readonly alphas?: readonly number[]
  /** Optimise unconstrained v with u = c + h tanh(v) inside the model's action bounds (default true with bounds). */
  readonly squash?: boolean
  /** A cost on the final state, replacing the model's. */
  readonly terminalCost?: (state: Value) => Value
  /** Use the second-order dynamics terms (DDP) instead of the Gauss-Newton iLQR backward pass (default false). */
  readonly secondOrder?: boolean
}

/** The state of an iLQR or DDP run. */
export interface IlqrState {
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
  /** The feedback gains K_t (`[m][n]`) and feedforward terms k_t of the last backward pass. */
  readonly gains: readonly (readonly (readonly number[])[])[]
  readonly feedforward: readonly (readonly number[])[]
  /** The cost after each accepted iteration (the first entry is the initial cost). */
  readonly history: readonly number[]
}

const rowsTensor = (rows: readonly (readonly number[])[], cols: number): Tensor => f64(rows.flat(), [rows.length, cols])
const vec = (v: readonly number[]): Tensor => f64(v, [v.length])
const asRows = (x: Value): number[][] => toArray(x) as number[][]

interface Squash {
  readonly model: DiscreteModel
  toRaw(u: readonly number[]): number[]
  toControl(v: readonly number[]): number[]
}

function squashModel(model: DiscreteModel, on: boolean): Squash {
  const low = model.actionLow
  const high = model.actionHigh
  if (!on || low === undefined || high === undefined) return { model, toRaw: (u) => [...u], toControl: (v) => [...v] }
  const c = low.map((l, i) => (l + (high[i] ?? l)) / 2)
  const h = low.map((l, i) => ((high[i] ?? l) - l) / 2)
  const toU = (v: Value): Value => add(mul(tanh(v), vec(h)), vec(c))
  const inner = model
  return {
    model: {
      ...inner,
      stateSize: inner.stateSize,
      actionSize: inner.actionSize,
      dynamics: (x, v) => inner.dynamics(x, toU(v)),
      cost: (x, v) => inner.cost(x, toU(v)),
      ...(inner.terminalCost === undefined
        ? {}
        : { terminalCost: (x: Value) => (inner.terminalCost as (s: Value) => Value)(x) }),
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

// The backward pass of Tassa, Erez and Todorov (2012) with the control-space regularisation Quu + mu I; with
// `secondOrder` the dynamics Hessians weighted by V_x are added (DDP). Undefined when a regularised Quu is not
// positive definite.
function backward(
  model: DiscreteModel,
  states: readonly (readonly number[])[],
  raw: readonly (readonly number[])[],
  mu: number,
  secondOrder: boolean,
): BackwardResult | undefined {
  const n = model.stateSize
  const m = model.actionSize
  const T = raw.length
  const d = stepDerivatives(model, states.slice(0, T), raw)
  let Vx: Value = vec(new Array<number>(n).fill(0))
  let Vxx: Value = zeros([n, n])
  if (model.terminalCost !== undefined) {
    const last = stepDerivatives(
      { ...model, cost: (x) => (model.terminalCost as (s: Value) => Value)(x) },
      [states[T] ?? []],
      [new Array<number>(m).fill(0)],
    )
    Vx = vec(last.lx[0] ?? [])
    Vxx = rowsTensor(last.lxx[0] ?? [], n)
  }
  const reg = mul(eye(m), mu)
  const K: Float64Array[] = new Array(T)
  const k: number[][] = new Array(T)
  let dV1 = 0
  let dV2 = 0
  for (let t = T - 1; t >= 0; t--) {
    const fx = rowsTensor(d.fx[t] ?? [], n)
    const fu = rowsTensor(d.fu[t] ?? [], m)
    const fxT = transpose(fx)
    const fuT = transpose(fu)
    const Qx = add(vec(d.lx[t] ?? []), matmul(fxT, Vx))
    const Qu = add(vec(d.lu[t] ?? []), matmul(fuT, Vx))
    let Qxx = add(rowsTensor(d.lxx[t] ?? [], n), matmul(fxT, matmul(Vxx, fx)))
    let Quu = add(rowsTensor(d.luu[t] ?? [], m), matmul(fuT, matmul(Vxx, fu)))
    let Qux = add(rowsTensor(d.lux[t] ?? [], n), matmul(fuT, matmul(Vxx, fx)))
    if (secondOrder) {
      const H = weightedDynamicsHessian(model, states[t] ?? [], raw[t] ?? [], toFlat(Vx))
      const w = n + m
      const block = (r0: number, c0: number, r: number, c: number): Tensor =>
        f64(
          Array.from({ length: r * c }, (_, idx) => H[(r0 + Math.floor(idx / c)) * w + c0 + (idx % c)] ?? 0),
          [r, c],
        )
      Qxx = add(Qxx, block(0, 0, n, n))
      Quu = add(Quu, block(n, n, m, m))
      Qux = add(Qux, block(n, 0, m, n))
    }
    const QuuR = symmetricPart(add(Quu, reg))
    try {
      cholesky(QuuR)
    } catch {
      return undefined
    }
    const kt = neg(solve(QuuR, Qu))
    const Kt = neg(solve(QuuR, Qux))
    const KtT = transpose(Kt)
    Vx = add(add(add(Qx, matmul(KtT, matmul(Quu, kt))), matmul(KtT, Qu)), matmul(transpose(Qux), kt))
    Vxx = symmetricPart(add(add(add(Qxx, matmul(KtT, matmul(Quu, Kt))), matmul(KtT, Qux)), matmul(transpose(Qux), Kt)))
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
  model: DiscreteModel,
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
  let total: Value = vec(new Array<number>(A).fill(0))
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
 * Iterative LQR (Li and Todorov 2004) with the regularisation and line search of Tassa, Erez and Todorov (2012), or
 * full DDP with `secondOrder`, as a traced algorithm: each step linearises the dynamics and quadratises the cost along
 * the trajectory (autodiff, all steps at once), runs the backward Riccati pass with Quu + mu I (raising mu until every
 * regularised Quu is positive definite), and rolls out u + alpha k + K (x - x_bar) for every step size in one batch,
 * accepting the largest that lowers the cost by a fair share of the predicted reduction. The cost never increases;
 * control bounds are handled through u = c + h tanh(v).
 *
 * @param model - The differentiable model.
 * @param state - The start state.
 * @param options - Horizon, initial controls, tolerance, regularisation, step sizes, squashing and DDP.
 * @returns The algorithm.
 * @concept optimal-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum, sliceAxis, concat } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(x, u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.ilqrAlgorithm(model, [1], { horizon: 5 }).name // => 'ilqr'
 * ```
 */
export function ilqrAlgorithm(
  model: DiscreteModel,
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
  const secondOrder = options.secondOrder ?? false
  const rest = model.equilibrium?.action ?? new Array<number>(m).fill(0)
  const increase = (s: { mu: number; delta: number }): { mu: number; delta: number } => {
    const delta = Math.max(delta0, s.delta * delta0)
    return { delta, mu: Math.max(muMin, s.mu * delta) }
  }
  return {
    name: secondOrder ? 'ddp' : 'ilqr',
    config: { model: model.id ?? 'model', horizon, tolerance: tol, alphas: [...alphas], secondOrder } as Json,
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
      const roll = rollout(sq.model, state, raw)
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
      let b = backward(sq.model, s.states, s.raw, reg.mu, secondOrder)
      while (b === undefined && reg.mu < muMax) {
        reg = increase(reg)
        b = backward(sq.model, s.states, s.raw, reg.mu, secondOrder)
      }
      if (b === undefined) return { ...s, mu: reg.mu, delta: reg.delta, alpha: 0, iterations: s.iterations + 1 }
      const found = lineSearch(sq.model, s.states, s.raw, b, alphas)
      const expectedFull = -(b.dV1 + b.dV2)
      let accepted = -1
      for (const [i, a] of alphas.entries()) {
        const expected = -(a * b.dV1 + a * a * b.dV2)
        const actual = s.cost - (found.costs[i] ?? Number.POSITIVE_INFINITY)
        if (actual > 0 && (expected <= 0 || actual / expected > 1e-4)) {
          accepted = i
          break
        }
      }
      const gains = b.K.map((Kt) =>
        Array.from({ length: m }, (_, i) => Array.from(Kt.slice(i * model.stateSize, (i + 1) * model.stateSize))),
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
  /** The optimised controls and the states they visit (horizon + 1, the start first). */
  readonly controls: number[][]
  readonly states: number[][]
  readonly cost: number
  /** The cost after each accepted iteration, starting from the initial controls'. */
  readonly history: readonly number[]
  /** The time-varying feedback gains K_t (in the optimised control coordinates). */
  readonly gains: readonly (readonly (readonly number[])[])[]
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Optimise a trajectory by iterative LQR or DDP (see {@link ilqrAlgorithm}).
 *
 * @param model - The differentiable model.
 * @param state - The start state.
 * @param options - iLQR options, the iteration budget (default 100) and the run options (tracer, policy).
 * @returns The controls, states, cost and its history.
 * @concept optimal-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(x, u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * const r = control.ilqr(model, [1], { horizon: 10 })
 * r.cost < (r.history[0] ?? 0) // => true
 * ```
 */
export function ilqr(
  model: DiscreteModel,
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
    secondOrder,
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
    ...(secondOrder === undefined ? {} : { secondOrder }),
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

/**
 * Differential dynamic programming (Mayne; Jacobson and Mayne 1970): iLQR with the second-order terms of the
 * dynamics in the backward pass, which converges quadratically near the optimum (see {@link ilqrAlgorithm}).
 *
 * @param model - The differentiable model.
 * @param state - The start state.
 * @param options - iLQR options and the iteration budget.
 * @returns The controls, states, cost and its history.
 * @concept differential-dynamic-programming
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum, sin } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(add(x, mul(sin(x), 0.1)), u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.ddp(model, [1], { horizon: 10 }).converged // => true
 * ```
 */
export function ddp(
  model: DiscreteModel,
  state: readonly number[],
  options: IlqrOptions & { readonly maxIterations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): IlqrResult {
  return ilqr(model, state, { ...options, secondOrder: true })
}

/** The result of a direct transcription (shooting or collocation). */
export interface TrajectoryResult {
  readonly controls: number[][]
  readonly states: number[][]
  readonly cost: number
  readonly converged: boolean
  readonly iterations: number
  readonly message: string
  /** The largest dynamics defect (0 for single shooting). */
  readonly defect: number
}

/** Options of the direct transcriptions. */
export interface TranscriptionOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  readonly horizon: number
  readonly initial?: readonly (readonly number[])[]
  readonly maxIterations?: number
  /** Box bounds on the controls (default the model's action bounds). */
  readonly actionLow?: readonly number[]
  readonly actionHigh?: readonly number[]
}

function actionBounds(
  model: DiscreteModel,
  options: TranscriptionOptions,
  T: number,
): [number | null, number | null][] {
  const lo = options.actionLow ?? model.actionLow
  const hi = options.actionHigh ?? model.actionHigh
  const out: [number | null, number | null][] = []
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < model.actionSize; j++) out.push([lo?.[j] ?? null, hi?.[j] ?? null])
  }
  return out
}

function totalCost(model: DiscreteModel, xs: readonly Value[], us: readonly Value[]): Value {
  let c: Value = mul(sum(xs[0] as Value), 0)
  us.forEach((u, t) => {
    c = add(c, sum(model.cost(xs[t] as Value, u)))
  })
  if (model.terminalCost !== undefined) c = add(c, sum(model.terminalCost(xs[xs.length - 1] as Value)))
  return c
}

/**
 * Single shooting: the controls are the only unknowns, the states follow by rolling the model out, and the total cost
 * (differentiated through the rollout) is minimised by L-BFGS-B within the action bounds.
 *
 * @param model - The model.
 * @param state - The start state.
 * @param options - Horizon, initial controls, iteration cap, bounds and tracing.
 * @returns The controls, states and cost.
 * @concept trajectory-optimisation
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(x, u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.singleShooting(model, [1], { horizon: 5 }).converged // => true
 * ```
 */
export function singleShooting(
  model: DiscreteModel,
  state: readonly number[],
  options: TranscriptionOptions,
): TrajectoryResult {
  const T = options.horizon
  const m = model.actionSize
  const x0 = vec(state)
  const objective = (w: Value): Value => {
    const xs: Value[] = [x0]
    const us: Value[] = []
    for (let t = 0; t < T; t++) {
      const u = sliceAxis(w, 0, t * m, (t + 1) * m)
      us.push(u)
      xs.push(model.dynamics(xs[t] as Value, u))
    }
    return totalCost(model, xs, us)
  }
  const start = f64(Array.from({ length: T }, (_, t) => options.initial?.[t] ?? new Array<number>(m).fill(0)).flat(), [
    T * m,
  ])
  const r = lbfgsb((w) => objective(w as Value), start, {
    bounds: actionBounds(model, options, T),
    maxIterations: options.maxIterations ?? 200,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
  })
  const w = toFlat(r.x as Tensor)
  const controls = Array.from({ length: T }, (_, t) => w.slice(t * m, (t + 1) * m))
  const roll = rollout(model, state, controls)
  return {
    controls,
    states: roll.states,
    cost: roll.cost,
    converged: r.converged,
    iterations: r.steps,
    message: r.message,
    defect: 0,
  }
}

/**
 * Multiple shooting: states and controls are both unknowns, linked by the defect constraints x_{t+1} = f(x_t, u_t),
 * and the total cost is minimised by SLSQP. The defects let the optimiser start from an infeasible guess, which makes
 * it robust on unstable or long horizons.
 *
 * @param model - The model.
 * @param state - The start state.
 * @param options - Horizon, initial controls, iteration cap, bounds and tracing.
 * @returns The controls, states, cost and the largest remaining defect.
 * @concept trajectory-optimisation
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(x, u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.multipleShooting(model, [1], { horizon: 5 }).defect < 1e-8 // => true
 * ```
 */
export function multipleShooting(
  model: DiscreteModel,
  state: readonly number[],
  options: TranscriptionOptions,
): TrajectoryResult {
  const T = options.horizon
  const n = model.stateSize
  const m = model.actionSize
  const x0 = vec(state)
  const split = (w: Value): { xs: Value[]; us: Value[] } => {
    const us = Array.from({ length: T }, (_, t) => sliceAxis(w, 0, t * m, (t + 1) * m))
    const xs = [x0, ...Array.from({ length: T }, (_, t) => sliceAxis(w, 0, T * m + t * n, T * m + (t + 1) * n))]
    return { xs, us }
  }
  const initial = Array.from({ length: T }, (_, t) => options.initial?.[t] ?? new Array<number>(m).fill(0))
  const guess = rollout(model, state, initial).states.slice(1)
  const start = f64([...initial.flat(), ...guess.flat()], [T * (m + n)])
  const defects = (w: Value): Value => {
    const { xs, us } = split(w)
    return concat(
      us.map((u, t) => sub(reshape(model.dynamics(xs[t] as Value, u), [n]), xs[t + 1] as Value)),
      0,
    )
  }
  const bounds = [
    ...actionBounds(model, options, T),
    ...Array.from({ length: T * n }, () => [null, null] as [null, null]),
  ]
  const r = slsqp(
    (w) => {
      const { xs, us } = split(w as Value)
      return totalCost(model, xs, us)
    },
    start,
    {
      constraints: [{ type: 'eq', fun: (w) => defects(w as Value) }],
      bounds,
      maxIterations: options.maxIterations ?? 200,
      ftol: 1e-10,
    },
  )
  const w = r.x as Tensor
  const { us } = split(w)
  const controls = us.map((u) => toFlat(u))
  const states = [
    [...state],
    ...Array.from({ length: T }, (_, t) => toFlat(w).slice(T * m + t * n, T * m + (t + 1) * n)),
  ]
  let defect = 0
  for (const v of toFlat(defects(w))) defect = Math.max(defect, Math.abs(v))
  return {
    controls,
    states,
    cost:
      toFlat(
        totalCost(
          model,
          states.map((x) => vec(x)),
          controls.map((u) => vec(u)),
        ),
      )[0] ?? Number.NaN,
    converged: r.converged,
    iterations: r.steps,
    message: r.message,
    defect,
  }
}

/** A continuous-time optimal control problem for {@link directCollocation}. */
export interface ContinuousProblem {
  readonly stateSize: number
  readonly actionSize: number
  /** x' = f(x, u) for single vectors. */
  dynamics(state: Value, action: Value): Value
  /** The running cost L(x, u). */
  cost(state: Value, action: Value): Value
  terminalCost?(state: Value): Value
}

/**
 * Direct collocation with the trapezoidal rule (Betts; Kelly 2017): states and controls at N + 1 nodes over [0, T],
 * defects x_{k+1} - x_k - h/2 (f_k + f_{k+1}) = 0, the trapezoidal cost sum h/2 (L_k + L_{k+1}), an optional fixed final
 * state, solved by SLSQP.
 *
 * @param problem - Dynamics and costs.
 * @param x0 - The initial state.
 * @param options - Duration T, nodes N, an optional final state, control bounds and the iteration cap.
 * @returns The node states and controls, the cost and the largest defect.
 * @concept direct-collocation
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { mul, sum, stack, sliceAxis } from 'whitebox-engine/tensor'
 * // Move a double integrator from rest at 0 to rest at 1 in time 1 with minimal control effort.
 * const problem = { stateSize: 2, actionSize: 1, dynamics: (x, u) => stack([sum(sliceAxis(x, 0, 1, 2)), sum(u)]), cost: (_x, u) => sum(mul(u, u)) }
 * const r = control.directCollocation(problem, [0, 0], { duration: 1, nodes: 20, final: [1, 0] })
 * // The exact minimum is 12 (u = 6 - 12 t); the trapezoidal rule is within 1%.
 * Math.abs(r.cost - 12) < 0.15 // => true
 * ```
 */
export function directCollocation(
  problem: ContinuousProblem,
  x0: readonly number[],
  options: {
    readonly duration: number
    readonly nodes: number
    readonly final?: readonly number[]
    readonly actionLow?: readonly number[]
    readonly actionHigh?: readonly number[]
    readonly maxIterations?: number
  },
): TrajectoryResult {
  const N = options.nodes
  if (!Number.isInteger(N) || N < 2) throw new WhiteboxError('value', 'directCollocation: at least two intervals')
  const n = problem.stateSize
  const m = problem.actionSize
  const h = options.duration / N
  const split = (w: Value): { xs: Value[]; us: Value[] } => ({
    xs: Array.from({ length: N + 1 }, (_, k) => sliceAxis(w, 0, k * n, (k + 1) * n)),
    us: Array.from({ length: N + 1 }, (_, k) => sliceAxis(w, 0, (N + 1) * n + k * m, (N + 1) * n + (k + 1) * m)),
  })
  const final = options.final
  const guessX = Array.from({ length: N + 1 }, (_, k) =>
    x0.map((v, i) => (final === undefined ? v : v + ((final[i] ?? v) - v) * (k / N))),
  )
  const start = f64([...guessX.flat(), ...new Array<number>((N + 1) * m).fill(0)], [(N + 1) * (n + m)])
  const cost = (w: Value): Value => {
    const { xs, us } = split(w)
    const L = xs.map((x, k) => sum(problem.cost(x, us[k] as Value)))
    let c: Value = mul(sum(L[0] as Value), 0)
    for (let k = 0; k < N; k++) c = add(c, mul(add(L[k] as Value, L[k + 1] as Value), h / 2))
    if (problem.terminalCost !== undefined) c = add(c, sum(problem.terminalCost(xs[N] as Value)))
    return c
  }
  const defects = (w: Value): Value => {
    const { xs, us } = split(w)
    const fs = xs.map((x, k) => reshape(problem.dynamics(x, us[k] as Value), [n]))
    const parts: Value[] = [sub(xs[0] as Value, vec(x0))]
    for (let k = 0; k < N; k++) {
      parts.push(sub(sub(xs[k + 1] as Value, xs[k] as Value), mul(add(fs[k] as Value, fs[k + 1] as Value), h / 2)))
    }
    if (final !== undefined) parts.push(sub(xs[N] as Value, vec(final)))
    return concat(parts, 0)
  }
  const bounds: [number | null, number | null][] = [
    ...Array.from({ length: (N + 1) * n }, () => [null, null] as [null, null]),
    ...Array.from(
      { length: (N + 1) * m },
      (_, idx) =>
        [options.actionLow?.[idx % m] ?? null, options.actionHigh?.[idx % m] ?? null] as [number | null, number | null],
    ),
  ]
  const r = slsqp((w) => cost(w as Value), start, {
    constraints: [{ type: 'eq', fun: (w) => defects(w as Value) }],
    bounds,
    maxIterations: options.maxIterations ?? 300,
    ftol: 1e-10,
  })
  const w = r.x as Tensor
  const flat = toFlat(w)
  let defect = 0
  for (const v of toFlat(defects(w))) defect = Math.max(defect, Math.abs(v))
  return {
    states: Array.from({ length: N + 1 }, (_, k) => flat.slice(k * n, (k + 1) * n)),
    controls: Array.from({ length: N + 1 }, (_, k) => flat.slice((N + 1) * n + k * m, (N + 1) * n + (k + 1) * m)),
    cost: toFlat(cost(w))[0] ?? Number.NaN,
    converged: r.converged,
    iterations: r.steps,
    message: r.message,
    defect,
  }
}
