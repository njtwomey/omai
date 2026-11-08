/**
 * Model predictive control: at each step, plan the inputs over a finite horizon by optimisation, apply only the first,
 * and plan again from the new state. Linear MPC in condensed form as a quadratic program, a traced receding-horizon
 * loop around a plant, and nonlinear MPC that plans with warm-started iLQR.
 */

import { OmaiError } from '../../base/index.ts'
import { qpInteriorPoint, quadprog } from '../../optim/index.ts'
import { asTensor, matmul, primalOf, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { fromNumbers } from '../ode/_vec.ts'
import { f64 } from './_model.ts'
import { type DiscreteModel, ilqr } from './trajectory.ts'

const at = (m: Float64Array, cols: number, r: number, c: number): number => m[r * cols + c] ?? 0

function matmulN(a: Float64Array, b: Float64Array, n: number, k: number, m: number): Float64Array {
  return Float64Array.from(primalOf(matmul(f64(Array.from(a), [n, k]), f64(Array.from(b), [k, m]))).data)
}

/** Options of {@link linearMpc}. */
export interface LinearMpcOptions {
  readonly A: TensorLike
  readonly B: TensorLike
  readonly Q: TensorLike
  readonly R: TensorLike
  /** Terminal weight (default Q). */
  readonly Qf?: TensorLike
  readonly horizon: number
  /** Bounds on every input component (scalars apply to all). */
  readonly uMin?: number | readonly number[]
  readonly uMax?: number | readonly number[]
  /** Bounds on every state component over the horizon. */
  readonly xMin?: number | readonly number[]
  readonly xMax?: number | readonly number[]
  /** The QP method (default active set). */
  readonly method?: 'active-set' | 'interior-point'
}

/** The plan of one MPC solve. */
export interface MpcPlan {
  /** The first input, to apply now. */
  readonly u: number[]
  /** All planned inputs u_0 .. u_{N-1}. */
  readonly controls: number[][]
  /** Predicted states x_1 .. x_N. */
  readonly states: number[][]
  readonly cost: number
  readonly status: string
}

/** A linear MPC controller (condensed form). */
export interface LinearMpc {
  readonly horizon: number
  /** The Hessian 2 (Gamma^T Qbar Gamma + Rbar) of the condensed QP. */
  readonly hessian: Float64Array
  /** Plan from a state (with an optional constant reference state). */
  solve(x0: readonly number[], reference?: readonly number[]): MpcPlan
}

const bound = (b: number | readonly number[] | undefined, i: number, d: number): number =>
  b === undefined ? d : typeof b === 'number' ? b : (b[i] ?? d)

/**
 * Linear model predictive control in the condensed form: the predictions X = Phi x0 + Gamma U turn the finite-horizon
 * cost sum x^T Q x + u^T R u (terminal weight Qf) into a quadratic program in the inputs alone, with input bounds and
 * state bounds as linear inequalities, solved by the active-set (or interior-point) QP solver. Without constraints
 * the first input equals the finite-horizon LQR feedback.
 *
 * @param options - Model, weights, horizon, bounds and QP method.
 * @returns The controller.
 * @throws OmaiError with code `value` for a horizon below 1.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * const mpc = control.linearMpc({ A: [[1, 1], [0, 1]], B: [[0.5], [1]], Q: [[1, 0], [0, 1]], R: [[0.1]], horizon: 10, uMin: -1, uMax: 1 })
 * Math.round((mpc.solve([5, 0]).u[0] ?? 0) * 1e9) / 1e9 // => -1
 * ```
 */
export function linearMpc(options: LinearMpcOptions): LinearMpc {
  const A = Float64Array.from(toFlat(primalOf(asTensor(options.A))))
  const B = Float64Array.from(toFlat(primalOf(asTensor(options.B))))
  const Q = Float64Array.from(toFlat(primalOf(asTensor(options.Q))))
  const R = Float64Array.from(toFlat(primalOf(asTensor(options.R))))
  const n = asTensor(options.A).shape[0] ?? 0
  const m = asTensor(options.B).shape[1] ?? 0
  const Qf = options.Qf === undefined ? Q : Float64Array.from(toFlat(primalOf(asTensor(options.Qf))))
  const N = options.horizon
  if (!Number.isInteger(N) || N < 1) throw new OmaiError('value', 'linearMpc: the horizon must be at least 1')
  // Powers A^k and the prediction matrices Phi (Nn x n) and Gamma (Nn x Nm).
  const powers: Float64Array[] = [Float64Array.from({ length: n * n }, (_, i) => (Math.floor(i / n) === i % n ? 1 : 0))]
  for (let k = 1; k <= N; k++) powers.push(matmulN(A, powers[k - 1] as Float64Array, n, n, n))
  const Phi = new Float64Array(N * n * n)
  for (let k = 0; k < N; k++) Phi.set(powers[k + 1] as Float64Array, k * n * n)
  const Gamma = new Float64Array(N * n * N * m)
  for (let k = 0; k < N; k++) {
    for (let j = 0; j <= k; j++) {
      const blk = matmulN(powers[k - j] as Float64Array, B, n, n, m)
      for (let r = 0; r < n; r++)
        for (let c = 0; c < m; c++) Gamma[(k * n + r) * (N * m) + j * m + c] = at(blk, m, r, c)
    }
  }
  // Qbar Gamma and the Hessian H = 2 (Gamma^T Qbar Gamma + Rbar).
  const nu = N * m
  const nx = N * n
  const qOf = (k: number): Float64Array => (k === N - 1 ? Qf : Q)
  const QG = new Float64Array(nx * nu)
  for (let k = 0; k < N; k++) {
    const blk = matmulN(qOf(k), Gamma.subarray(k * n * nu, (k + 1) * n * nu), n, n, nu)
    QG.set(blk, k * n * nu)
  }
  const H = new Float64Array(nu * nu)
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nu; j++) {
      let s = 0
      for (let r = 0; r < nx; r++) s += (Gamma[r * nu + i] ?? 0) * (QG[r * nu + j] ?? 0)
      H[i * nu + j] = 2 * s
    }
  }
  for (let k = 0; k < N; k++)
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) {
        H[(k * m + i) * nu + k * m + j] = (H[(k * m + i) * nu + k * m + j] ?? 0) + 2 * at(R, m, i, j)
      }
  const hasX = options.xMin !== undefined || options.xMax !== undefined
  return {
    horizon: N,
    hessian: H,
    solve: (x0, reference) => {
      const free = Float64Array.from({ length: nx }, (_, r) => {
        const k = Math.floor(r / n)
        const i = r % n
        let s = 0
        for (let c = 0; c < n; c++) s += (Phi[(k * n + i) * n + c] ?? 0) * (x0[c] ?? 0)
        return s - (reference?.[i] ?? 0)
      })
      // f = 2 Gamma^T Qbar (Phi x0 - r)
      const f = new Float64Array(nu)
      for (let j = 0; j < nu; j++) {
        let s = 0
        for (let r = 0; r < nx; r++) s += (QG[r * nu + j] ?? 0) * (free[r] ?? 0)
        f[j] = 2 * s
      }
      const lb = Float64Array.from({ length: nu }, (_, j) => bound(options.uMin, j % m, Number.NEGATIVE_INFINITY))
      const ub = Float64Array.from({ length: nu }, (_, j) => bound(options.uMax, j % m, Number.POSITIVE_INFINITY))
      const G: number[][] = []
      const h: number[] = []
      if (hasX) {
        for (let r = 0; r < nx; r++) {
          const i = r % n
          const row = Array.from({ length: nu }, (_, j) => Gamma[r * nu + j] ?? 0)
          const pred = (free[r] ?? 0) + (reference?.[i] ?? 0)
          const hi = bound(options.xMax, i, Number.POSITIVE_INFINITY)
          const lo = bound(options.xMin, i, Number.NEGATIVE_INFINITY)
          if (Number.isFinite(hi)) {
            G.push(row)
            h.push(hi - pred)
          }
          if (Number.isFinite(lo)) {
            G.push(row.map((v) => -v))
            h.push(pred - lo)
          }
        }
      }
      const toMatrix = (flat: Float64Array, rows: number, cols: number): number[][] =>
        Array.from({ length: rows }, (_, i) => Array.from(flat.subarray(i * cols, (i + 1) * cols)))
      const r = quadprog(
        {
          P: toMatrix(H, nu, nu),
          q: Array.from(f),
          ...(G.length > 0 ? { G, h } : {}),
          lb: Array.from(lb),
          ub: Array.from(ub),
        },
        options.method === 'interior-point' ? { method: qpInteriorPoint } : {},
      )
      const U = Array.from(r.x.data)
      const controls = Array.from({ length: N }, (_, k) => U.slice(k * m, (k + 1) * m))
      const X = Float64Array.from({ length: nx }, (_, row) => {
        let s = (free[row] ?? 0) + (reference?.[row % n] ?? 0)
        for (let j = 0; j < nu; j++) s += (Gamma[row * nu + j] ?? 0) * (U[j] ?? 0)
        return s
      })
      const states = Array.from({ length: N }, (_, k) => Array.from(X.subarray(k * n, (k + 1) * n)))
      return { u: controls[0] ?? [], controls, states, cost: r.fun, status: r.status }
    },
  }
}

/** A planner for {@link recedingHorizonAlgorithm}: plans from a state; one with memory also plans from a given warm start. */
export interface Planner {
  /** Plan from a state. */
  solve(x: readonly number[]): MpcPlan
  /**
   * Plan from a state warm-started by the previous plan's inputs shifted by one step (a planner that keeps no memory of
   * its own, so the loop's state alone resumes the run).
   */
  solveWarm?(x: readonly number[], warm: readonly (readonly number[])[]): MpcPlan
}

/** The state of {@link recedingHorizonAlgorithm}. */
export interface RecedingState {
  /** Steps taken. */
  readonly step: number
  /** The plant state. */
  readonly x: readonly number[]
  /** Every state visited (the first is the start). */
  readonly states: readonly (readonly number[])[]
  /** Every input applied and the cost of every plan. */
  readonly inputs: readonly (readonly number[])[]
  readonly costs: readonly number[]
  /** The status of the last plan. */
  readonly status: string
  /** The last plan's inputs shifted by one step, the next warm start (empty without a planner that takes one). */
  readonly warm: readonly (readonly number[])[]
}

/** The paths {@link recedingHorizonAlgorithm} logs, as a type. */
export type MpcPath = 'mpc/cost' | 'mpc/u' | 'mpc/plan' | 'mpc/predicted' | 'plant/x'

/**
 * What a receding-horizon loop logs, by path: the cost of the current plan, the input applied, the whole planned input
 * sequence and the states it predicts, and the plant state after the step.
 */
export const mpcContract: Contract<RecedingState, MpcPath> = defineContract({
  'mpc/cost': scalar({ description: 'the cost of the current plan', direction: 'lower' }),
  'mpc/u': tensor({ role: 'output', shape: ['input'], description: 'the first planned input, applied now' }),
  'mpc/plan': tensor({ role: 'output', shape: ['horizon', 'input'], description: 'all planned inputs' }),
  'mpc/predicted': tensor({ role: 'output', shape: ['horizon', 'state'], description: 'the states the plan predicts' }),
  'plant/x': tensor({ role: 'state', shape: ['state'], description: 'the plant state after this step' }),
})

/**
 * Receding-horizon control as a steppable algorithm: at every step plan from the current state, apply the first input
 * to the plant, and move on. Works with any planner: a {@link linearMpc} controller or {@link nonlinearMpc}. The state
 * holds the plant state, the history and the warm start, so it serialises and a run resumes with the same plans; it
 * logs the paths of {@link mpcContract}.
 *
 * @param planner - Plans from a state.
 * @param plant - The true system x' = plant(x, u) (default: the planner's own prediction).
 * @param x0 - The initial state.
 * @param options - The number of steps.
 * @returns The algorithm; it is done after `steps` periods.
 * @throws OmaiError with code `value` for a negative step count.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const mpc = control.linearMpc({ A: [[1, 1], [0, 1]], B: [[0.5], [1]], Q: [[1, 0], [0, 1]], R: [[0.1]], horizon: 5, uMin: -1, uMax: 1 })
 * const alg = control.recedingHorizonAlgorithm(mpc, undefined, [5, 0], { steps: 3 })
 * runAlgorithm(alg, { maxSteps: 3 }).state.inputs.length // => 3
 * ```
 */
export function recedingHorizonAlgorithm(
  planner: Planner,
  plant: ((x: readonly number[], u: readonly number[]) => number[]) | undefined,
  x0: readonly number[],
  options: { readonly steps: number },
): Algorithm<RecedingState, MpcPath> {
  if (!(options.steps >= 0)) throw new OmaiError('value', 'recedingHorizon: steps >= 0')
  return {
    name: 'receding-horizon',
    config: { steps: options.steps },
    schema: channels({
      metrics: {
        cost: { description: 'the cost of the current plan', direction: 'lower' },
        u0: { description: 'the first input' },
      },
      tensors: { x: { role: 'state', description: 'the plant state' } },
    }),
    init: () => ({ step: 0, x: [...x0], states: [[...x0]], inputs: [], costs: [], status: 'running', warm: [] }),
    step: (s, ctx) => {
      const plan = planner.solveWarm === undefined ? planner.solve(s.x) : planner.solveWarm(s.x, s.warm)
      const next = plant === undefined ? (plan.states[0] ?? s.x) : plant(s.x, plan.u)
      const warm =
        planner.solveWarm === undefined ? [] : [...plan.controls.slice(1), plan.controls.at(-1) ?? plan.u.map(() => 0)]
      if (ctx.wants('mpc/cost')) ctx.log('mpc/cost', plan.cost)
      if (ctx.wants('mpc/u')) ctx.log('mpc/u', plan.u)
      if (ctx.wants('mpc/plan'))
        ctx.log('mpc/plan', fromNumbers(plan.controls.flat(), [plan.controls.length, plan.u.length]))
      if (ctx.wants('mpc/predicted'))
        ctx.log('mpc/predicted', fromNumbers(plan.states.flat(), [plan.states.length, next.length]))
      if (ctx.wants('plant/x')) ctx.log('plant/x', [...next])
      return {
        step: s.step + 1,
        x: [...next],
        states: [...s.states, [...next]],
        inputs: [...s.inputs, plan.u],
        costs: [...s.costs, plan.cost],
        status: plan.status,
        warm,
      }
    },
    done: (s) => (s.step >= options.steps ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { cost: s.costs.at(-1) ?? 0, u0: s.inputs.at(-1)?.[0] ?? 0 },
      tensors: { x: { role: 'state', value: [...s.x] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RecedingState,
  }
}

/**
 * Receding-horizon control as a traced algorithm: at every step plan from the current state, apply the first input
 * to the plant, and move on (the plan's cost and the applied input are recorded). Works with any planner: a
 * {@link linearMpc} controller or {@link nonlinearMpc}. {@link recedingHorizonAlgorithm} run to the end.
 *
 * @param planner - Plans from a state.
 * @param plant - The true system x' = plant(x, u) (default: the planner's own prediction).
 * @param x0 - The initial state.
 * @param options - Steps and tracing.
 * @returns The closed-loop states, inputs and plan costs.
 * @throws OmaiError with code `value` for a negative step count.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * const mpc = control.linearMpc({ A: [[1, 1], [0, 1]], B: [[0.5], [1]], Q: [[1, 0], [0, 1]], R: [[0.1]], horizon: 10, uMin: -1, uMax: 1 })
 * const run = control.recedingHorizon(mpc, (x, u) => [x[0] + x[1] + 0.5 * u[0], x[1] + u[0]], [5, 0], { steps: 30 })
 * Math.abs(run.states.at(-1)[0]) < 0.05 // => true
 * ```
 */
export function recedingHorizon(
  planner: Planner,
  plant: ((x: readonly number[], u: readonly number[]) => number[]) | undefined,
  x0: readonly number[],
  options: { readonly steps: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): RecedingState {
  const { steps, ...run } = options
  const algorithm = recedingHorizonAlgorithm(planner, plant, x0, { steps })
  return runAlgorithm(algorithm, { maxSteps: steps, ...run }).state
}

/**
 * Nonlinear MPC: plan each step by iLQR on a nonlinear discrete model (bounds through squashing), warm-started from
 * the previous plan shifted by one step.
 *
 * @param model - The model.
 * @param options - Horizon, iLQR iterations per solve (default 20) and a terminal cost.
 * @returns A planner for {@link recedingHorizon}.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { add, mul, sum, sin } from 'omai-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(add(x, mul(sin(x), 0.2)), u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.nonlinearMpc(model, { horizon: 8 }).solve([1]).controls.length // => 8
 * ```
 */
export function nonlinearMpc(
  model: DiscreteModel,
  options: { readonly horizon: number; readonly iterations?: number; readonly terminalCost?: (x: Value) => Value },
): Planner {
  let warm: number[][] | undefined
  const plan = (x: readonly number[], start: readonly (readonly number[])[] | undefined): MpcPlan => {
    const r = ilqr(model, x, {
      horizon: options.horizon,
      maxIterations: options.iterations ?? 20,
      ...(start === undefined ? {} : { initial: start.map((u) => [...u]) }),
      ...(options.terminalCost === undefined ? {} : { terminalCost: options.terminalCost }),
    })
    return {
      u: r.controls[0] ?? [],
      controls: r.controls,
      states: r.states.slice(1),
      cost: r.cost,
      status: r.converged ? 'optimal' : 'iteration limit',
    }
  }
  const shifted = (p: MpcPlan): number[][] => [
    ...p.controls.slice(1),
    p.controls.at(-1) ?? new Array<number>(model.actionSize).fill(0),
  ]
  return {
    solve: (x) => {
      const p = plan(x, warm)
      warm = shifted(p)
      return p
    },
    // The same plan from a warm start the caller holds (the loop's state), so a resumed run needs no memory here.
    solveWarm: (x, start) => plan(x, start.length === 0 ? undefined : start),
  }
}
