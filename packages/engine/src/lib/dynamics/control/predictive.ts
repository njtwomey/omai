import { WhiteboxError } from '../../base/index.ts'
import { qpInteriorPoint, quadprog } from '../../optim/index.ts'
import { asTensor, matmul, primalOf, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { f64 } from './_model.ts'
import { type DiscreteModel, ilqr } from './trajectory.ts'

type Mat = Float64Array

const at = (m: Mat, cols: number, r: number, c: number): number => m[r * cols + c] ?? 0

function matmulN(a: Mat, b: Mat, n: number, k: number, m: number): Mat {
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
 * @throws WhiteboxError with code `value` for a horizon below 1.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
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
  if (!Number.isInteger(N) || N < 1) throw new WhiteboxError('value', 'linearMpc: the horizon must be at least 1')
  // Powers A^k and the prediction matrices Phi (Nn x n) and Gamma (Nn x Nm).
  const powers: Mat[] = [Float64Array.from({ length: n * n }, (_, i) => (Math.floor(i / n) === i % n ? 1 : 0))]
  for (let k = 1; k <= N; k++) powers.push(matmulN(A, powers[k - 1] as Mat, n, n, n))
  const Phi = new Float64Array(N * n * n)
  for (let k = 0; k < N; k++) Phi.set(powers[k + 1] as Mat, k * n * n)
  const Gamma = new Float64Array(N * n * N * m)
  for (let k = 0; k < N; k++) {
    for (let j = 0; j <= k; j++) {
      const blk = matmulN(powers[k - j] as Mat, B, n, n, m)
      for (let r = 0; r < n; r++)
        for (let c = 0; c < m; c++) Gamma[(k * n + r) * (N * m) + j * m + c] = at(blk, m, r, c)
    }
  }
  // Qbar Gamma and the Hessian H = 2 (Gamma^T Qbar Gamma + Rbar).
  const nu = N * m
  const nx = N * n
  const qOf = (k: number): Mat => (k === N - 1 ? Qf : Q)
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

/** The state of {@link recedingHorizon}. */
export interface RecedingState {
  readonly x: number[]
  readonly states: number[][]
  readonly inputs: number[][]
  readonly costs: number[]
  readonly status: string
}

/**
 * Receding-horizon control as a traced algorithm: at every step plan from the current state, apply the first input
 * to the plant, and move on (the plan's cost and the applied input are recorded). Works with any planner: a
 * {@link linearMpc} controller or {@link nonlinearMpc}.
 *
 * @param planner - Plans from a state.
 * @param plant - The true system x' = plant(x, u) (default: the planner's own prediction).
 * @param x0 - The initial state.
 * @param options - Steps and tracing.
 * @returns The closed-loop states, inputs and plan costs.
 * @throws WhiteboxError with code `value` for a negative step count.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * const mpc = control.linearMpc({ A: [[1, 1], [0, 1]], B: [[0.5], [1]], Q: [[1, 0], [0, 1]], R: [[0.1]], horizon: 10, uMin: -1, uMax: 1 })
 * const run = control.recedingHorizon(mpc, (x, u) => [x[0] + x[1] + 0.5 * u[0], x[1] + u[0]], [5, 0], { steps: 30 })
 * Math.abs(run.states.at(-1)[0]) < 0.05 // => true
 * ```
 */
export function recedingHorizon(
  planner: { solve(x: readonly number[]): MpcPlan },
  plant: ((x: readonly number[], u: readonly number[]) => number[]) | undefined,
  x0: readonly number[],
  options: { readonly steps: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): RecedingState {
  if (!(options.steps >= 0)) throw new WhiteboxError('value', 'recedingHorizon: steps >= 0')
  const algorithm: Algorithm<RecedingState> = {
    name: 'receding-horizon',
    config: { steps: options.steps },
    schema: channels({
      metrics: {
        cost: { description: 'the cost of the current plan', direction: 'lower' },
        u0: { description: 'the first input' },
      },
      tensors: { x: { role: 'state', description: 'the plant state' } },
    }),
    init: () => ({ x: [...x0], states: [[...x0]], inputs: [], costs: [], status: 'running' }),
    step: (s) => {
      const plan = planner.solve(s.x)
      const next = plant === undefined ? (plan.states[0] ?? s.x) : plant(s.x, plan.u)
      s.states.push([...next])
      s.inputs.push(plan.u)
      s.costs.push(plan.cost)
      return { ...s, x: [...next], status: plan.status }
    },
    observe: (s) => ({
      metrics: { cost: s.costs.at(-1) ?? 0, u0: s.inputs.at(-1)?.[0] ?? 0 },
      tensors: { x: { role: 'state', value: s.x } },
    }),
  }
  return runAlgorithm(algorithm, { maxSteps: options.steps, ...options }).state
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
 * import { control } from 'whitebox-engine/dynamics'
 * import { add, mul, sum, sin } from 'whitebox-engine/tensor'
 * const model = { stateSize: 1, actionSize: 1, dynamics: (x, u) => add(add(x, mul(sin(x), 0.2)), u), cost: (x, u) => add(sum(mul(x, x), -1), sum(mul(u, u), -1)) }
 * control.nonlinearMpc(model, { horizon: 8 }).solve([1]).controls.length // => 8
 * ```
 */
export function nonlinearMpc(
  model: DiscreteModel,
  options: { readonly horizon: number; readonly iterations?: number; readonly terminalCost?: (x: Value) => Value },
): { solve(x: readonly number[]): MpcPlan } {
  let warm: number[][] | undefined
  return {
    solve: (x) => {
      const r = ilqr(model, x, {
        horizon: options.horizon,
        maxIterations: options.iterations ?? 20,
        ...(warm === undefined ? {} : { initial: warm }),
        ...(options.terminalCost === undefined ? {} : { terminalCost: options.terminalCost }),
      })
      warm = [...r.controls.slice(1), r.controls.at(-1) ?? new Array<number>(model.actionSize).fill(0)]
      return {
        u: r.controls[0] ?? [],
        controls: r.controls,
        states: r.states.slice(1),
        cost: r.cost,
        status: r.converged ? 'optimal' : 'iteration limit',
      }
    },
  }
}
