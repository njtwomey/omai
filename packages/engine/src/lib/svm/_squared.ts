// The squared-hinge (L2-loss) support vector classifier and the solver slot of the support vector classifier. With the
// squared hinge the primal, `1/2 b'K b + sum_i C_i max(0, 1 - y_i (K b + c)_i)^2`, is a smooth objective of the
// kernel-expansion coefficients, so every solver of `smoothSolvers` minimises it; its dual is the C-SVC dual with
// `1 / (2 C_i)` added to the diagonal and no upper bound, so SMO solves the same problem exactly. `svcSolvers` is the
// table of the two (SMO the default, the generic solvers admissible for the squared hinge only). Not exported from the
// barrel.

import { OmaiError } from '../base/index.ts'
import {
  analyticObjective,
  familySolvers,
  type SmoothProblem,
  type SmoProblem,
  type SmoState,
  smoAlgorithm,
  smoRho,
  type VectorParams,
} from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { dict, type SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import {
  type Algorithm,
  type ModeSpec,
  type ModeTable,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import type { Dual, DualOptions, Expansion } from './_dual.ts'

/** An upper bound that never binds, finite so that the pair updates stay free of NaN. */
export const UNBOUNDED = 1e30

/** One pair's classification problem on its Gram matrix. */
export interface SvcPairProblem extends SmoothProblem {
  readonly K: Float64Array
  readonly y: Float64Array
  readonly C: Float64Array
  readonly options: DualOptions
}

export const sumOf = (v: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] ?? 0
  return s
}

/** `K v` for a symmetric `n x n` matrix. */
export function times(K: Float64Array, v: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (K[i * n + j] ?? 0) * (v[j] ?? 0)
    out[i] = s
  }
  return out
}

/** The squared-hinge primal as a smooth problem over `(beta, c)`: the objective, its analytic gradient, a zero start. */
export function squaredHingePrimal(
  K: Float64Array,
  y: Float64Array,
  C: Float64Array,
  options: DualOptions,
): SvcPairProblem {
  const n = y.length
  const parts = (z: Float64Array): { kb: Float64Array; e: Float64Array } => {
    const kb = times(K, z.subarray(0, n), n)
    const c = z[n] ?? 0
    const e = Float64Array.from({ length: n }, (_, i) => Math.max(0, 1 - (y[i] ?? 1) * ((kb[i] ?? 0) + c)))
    return { kb, e }
  }
  const objective = analyticObjective({
    value: (z) => {
      const { kb, e } = parts(z)
      let v = 0
      for (let i = 0; i < n; i++) v += 0.5 * (z[i] ?? 0) * (kb[i] ?? 0) + (C[i] ?? 0) * (e[i] ?? 0) ** 2
      return v
    },
    gradient: (z) => {
      const { kb, e } = parts(z)
      const w = Float64Array.from({ length: n }, (_, i) => -2 * (C[i] ?? 0) * (e[i] ?? 0) * (y[i] ?? 1))
      const kw = times(K, w, n)
      const g = new Float64Array(n + 1)
      for (let i = 0; i < n; i++) g[i] = (kb[i] ?? 0) + (kw[i] ?? 0)
      g[n] = sumOf(w)
      return g
    },
  })
  return { objective, x0: new Float64Array(n + 1), K, y, C, options }
}

/** The pieces every squared-hinge dual shares: the dual objective, the primal at an expansion, the margins. */
function squaredQuantities(
  K: Float64Array,
  y: Float64Array,
  C: Float64Array,
): {
  readonly primal: (coef: ArrayLike<number>, intercept: number) => number
  readonly marginsOf: (coef: ArrayLike<number>, intercept: number) => Float64Array
  readonly dualOf: (alpha: ArrayLike<number>) => number
} {
  const n = y.length
  const marginsOf = (coef: ArrayLike<number>, intercept: number): Float64Array => {
    const kb = times(K, coef, n)
    return Float64Array.from({ length: n }, (_, i) => (y[i] ?? 1) * ((kb[i] ?? 0) + intercept))
  }
  return {
    marginsOf,
    primal: (coef, intercept) => {
      const kb = times(K, coef, n)
      let v = 0
      for (let i = 0; i < n; i++) {
        const m = 1 - (y[i] ?? 1) * ((kb[i] ?? 0) + intercept)
        v += 0.5 * (coef[i] ?? 0) * (kb[i] ?? 0) + (C[i] ?? 0) * Math.max(0, m) ** 2
      }
      return v
    },
    // sum alpha - 1/2 alpha' (Q + D) alpha, with Q = y y' K and D = diag(1 / (2 C)).
    dualOf: (alpha) => {
      const beta = Float64Array.from({ length: n }, (_, i) => (alpha[i] ?? 0) * (y[i] ?? 1))
      const kb = times(K, beta, n)
      let quad = 0
      for (let i = 0; i < n; i++) quad += (beta[i] ?? 0) * (kb[i] ?? 0) + (alpha[i] ?? 0) ** 2 / (2 * (C[i] ?? 1))
      return sumOf(alpha) - 0.5 * quad
    },
  }
}

/** The squared-hinge C-SVC dual solved by SMO: the C-SVC dual with `1 / (2 C_i)` on the diagonal and no upper bound. */
export function squaredSvcSmoDual(K: Float64Array, y: Float64Array, C: Float64Array, options: DualOptions): Dual {
  const n = y.length
  const Q = Array.from({ length: n }, (_, i) =>
    Array.from(
      { length: n },
      (_, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[i * n + j] ?? 0) + (i === j ? 1 / (2 * (C[i] ?? 1)) : 0),
    ),
  )
  const unbounded = new Float64Array(n).fill(UNBOUNDED)
  const problem: SmoProblem = { Q, p: new Float64Array(n).fill(-1), y, C: unbounded }
  const algorithm = smoAlgorithm(problem, { tol: options.tol, selection: options.selection, shrinking: false })
  const q = squaredQuantities(K, y, C)
  const expansion = (s: SmoState): Expansion => ({
    coef: s.alpha.map((a, i) => a * (y[i] ?? 1)),
    rho: smoRho(s.alpha, s.grad, y, unbounded),
  })
  return {
    n,
    algorithm,
    expansion,
    dual: (s) => q.dualOf(s.alpha),
    gap: (s) => {
      const e = expansion(s)
      return Math.max(0, q.primal(e.coef, -e.rho) - q.dualOf(s.alpha))
    },
    margins: (s) => {
      const e = expansion(s)
      return q.marginsOf(e.coef, -e.rho)
    },
  }
}

/** The state of the adapter: an SMO-shaped view of a generic solver's iterate, with the iterate itself. */
export type SmoothDualState = SmoState & {
  /** The iterate `(beta, c)` and the primal objective there. */
  readonly z: Float64Array
  readonly objective: number
  /** The solver's own state, to step it. */
  readonly inner: unknown
}

/**
 * A generic solver seen as an SMO-shaped algorithm: each state is the solver's own state with the dual quantities its
 * iterate implies (`view`), and the run is finished once the solver says so.
 */
export function smoothAlgorithm(
  inner: Algorithm<unknown>,
  name: string,
  view: (inner: unknown, iterations: number) => SmoothDualState,
): Algorithm<SmoState> {
  return {
    name,
    config: inner.config,
    init: (key) => view(inner.init(key), 0),
    step: (s, ctx) => {
      const state = s as SmoothDualState
      const next = inner.step(state.inner, ctx)
      const out = view(next, state.iterations + 1)
      return { ...out, finished: inner.done?.(next, out.iterations) !== undefined }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SmoState,
  }
}

/**
 * The squared-hinge SVC primal solved by a generic solver of the slot, seen as a dual: the model of a state is the
 * solver's own iterate `(beta, c)`, and its dual variables are the ones the iterate implies (`alpha_i = 2 C_i
 * max(0, 1 - y_i f_i)`, which satisfy the equality only at the optimum), so `solver/dual` and `solver/gap` close at
 * the optimum like SMO's.
 */
export function smoothSvcDual(problem: SvcPairProblem, inner: Algorithm<unknown>, q: { readonly name: string }): Dual {
  const { K, y, C } = problem
  const n = y.length
  const quantities = squaredQuantities(K, y, C)
  const view = (inner: unknown, iterations: number): SmoothDualState => {
    const r = inner as { readonly x: Float64Array; readonly value: number; readonly grad: Float64Array }
    const margins = quantities.marginsOf(r.x.subarray(0, n), r.x[n] ?? 0)
    const alpha = Float64Array.from({ length: n }, (_, i) => 2 * (C[i] ?? 0) * Math.max(0, 1 - (margins[i] ?? 0)))
    const beta = Float64Array.from({ length: n }, (_, i) => (alpha[i] ?? 0) * (y[i] ?? 1))
    const kb = times(K, beta, n)
    const grad = Float64Array.from(
      { length: n },
      (_, i) => (y[i] ?? 1) * (kb[i] ?? 0) + (alpha[i] ?? 0) / (2 * (C[i] ?? 1)) - 1,
    )
    let violation = 0
    for (const g of r.grad) violation = Math.max(violation, Math.abs(g))
    return {
      alpha,
      grad,
      active: Array.from({ length: n }, (_, i) => i),
      violation,
      iterations,
      counter: 0,
      unshrunk: true,
      finished: false,
      z: Float64Array.from(r.x),
      objective: r.value,
      inner,
    }
  }
  const algorithm = smoothAlgorithm(inner, q.name, view)
  const expansion = (s: SmoState): Expansion => {
    const z = (s as SmoothDualState).z
    return { coef: z.slice(0, n), rho: -(z[n] ?? 0) }
  }
  return {
    n,
    algorithm,
    expansion,
    dual: (s) => quantities.dualOf(s.alpha),
    gap: (s) => Math.max(0, (s as SmoothDualState).objective - quantities.dualOf(s.alpha)),
    margins: (s) => {
      const e = expansion(s)
      return quantities.marginsOf(e.coef, -e.rho)
    },
  }
}

/** The losses of the support vector classifier: the hinge (the C-SVC) and the squared hinge (the L2-loss SVM). */
export type SvcLoss = 'hinge' | 'squaredHinge'

const smo: ModeSpec<SvcPairProblem, SmoState, VectorParams> = {
  description:
    'sequential minimal optimisation of the dual (libsvm): exact for the hinge, and for the squared hinge with 1 / (2C) on the diagonal',
  space: dict({}),
  make: (p) => squaredSvcSmoDual(p.K, p.y, p.C, p.options).algorithm,
  params: (s) => ({ x: Array.from(s.alpha) }),
}

/**
 * The solvers of the support vector classifier as one slot: `smo` (default, libsvm's, for both losses) and every solver
 * of `smoothSolvers` on the squared-hinge primal over the kernel-expansion coefficients and the intercept (admissible
 * with `loss: 'squaredHinge'` only: the hinge is not smooth, so only the dual's SMO reaches it).
 */
export const svcSolvers: ModeTable<SvcPairProblem, VectorParams> = familySolvers<SvcPairProblem>({ smo }, (p) => p, {
  default: 'smo',
  description: 'how the support vector classifier is fitted',
})

/** Whether a solver of {@link svcSolvers} admits a loss. */
export function checkSolver(who: string, solver: { readonly kind?: unknown }, loss: SvcLoss): void {
  if (loss === 'hinge' && solver.kind !== 'smo' && solver.kind !== undefined)
    throw new OmaiError(
      'value',
      `${who}: the hinge loss is not smooth, only the 'smo' solver reaches it (use loss 'squaredHinge' for ${String(solver.kind)})`,
    )
}

/** Build the dual of one pair for a loss and a solver configuration. */
export function squaredDualFor(problem: SvcPairProblem, solver: SpaceValue, key: Key): Dual {
  const mode = svcSolvers.modeOf(solver).name
  if (mode === 'smo') return squaredSvcSmoDual(problem.K, problem.y, problem.C, problem.options)
  return smoothSvcDual(problem, svcSolvers.make(problem, solver, key), { name: `svc-${mode}` })
}

/** The squared-hinge SVC of a pair solved to its optimum by SMO: the expansion, for the cross-validation behind Platt. */
export function solveSquaredSvc(
  K: Float64Array,
  y: Float64Array,
  C: Float64Array,
  options: DualOptions,
): { readonly coef: Tensor; readonly bias: number } {
  const dual = squaredSvcSmoDual(K, y, C, options)
  const r = runAlgorithm(dual.algorithm, { maxSteps: Math.max(10_000, 100 * y.length * y.length) })
  const e = dual.expansion(r.state)
  return { coef: fromStorage(e.coef, [e.coef.length], 'float64'), bias: -e.rho }
}
