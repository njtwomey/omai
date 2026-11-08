// The squared-slack (L2) support vector machines with a smooth primal, and the solver slots of the two estimators that
// have them: the squared epsilon-insensitive regressor (`svr`) and the squared-slack support vector data description
// (`svdd`). In each the primal is a smooth objective of the kernel-expansion coefficients (and the intercept, or the
// squared radius), so every solver of `smoothSolvers` minimises it; its dual is the usual one with `1 / (2 C_i)` added to
// the diagonal and no upper bound, so SMO solves the same problem exactly. `svrSolvers` and `svddSolvers` are the tables
// of the two (SMO the default; the generic solvers admissible for the squared variant only). The one-class machine has
// no such variant (see `one-class.ts`). Not exported from the barrel.

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
import type { Algorithm, ModeSpec, ModeTable } from '../trace/index.ts'
import type { Dual, DualOptions, Expansion } from './_dual.ts'
import { type SmoothDualState, smoothAlgorithm, sumOf, times, UNBOUNDED } from './_squared.ts'

/** A smooth problem that also knows how to build the SMO dual that solves it. */
export interface SquaredProblem extends SmoothProblem {
  /** The SMO dual of the same problem. */
  smo(): Dual
}

/** The diagonal `1 / (2 C)` of a squared-slack dual; a zero penalty (a zero-weight sample) pins its variable at zero. */
const diagonal = (C: number): number => (C > 0 ? 1 / (2 * C) : 1e12)

/** The squared penalty term of a dual variable, `a^2 / (4 C)`. */
const slackTerm = (a: number, C: number): number => (a === 0 ? 0 : C > 0 ? (a * a) / (4 * C) : Infinity)

/** The inner solver's reached point, as the smooth adapters read it. */
const reached = (inner: unknown): { readonly x: Float64Array; readonly value: number; readonly grad: Float64Array } =>
  inner as { readonly x: Float64Array; readonly value: number; readonly grad: Float64Array }

const maxAbs = (v: ArrayLike<number>): number => {
  let m = 0
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i] ?? 0))
  return m
}

/** The losses of {@link svr}: the epsilon-insensitive (libsvm's) and its square (the L2-loss SVR, smooth). */
export type SvrLoss = 'epsilonInsensitive' | 'squaredEpsilonInsensitive'

/** The losses of {@link svdd}: the linear slack (Tax and Duin's) and the squared slack (the L2-SVDD, smooth). */
export type SvddLoss = 'slack' | 'squaredSlack'

/** Whether a solver of a slot admits a loss: only `smo` reaches the non-smooth one. */
export function checkSmoothSolver(
  who: string,
  solver: { readonly kind?: unknown },
  smooth: boolean,
  loss: string,
  squared: string,
): void {
  if (!smooth && solver.kind !== 'smo' && solver.kind !== undefined)
    throw new OmaiError(
      'value',
      `${who}: the ${loss} loss is not smooth, only the 'smo' solver reaches it (use loss '${squared}' for ${String(solver.kind)})`,
    )
}

// ------------------------------------------------------------------------------------------------------------------
// The squared epsilon-insensitive regressor
// ------------------------------------------------------------------------------------------------------------------

/** The squared epsilon-insensitive SVR primal of a sample, with its SMO dual. */
export interface SvrSquaredProblem extends SquaredProblem {
  readonly K: Float64Array
  readonly t: Float64Array
  readonly C: Float64Array
  readonly epsilon: number
  readonly options: DualOptions
}

/** The quantities of the squared SVR shared by its SMO dual and its smooth adapter. */
function svrQuantities(
  K: Float64Array,
  t: Float64Array,
  C: Float64Array,
  epsilon: number,
): {
  readonly residualsOf: (coef: ArrayLike<number>, intercept: number) => Float64Array
  readonly primal: (coef: ArrayLike<number>, intercept: number) => number
  readonly dualOf: (alpha: ArrayLike<number>) => number
} {
  const n = t.length
  const residualsOf = (coef: ArrayLike<number>, intercept: number): Float64Array => {
    const kb = times(K, coef, n)
    return Float64Array.from({ length: n }, (_, i) => (t[i] ?? 0) - (kb[i] ?? 0) - intercept)
  }
  return {
    residualsOf,
    primal: (coef, intercept) => {
      const kb = times(K, coef, n)
      let v = 0
      for (let i = 0; i < n; i++) {
        const e = Math.max(0, Math.abs((t[i] ?? 0) - (kb[i] ?? 0) - intercept) - epsilon)
        v += 0.5 * (coef[i] ?? 0) * (kb[i] ?? 0) + (C[i] ?? 0) * e * e
      }
      return v
    },
    // t' beta - epsilon sum (a + a*) - 1/2 beta' K beta - sum (a^2 + a*^2) / (4 C), with beta = a - a*.
    dualOf: (alpha) => {
      const beta = Float64Array.from({ length: n }, (_, i) => (alpha[i] ?? 0) - (alpha[i + n] ?? 0))
      const kb = times(K, beta, n)
      let v = -epsilon * sumOf(alpha)
      for (let i = 0; i < n; i++)
        v +=
          (beta[i] ?? 0) * (t[i] ?? 0) -
          0.5 * (beta[i] ?? 0) * (kb[i] ?? 0) -
          slackTerm(alpha[i] ?? 0, C[i] ?? 0) -
          slackTerm(alpha[i + n] ?? 0, C[i] ?? 0)
      return v
    },
  }
}

/** The squared epsilon-insensitive SVR primal over `(beta, c)`: the objective, its analytic gradient, a zero start. */
export function squaredSvrPrimal(
  K: Float64Array,
  t: Float64Array,
  C: Float64Array,
  epsilon: number,
  options: DualOptions,
): SvrSquaredProblem {
  const n = t.length
  const q = svrQuantities(K, t, C, epsilon)
  const objective = analyticObjective({
    value: (z) => q.primal(z.subarray(0, n), z[n] ?? 0),
    gradient: (z) => {
      const b = z.subarray(0, n)
      const kb = times(K, b, n)
      const r = q.residualsOf(b, z[n] ?? 0)
      // d/df_i of C_i max(0, |r_i| - epsilon)^2 with r_i = t_i - f_i.
      const w = Float64Array.from({ length: n }, (_, i) => {
        const e = Math.max(0, Math.abs(r[i] ?? 0) - epsilon)
        return -2 * (C[i] ?? 0) * e * Math.sign(r[i] ?? 0)
      })
      const kw = times(K, w, n)
      const g = new Float64Array(n + 1)
      for (let i = 0; i < n; i++) g[i] = (kb[i] ?? 0) + (kw[i] ?? 0)
      g[n] = sumOf(w)
      return g
    },
  })
  return {
    objective,
    x0: new Float64Array(n + 1),
    K,
    t,
    C,
    epsilon,
    options,
    smo: () => squaredSvrSmoDual(K, t, C, epsilon, options),
  }
}

/** The squared epsilon-insensitive SVR dual solved by SMO: the epsilon-SVR dual with `1 / (2 C_i)` on the diagonal and no bound. */
export function squaredSvrSmoDual(
  K: Float64Array,
  t: Float64Array,
  C: Float64Array,
  epsilon: number,
  options: DualOptions,
): Dual {
  const n = t.length
  const m = 2 * n
  const y = Float64Array.from({ length: m }, (_, i) => (i < n ? 1 : -1))
  const Q = Array.from({ length: m }, (_, i) =>
    Array.from(
      { length: m },
      (_, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[(i % n) * n + (j % n)] ?? 0) + (i === j ? diagonal(C[i % n] ?? 0) : 0),
    ),
  )
  const p = Float64Array.from({ length: m }, (_, i) => (i < n ? epsilon - (t[i] ?? 0) : epsilon + (t[i - n] ?? 0)))
  const unbounded = new Float64Array(m).fill(UNBOUNDED)
  const problem: SmoProblem = { Q, p, y, C: unbounded }
  const algorithm = smoAlgorithm(problem, { tol: options.tol, selection: options.selection, shrinking: false })
  const q = svrQuantities(K, t, C, epsilon)
  const expansion = (s: SmoState): Expansion => ({
    coef: Float64Array.from({ length: n }, (_, i) => (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0)),
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
      return q.residualsOf(e.coef, -e.rho)
    },
    tube: () => epsilon,
  }
}

/**
 * The squared epsilon-insensitive SVR primal solved by a generic solver of the slot, seen as a dual: the model of a
 * state is the solver's own iterate `(beta, c)`, and its dual variables are the ones the iterate implies
 * (`alpha_i = 2 C_i max(0, r_i - epsilon)`, `alpha*_i = 2 C_i max(0, -r_i - epsilon)`), so `solver/dual` and
 * `solver/gap` close at the optimum like SMO's.
 */
export function smoothSvrDual(problem: SvrSquaredProblem, inner: Algorithm<unknown>, name: string): Dual {
  const { K, t, C, epsilon } = problem
  const n = t.length
  const q = svrQuantities(K, t, C, epsilon)
  const view = (state: unknown, iterations: number): SmoothDualState => {
    const r = reached(state)
    const res = q.residualsOf(r.x.subarray(0, n), r.x[n] ?? 0)
    const alpha = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      const e = Math.max(0, Math.abs(res[i] ?? 0) - epsilon)
      alpha[(res[i] ?? 0) >= 0 ? i : i + n] = 2 * (C[i] ?? 0) * e
    }
    const beta = Float64Array.from({ length: n }, (_, i) => (alpha[i] ?? 0) - (alpha[i + n] ?? 0))
    const kb = times(K, beta, n)
    const grad = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      grad[i] = (kb[i] ?? 0) + (alpha[i] ?? 0) * diagonal(C[i] ?? 0) + epsilon - (t[i] ?? 0)
      grad[i + n] = -(kb[i] ?? 0) + (alpha[i + n] ?? 0) * diagonal(C[i] ?? 0) + epsilon + (t[i] ?? 0)
    }
    return {
      alpha,
      grad,
      active: Array.from({ length: 2 * n }, (_, i) => i),
      violation: maxAbs(r.grad),
      iterations,
      counter: 0,
      unshrunk: true,
      finished: false,
      z: Float64Array.from(r.x),
      objective: r.value,
      inner: state,
    }
  }
  const expansion = (s: SmoState): Expansion => {
    const z = (s as SmoothDualState).z
    return { coef: z.slice(0, n), rho: -(z[n] ?? 0) }
  }
  return {
    n,
    algorithm: smoothAlgorithm(inner, name, view),
    expansion,
    dual: (s) => q.dualOf(s.alpha),
    gap: (s) => Math.max(0, (s as SmoothDualState).objective - q.dualOf(s.alpha)),
    margins: (s) => {
      const e = expansion(s)
      return q.residualsOf(e.coef, -e.rho)
    },
    tube: () => epsilon,
  }
}

const svrSmo: ModeSpec<SvrSquaredProblem, SmoState, VectorParams> = {
  description:
    'sequential minimal optimisation of the dual (libsvm): exact for the epsilon-insensitive loss, and for its square with 1 / (2C) on the diagonal',
  space: dict({}),
  make: (p) => p.smo().algorithm,
  params: (s) => ({ x: Array.from(s.alpha) }),
}

/**
 * The solvers of the support vector regressor as one slot: `smo` (default, libsvm's, for both losses) and every solver
 * of `smoothSolvers` on the squared epsilon-insensitive primal over the kernel-expansion coefficients and the intercept
 * (admissible with `loss: 'squaredEpsilonInsensitive'` only: the epsilon-insensitive loss is not smooth, so only the
 * dual's SMO reaches it).
 */
export const svrSolvers: ModeTable<SvrSquaredProblem, VectorParams> = familySolvers<SvrSquaredProblem>(
  { smo: svrSmo },
  (p) => p,
  { default: 'smo', description: 'how the support vector regressor is fitted' },
)

/** The dual of the squared SVR for a solver configuration: SMO's, or a generic solver seen as a dual. */
export function squaredSvrDualFor(problem: SvrSquaredProblem, solver: SpaceValue, key: Key): Dual {
  const mode = svrSolvers.modeOf(solver).name
  if (mode === 'smo') return problem.smo()
  return smoothSvrDual(problem, svrSolvers.make(problem, solver, key), `svr-${mode}`)
}

// ------------------------------------------------------------------------------------------------------------------
// The squared-slack support vector data description
// ------------------------------------------------------------------------------------------------------------------

/** The squared-slack SVDD primal, with its SMO dual. */
export interface SvddSquaredProblem extends SquaredProblem {
  readonly K: Float64Array
  readonly C: number
  readonly options: DualOptions
}

/** The quantities of the squared-slack SVDD shared by its SMO dual and its smooth adapter. */
function svddQuantities(
  K: Float64Array,
  C: number,
  n: number,
): {
  readonly diag: Float64Array
  /** The squared feature-space distance of every sample to the centre `sum_j coef_j phi(x_j)`. */
  readonly distances: (coef: ArrayLike<number>) => {
    readonly d: Float64Array
    readonly kb: Float64Array
    readonly q: number
  }
  readonly primal: (coef: ArrayLike<number>, radius2: number) => number
  readonly dualOf: (alpha: ArrayLike<number>) => number
} {
  const diag = Float64Array.from({ length: n }, (_, i) => K[i * n + i] ?? 0)
  const distances = (coef: ArrayLike<number>): { d: Float64Array; kb: Float64Array; q: number } => {
    const kb = times(K, coef, n)
    let q = 0
    for (let i = 0; i < n; i++) q += (coef[i] ?? 0) * (kb[i] ?? 0)
    return { d: Float64Array.from({ length: n }, (_, i) => (diag[i] ?? 0) - 2 * (kb[i] ?? 0) + q), kb, q }
  }
  return {
    diag,
    distances,
    primal: (coef, radius2) => {
      const { d } = distances(coef)
      let v = radius2
      for (let i = 0; i < n; i++) v += C * Math.max(0, (d[i] ?? 0) - radius2) ** 2
      return v
    },
    // sum a_i K_ii - a' K a - sum a_i^2 / (4 C).
    dualOf: (alpha) => {
      const kb = times(K, alpha, n)
      let v = 0
      for (let i = 0; i < n; i++)
        v += (alpha[i] ?? 0) * (diag[i] ?? 0) - (alpha[i] ?? 0) * (kb[i] ?? 0) - slackTerm(alpha[i] ?? 0, C)
      return v
    },
  }
}

/** The squared-slack SVDD primal over `(centre coefficients, squared radius)`: the objective, its analytic gradient, a centroid start. */
export function squaredSvddPrimal(K: Float64Array, C: number, options: DualOptions): SvddSquaredProblem {
  const n = Math.round(Math.sqrt(K.length))
  const q = svddQuantities(K, C, n)
  const objective = analyticObjective({
    value: (z) => q.primal(z.subarray(0, n), z[n] ?? 0),
    gradient: (z) => {
      const b = z.subarray(0, n)
      const { d, kb } = q.distances(b)
      const slack = Float64Array.from({ length: n }, (_, i) => Math.max(0, (d[i] ?? 0) - (z[n] ?? 0)))
      const total = sumOf(slack)
      const ks = times(K, slack, n)
      const g = new Float64Array(n + 1)
      for (let i = 0; i < n; i++) g[i] = 4 * C * (total * (kb[i] ?? 0) - (ks[i] ?? 0))
      g[n] = 1 - 2 * C * total
      return g
    },
  })
  const x0 = new Float64Array(n + 1)
  x0.fill(1 / n, 0, n)
  return { objective, x0, K, C, options, smo: () => squaredSvddSmoDual(K, C, options) }
}

/** The squared-slack SVDD dual solved by SMO: the SVDD dual with `1 / (2C)` on the diagonal and no upper bound. */
export function squaredSvddSmoDual(K: Float64Array, C: number, options: DualOptions): Dual {
  const n = Math.round(Math.sqrt(K.length))
  const q = svddQuantities(K, C, n)
  const ones = new Float64Array(n).fill(1)
  const alpha0 = new Float64Array(n)
  alpha0[0] = 1
  const Q = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 2 * (K[i * n + j] ?? 0) + (i === j ? diagonal(C) : 0)),
  )
  const unbounded = new Float64Array(n).fill(UNBOUNDED)
  const algorithm = smoAlgorithm(
    { Q, p: q.diag.map((v) => -v), y: ones, C: unbounded, alpha0 },
    { tol: options.tol, selection: options.selection, shrinking: false },
  )
  const expansion = (s: SmoState): Expansion => ({
    coef: Float64Array.from(s.alpha),
    rho: smoRho(s.alpha, s.grad, ones, unbounded),
  })
  const scores = (e: Expansion): { readonly scores: Float64Array; readonly radius2: number } => {
    const { d, q: quad } = q.distances(e.coef)
    const radius2 = quad - e.rho
    return { scores: d.map((v) => radius2 - v), radius2 }
  }
  return {
    n,
    algorithm,
    expansion,
    dual: (s) => q.dualOf(s.alpha),
    gap: (s) => {
      const e = expansion(s)
      return Math.max(0, q.primal(e.coef, scores(e).radius2) - q.dualOf(s.alpha))
    },
    margins: (s) => scores(expansion(s)).scores,
  }
}

/**
 * The squared-slack SVDD primal solved by a generic solver of the slot, seen as a dual: the model of a state is the
 * solver's own iterate (the centre's coefficients and the squared radius), and its dual variables are the ones the
 * iterate implies (`alpha_i = 2 C max(0, d_i - R^2)`), so `solver/dual` and `solver/gap` close at the optimum like SMO's.
 */
export function smoothSvddDual(problem: SvddSquaredProblem, inner: Algorithm<unknown>, name: string): Dual {
  const { K, C } = problem
  const n = Math.round(Math.sqrt(K.length))
  const q = svddQuantities(K, C, n)
  const view = (state: unknown, iterations: number): SmoothDualState => {
    const r = reached(state)
    const { d } = q.distances(r.x.subarray(0, n))
    const alpha = Float64Array.from({ length: n }, (_, i) => 2 * C * Math.max(0, (d[i] ?? 0) - (r.x[n] ?? 0)))
    const ka = times(K, alpha, n)
    const grad = Float64Array.from(
      { length: n },
      (_, i) => 2 * (ka[i] ?? 0) + (alpha[i] ?? 0) * diagonal(C) - (q.diag[i] ?? 0),
    )
    return {
      alpha,
      grad,
      active: Array.from({ length: n }, (_, i) => i),
      violation: maxAbs(r.grad),
      iterations,
      counter: 0,
      unshrunk: true,
      finished: false,
      z: Float64Array.from(r.x),
      objective: r.value,
      inner: state,
    }
  }
  // The centre `b` with squared radius R^2 is the expansion with rho = b' K b - R^2.
  const expansion = (s: SmoState): Expansion => {
    const z = (s as SmoothDualState).z
    const b = z.slice(0, n)
    return { coef: b, rho: q.distances(b).q - (z[n] ?? 0) }
  }
  return {
    n,
    algorithm: smoothAlgorithm(inner, name, view),
    expansion,
    dual: (s) => q.dualOf(s.alpha),
    gap: (s) => Math.max(0, (s as SmoothDualState).objective - q.dualOf(s.alpha)),
    margins: (s) => {
      const z = (s as SmoothDualState).z
      const { d } = q.distances(z.subarray(0, n))
      return Float64Array.from({ length: n }, (_, i) => (z[n] ?? 0) - (d[i] ?? 0))
    },
  }
}

const svddSmo: ModeSpec<SvddSquaredProblem, SmoState, VectorParams> = {
  description:
    'sequential minimal optimisation of the dual (libsvm-style): exact for the linear slack, and for the squared slack with 1 / (2C) on the diagonal',
  space: dict({}),
  make: (p) => p.smo().algorithm,
  params: (s) => ({ x: Array.from(s.alpha) }),
}

/**
 * The solvers of the support vector data description as one slot: `smo` (default, for both losses) and every solver of
 * `smoothSolvers` on the squared-slack primal over the centre's kernel-expansion coefficients and the squared radius
 * (admissible with `loss: 'squaredSlack'` only: the linear slack is not smooth, so only the dual's SMO reaches it).
 */
export const svddSolvers: ModeTable<SvddSquaredProblem, VectorParams> = familySolvers<SvddSquaredProblem>(
  { smo: svddSmo },
  (p) => p,
  { default: 'smo', description: 'how the support vector data description is fitted' },
)

/** The dual of the squared-slack SVDD for a solver configuration: SMO's, or a generic solver seen as a dual. */
export function squaredSvddDualFor(problem: SvddSquaredProblem, solver: SpaceValue, key: Key): Dual {
  const mode = svddSolvers.modeOf(solver).name
  if (mode === 'smo') return problem.smo()
  return smoothSvddDual(problem, svddSolvers.make(problem, solver, key), `svdd-${mode}`)
}
