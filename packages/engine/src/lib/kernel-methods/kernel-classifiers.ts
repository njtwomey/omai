/**
 * Kernel classifiers by regularised empirical risk in the dual: kernel logistic regression (`kernelLogisticRegression`,
 * its objective as `kernelLogisticProblem` for the solver slot, with iteratively reweighted least squares in the dual as
 * the default mode and every solver of `smoothSolvers` as the others), the import vector machine, its sparse greedy
 * variant (`importVectorMachine`), the least-squares support vector machine (`lssvm`, closed form by default) and kernel
 * ridge classification (`kernelRidgeClassifier`, one-versus-rest regression on plus and minus one targets). The kernel
 * Fisher discriminant and the Gaussian-process classifier live in `learn` and `gp`; the support vector machines of
 * the hinge loss in `svm`.
 */

import { MdmlError } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import type { ResolvedKernel } from '../kernels/index.ts'
import { kernelSpace } from '../kernels/index.ts'
import type { IterativeEstimator, LearnData } from '../learn/index.ts'
import { encodeLabels, iterative } from '../learn/index.ts'
import { dgemm, solve } from '../numerics/linalg/index.ts'
import { type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import { permutation } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { asTensor, fromStorage, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type ClassificationSetup,
  classificationSetup,
  classifierVerbs,
  crossKernel,
  KERNEL_DEFAULTS,
  type KernelClassifierVerbs,
  type KernelOptions,
  type KernelProbabilityVerbs,
  probabilityVerbs,
  queryView,
  sigmoid,
  softplus,
} from './_classifier.ts'
import { infNorm, problemOf, type SlotState, type Smooth, type SolverChoice, slotted } from './_slot.ts'

const solveDense = (a: Float64Array, n: number, b: Float64Array, k: number): Float64Array =>
  Float64Array.from(toFlat(primalOf(solve(fromStorage(a, [n, n], 'float64'), fromStorage(b, [n, k], 'float64')))))

// ---------------------------------------------------------------------------------------------------------------------
// Kernel logistic regression
// ---------------------------------------------------------------------------------------------------------------------

/** Two-class targets as zeros and ones (the second sorted label is one). */
function binaryTargets(y: TensorLike, n: number, who: string): Float64Array {
  const enc = encodeLabels(y, who, n)
  if (enc.classes.length !== 2) throw new MdmlError('value', `${who}: expected two classes, got ${enc.classes.length}`)
  return Float64Array.from(enc.codes)
}

/** The kernel logistic objective on the unknowns `(alpha, b)`: `sum log(1 + exp(-y f)) + lambda/2 alpha' K alpha`. */
function logisticSmooth(k: Float64Array, t: Float64Array, n: number, lambda: number): Smooth {
  const margins = (x: ArrayLike<number>): Float64Array => {
    const f = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = x[n] ?? 0
      for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (x[j] ?? 0)
      f[i] = s
    }
    return f
  }
  return {
    value: (x) => {
      const f = margins(x)
      let v = 0
      for (let i = 0; i < n; i++) v += softplus(f[i] ?? 0) - (t[i] ?? 0) * (f[i] ?? 0)
      let q = 0
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q += (x[i] ?? 0) * (k[i * n + j] ?? 0) * (x[j] ?? 0)
      return v + 0.5 * lambda * q
    },
    gradient: (x) => {
      const f = margins(x)
      const r = new Float64Array(n)
      let sum = 0
      for (let i = 0; i < n; i++) {
        r[i] = sigmoid(f[i] ?? 0) - (t[i] ?? 0)
        sum += r[i] ?? 0
      }
      const g = new Float64Array(n + 1)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * ((r[j] ?? 0) + lambda * (x[j] ?? 0))
        g[i] = s
      }
      g[n] = sum
      return g
    },
    hessian: (x) => {
      const f = margins(x)
      const w = Float64Array.from(f, (v) => sigmoid(v) * (1 - sigmoid(v)))
      const kw = new Float64Array(n * n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) kw[i * n + j] = (k[i * n + j] ?? 0) * (w[j] ?? 0)
      const kwk = dgemm(kw, k, n, n, n)
      const h = new Float64Array((n + 1) * (n + 1))
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) h[i * (n + 1) + j] = (kwk[i * n + j] ?? 0) + lambda * (k[i * n + j] ?? 0)
        let kwi = 0
        for (let j = 0; j < n; j++) kwi += kw[i * n + j] ?? 0
        h[i * (n + 1) + n] = kwi
        h[n * (n + 1) + i] = kwi
        h[n * (n + 1) + n] = (h[n * (n + 1) + n] ?? 0) + (w[i] ?? 0)
      }
      return h
    },
  }
}

/** One iteratively reweighted least squares step in the dual from `(alpha, b)`: Newton's step on `f = K alpha + b`. */
function logisticIrls(k: Float64Array, t: Float64Array, n: number, lambda: number, x: ArrayLike<number>): Float64Array {
  const f = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = x[n] ?? 0
    for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (x[j] ?? 0)
    f[i] = s
  }
  const w = Float64Array.from(f, (v) => Math.max(sigmoid(v) * (1 - sigmoid(v)), 1e-10))
  const z = Float64Array.from(f, (v, i) => v + ((t[i] ?? 0) - sigmoid(v)) / (w[i] ?? 1))
  // (W K + lambda I) alpha + W 1 b = W z ;  1' W K alpha + (1' W 1) b = 1' W z
  const a = new Float64Array((n + 1) * (n + 1))
  const rhs = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) a[i * (n + 1) + j] = (w[i] ?? 0) * (k[i * n + j] ?? 0) + (i === j ? lambda : 0)
    a[i * (n + 1) + n] = w[i] ?? 0
    rhs[i] = (w[i] ?? 0) * (z[i] ?? 0)
    for (let j = 0; j < n; j++) a[n * (n + 1) + j] = (a[n * (n + 1) + j] ?? 0) + (w[i] ?? 0) * (k[i * n + j] ?? 0)
    a[n * (n + 1) + n] = (a[n * (n + 1) + n] ?? 0) + (w[i] ?? 0)
    rhs[n] = (rhs[n] ?? 0) + (w[i] ?? 0) * (z[i] ?? 0)
  }
  return solveDense(a, n + 1, rhs, 1)
}

/**
 * The kernel logistic regression objective as a smooth problem for the solver slot. The unknowns are the dual weights
 * `alpha` (`n`) and the intercept `b`; the objective is `sum_i [log(1 + exp(f_i)) - t_i f_i] + lambda/2 alpha' K alpha`
 * with `f = K alpha + b` and `t` the zero-one targets (the second sorted label is one), with its gradient and Hessian.
 *
 * @param k - Gram matrix `[n, n]`.
 * @param y - Two-class labels `[n]`.
 * @param lambda - Penalty `lambda > 0` on the squared RKHS norm (default 1).
 * @returns The problem, starting at zero.
 * @throws MdmlError `value` unless there are two classes and `lambda > 0`; `shape` for a mismatched Gram matrix.
 * @concept kernel-classifiers
 * @reference J. Zhu and T. Hastie, Kernel logistic regression and the import vector machine (J. Comp. Graph. Stat. 14, 2005).
 * @example
 * ```ts
 * import { kernelLogisticProblem } from 'mdml-engine/kernel-methods'
 * kernelLogisticProblem([[1, 0], [0, 1]], [0, 1], 0.5).x0 // => [0, 0, 0]
 * ```
 */
export function kernelLogisticProblem(k: TensorLike, y: TensorLike, lambda = 1): SmoothProblem {
  const g = primalOf(asTensor(k))
  const n = g.shape[0] ?? 0
  if (g.shape.length !== 2 || g.shape[1] !== n) throw new MdmlError('shape', 'kernelLogisticProblem: k must be square')
  if (!(lambda > 0)) throw new MdmlError('value', `kernelLogisticProblem: lambda must be positive, got ${lambda}`)
  const t = binaryTargets(y, n, 'kernelLogisticProblem')
  return problemOf(logisticSmooth(Float64Array.from(toFlat(g)), t, n, lambda), new Array<number>(n + 1).fill(0))
}

/** The fitted parameters of a kernel expansion classifier: plain JSON. */
export type ExpansionClassifierParams = {
  readonly kernel: ResolvedKernel
  readonly classes: number[]
  /** The centres of the expansion `[m][d]` (the training rows, or the import vectors). */
  readonly centres: number[][]
  readonly coef: number[]
  readonly intercept: number
  readonly nFeaturesIn: number
}

function expansionVerbs(p: ExpansionClassifierParams, h: KernelOptions, who: string): KernelProbabilityVerbs {
  return probabilityVerbs(
    p.classes,
    p.nFeaturesIn,
    who,
    (x) => {
      const q = queryView(x, p.nFeaturesIn, who)
      const m = p.centres.length
      const cross = m === 0 ? new Float64Array(0) : crossKernel(h, p.kernel, x, p.centres)
      const values = new Float64Array(q.n)
      for (let i = 0; i < q.n; i++) {
        let s = p.intercept
        for (let j = 0; j < m; j++) s += (cross[i * m + j] ?? 0) * (p.coef[j] ?? 0)
        values[i] = s
      }
      return { m: q.n, k: 1, values }
    },
    sigmoid,
  )
}

/** The hyperparameters of {@link kernelLogisticRegression}. */
export interface KernelLogisticHyper extends KernelOptions {
  /** The penalty `lambda` on the squared RKHS norm, `1 / C` of a primal logistic regression (default 1). */
  readonly lambda: number
  /** `'irls'` (Newton in the dual as iteratively reweighted least squares, default) or a value of `smoothSolvers.space`. */
  readonly solver: SolverChoice
  readonly tol: number
  readonly maxIter: number
}

/** The state of {@link kernelLogisticRegression}: the dual weights, then the intercept. */
export type KernelLogisticState = SlotState & {
  readonly theta: number[]
  readonly objective: number
  readonly gradient: number
}

const LOGISTIC_DEFAULTS: KernelLogisticHyper = {
  ...KERNEL_DEFAULTS,
  lambda: 1,
  solver: 'irls',
  tol: 1e-9,
  maxIter: 100,
}

const logisticSchema = {
  metrics: {
    objective: { description: 'the regularised negative log likelihood', direction: 'lower' as const },
    gradient: { description: 'the largest gradient entry', direction: 'lower' as const },
  },
  tensors: {},
}

/**
 * Kernel logistic regression (binary): the penalised maximum likelihood `f = K alpha + b` over a kernel's span. The
 * minimisation is the solver slot: iteratively reweighted least squares in the dual (`'irls'`, Newton's method, the
 * default) or any solver of `smoothSolvers`, all reaching the same optimum; with a linear kernel it is the L2-penalised
 * logistic regression with `C = 1 / lambda`. Multi-class problems go through the reductions of `learn`.
 *
 * @param hyper - `kernel`, `gamma`, `degree`, `coef0`, `lambda`, `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @concept kernel-classifiers
 * @reference J. Zhu and T. Hastie, Kernel logistic regression and the import vector machine (J. Comp. Graph. Stat. 14, 2005).
 * @index estimator Kernel logistic regression.
 * @example
 * ```ts
 * import { kernelLogisticRegression } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0.1, 0], [0, 0.2], [-0.1, 0], [2, 0], [0, 2.1], [-2, 0]]
 * const fit = kernelLogisticRegression({ gamma: 0.5, lambda: 0.1 }).fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * toFlat(fit.predict([[0.05, 0.05], [1.9, 0.3]])) // => [0, 1]
 * ```
 */
export function kernelLogisticRegression(
  hyper: Partial<KernelLogisticHyper> = {},
): IterativeEstimator<KernelLogisticHyper, KernelLogisticState, ExpansionClassifierParams, KernelProbabilityVerbs> {
  const who = 'kernelLogisticRegression'
  return iterative<KernelLogisticHyper, KernelLogisticState, ExpansionClassifierParams, KernelProbabilityVerbs>(
    {
      name: 'kernel-methods/kernel-logistic-regression',
      kind: 'classifier',
      defaults: LOGISTIC_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          lambda: real(Math.min(1e-6, h.lambda), Math.max(1e3, h.lambda), {
            log: true,
            default: h.lambda,
            label: 'penalty',
          }),
          ...kernelSpace(h),
          ...(h.solver === 'irls' ? {} : { solver: smoothSolvers.space }),
        }),
      algorithm: (data, h, key) => {
        const s = classificationSetup(data, h, who)
        if (!(h.lambda > 0)) throw new MdmlError('value', `${who}: lambda must be positive, got ${h.lambda}`)
        const t = binaryTargets(data.y ?? [], s.n, who)
        const smooth = logisticSmooth(s.gram, t, s.n, h.lambda)
        return slotted<KernelLogisticState>({
          name: 'kernel-logistic-regression',
          config: { lambda: h.lambda, kernel: s.resolved.kernel, gamma: s.resolved.gamma },
          schema: logisticSchema,
          problem: problemOf(smooth, new Array<number>(s.n + 1).fill(0)),
          smooth,
          solver: h.solver,
          direct: (x) => logisticIrls(s.gram, t, s.n, h.lambda, x),
          key,
          tol: h.tol,
          view: (r) => ({ theta: Array.from(r.x), objective: r.value, gradient: infNorm(r.grad) }),
          observe: (st) => ({ metrics: { objective: st.objective, gradient: st.gradient } }),
        })
      },
      maxSteps: (h) => h.maxIter,
      params: (st, data, h) => {
        const s = classificationSetup(data, h, who)
        return {
          kernel: s.resolved,
          classes: s.classes,
          centres: s.rows,
          coef: st.theta.slice(0, s.n),
          intercept: st.theta[s.n] ?? 0,
          nFeaturesIn: s.d,
        }
      },
      verbs: (p, h) => expansionVerbs(p, h, who),
      logs: { 'solver/objective': Log(), 'solver/gradient': Log() },
    },
    { ...LOGISTIC_DEFAULTS, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// The import vector machine
// ---------------------------------------------------------------------------------------------------------------------

/** The hyperparameters of {@link importVectorMachine}. */
export interface ImportVectorHyper extends KernelOptions {
  readonly lambda: number
  /** The most import vectors (default 20). */
  readonly nImport: number
  /** How many remaining points are tried at each step (default: all of them). */
  readonly candidates: number
  /** Stop when an addition lowers the objective by less than this fraction (default 1e-6). */
  readonly tol: number
}

/** The state of {@link importVectorMachine}: the chosen rows and their fit. */
export type ImportVectorState = {
  readonly step: number
  /** The indices of the import vectors, in the order chosen. */
  readonly imports: number[]
  readonly coef: number[]
  readonly intercept: number
  readonly objective: number
  readonly stopped: boolean
}

const IVM_DEFAULTS: ImportVectorHyper = { ...KERNEL_DEFAULTS, lambda: 1, nImport: 20, candidates: 1_000_000, tol: 1e-6 }

/** The objective and the Newton fit of the expansion over the columns `cols` of the Gram matrix. */
function importFit(
  s: ClassificationSetup,
  t: Float64Array,
  lambda: number,
  cols: readonly number[],
  start: readonly number[],
  start0: number,
  sweeps: number,
): { coef: number[]; intercept: number; objective: number } {
  const { n, gram } = s
  const m = cols.length
  let coef = Float64Array.from(start)
  let b = start0
  const f = (): Float64Array => {
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let v = b
      for (let c = 0; c < m; c++) v += (gram[i * n + (cols[c] ?? 0)] ?? 0) * (coef[c] ?? 0)
      out[i] = v
    }
    return out
  }
  const objective = (): number => {
    const fv = f()
    let v = 0
    for (let i = 0; i < n; i++) v += softplus(fv[i] ?? 0) - (t[i] ?? 0) * (fv[i] ?? 0)
    let q = 0
    for (let a = 0; a < m; a++)
      for (let c = 0; c < m; c++)
        q += (coef[a] ?? 0) * (gram[(cols[a] ?? 0) * n + (cols[c] ?? 0)] ?? 0) * (coef[c] ?? 0)
    return v + 0.5 * lambda * q
  }
  for (let sweep = 0; sweep < sweeps; sweep++) {
    const fv = f()
    const w = Float64Array.from(fv, (v) => Math.max(sigmoid(v) * (1 - sigmoid(v)), 1e-10))
    const z = Float64Array.from(fv, (v, i) => v + ((t[i] ?? 0) - sigmoid(v)) / (w[i] ?? 1))
    const p = m + 1
    const a = new Float64Array(p * p)
    const rhs = new Float64Array(p)
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(p)
      for (let c = 0; c < m; c++) row[c] = gram[i * n + (cols[c] ?? 0)] ?? 0
      row[m] = 1
      for (let u = 0; u < p; u++) {
        rhs[u] = (rhs[u] ?? 0) + (w[i] ?? 0) * (row[u] ?? 0) * (z[i] ?? 0)
        for (let v = 0; v < p; v++) a[u * p + v] = (a[u * p + v] ?? 0) + (w[i] ?? 0) * (row[u] ?? 0) * (row[v] ?? 0)
      }
    }
    for (let u = 0; u < m; u++)
      for (let v = 0; v < m; v++)
        a[u * p + v] =
          (a[u * p + v] ?? 0) + lambda * (gram[(cols[u] ?? 0) * n + (cols[v] ?? 0)] ?? 0) + (u === v ? 1e-10 : 0)
    const sol = solveDense(a, p, rhs, 1)
    coef = sol.slice(0, m)
    b = sol[m] ?? 0
  }
  return { coef: Array.from(coef), intercept: b, objective: objective() }
}

/**
 * The import vector machine (Zhu and Hastie), the sparse greedy variant of kernel logistic regression: the expansion
 * `f = sum_{j in S} alpha_j k(x_j, .) + b` is grown one import vector at a time, each step adding the training row whose
 * inclusion (after a few Newton steps of the refit) lowers the penalised negative log likelihood most, and stopping at
 * `nImport` vectors or when the gain falls below `tol`. The tape logs the objective and the number of imports.
 *
 * @param hyper - The kernel options, `lambda`, `nImport`, `candidates`, `tol`.
 * @returns The estimator.
 * @concept kernel-classifiers
 * @reference J. Zhu and T. Hastie, Kernel logistic regression and the import vector machine (J. Comp. Graph. Stat. 14, 2005).
 * @example
 * ```ts
 * import { importVectorMachine } from 'mdml-engine/kernel-methods'
 * const x = [[0.1, 0], [0, 0.2], [-0.1, 0], [2, 0], [0, 2.1], [-2, 0], [2, 2], [0.2, 0.1]]
 * const fit = importVectorMachine({ gamma: 0.5, nImport: 3 }).fit({ x, y: [0, 0, 0, 1, 1, 1, 1, 0] })
 * fit.params.centres.length // => 3
 * ```
 */
export function importVectorMachine(
  hyper: Partial<ImportVectorHyper> = {},
): IterativeEstimator<ImportVectorHyper, ImportVectorState, ExpansionClassifierParams, KernelProbabilityVerbs> {
  const who = 'importVectorMachine'
  return iterative<ImportVectorHyper, ImportVectorState, ExpansionClassifierParams, KernelProbabilityVerbs>(
    {
      name: 'kernel-methods/import-vector-machine',
      kind: 'classifier',
      defaults: IVM_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          lambda: real(Math.min(1e-6, h.lambda), Math.max(1e3, h.lambda), {
            log: true,
            default: h.lambda,
            label: 'penalty',
          }),
          nImport: int(1, Math.max(200, h.nImport), { default: h.nImport, label: 'import vectors' }),
          ...kernelSpace(h),
        }),
      algorithm: (data, h, key): Algorithm<ImportVectorState> => {
        const s = classificationSetup(data, h, who)
        if (!(h.lambda > 0)) throw new MdmlError('value', `${who}: lambda must be positive, got ${h.lambda}`)
        if (!Number.isInteger(h.nImport) || h.nImport < 1)
          throw new MdmlError('value', `${who}: nImport must be a positive integer`)
        const t = binaryTargets(data.y ?? [], s.n, who)
        const rng = resolveKey({ key }, 'importVectorMachine')
        const order = Array.from(permutation(rng, s.n).data)
        const intercept = importFit(s, t, h.lambda, [], [], 0, 8)
        return {
          name: 'import-vector-machine',
          config: { lambda: h.lambda, nImport: h.nImport, kernel: s.resolved.kernel, gamma: s.resolved.gamma },
          schema: channels({
            metrics: {
              objective: { description: 'the regularised negative log likelihood', direction: 'lower' },
              imports: { description: 'the number of import vectors' },
            },
          }),
          init: () => ({
            step: 0,
            imports: [],
            coef: [],
            intercept: intercept.intercept,
            objective: intercept.objective,
            stopped: false,
          }),
          step: (st) => {
            const pool = order.filter((j) => !st.imports.includes(j)).slice(0, h.candidates)
            let best: { j: number; objective: number } | undefined
            for (const j of pool) {
              const trial = importFit(s, t, h.lambda, [...st.imports, j], [...st.coef, 0], st.intercept, 2)
              if (best === undefined || trial.objective < best.objective) best = { j, objective: trial.objective }
            }
            if (best === undefined) return { ...st, step: st.step + 1, stopped: true }
            const cols = [...st.imports, best.j]
            const fit = importFit(s, t, h.lambda, cols, [...st.coef, 0], st.intercept, 12)
            const gain = (st.objective - fit.objective) / Math.max(Math.abs(st.objective), 1e-12)
            if (gain < h.tol && st.imports.length > 0) return { ...st, step: st.step + 1, stopped: true }
            return {
              step: st.step + 1,
              imports: cols,
              coef: fit.coef,
              intercept: fit.intercept,
              objective: fit.objective,
              stopped: false,
            }
          },
          done: (st) => (st.stopped || st.imports.length >= h.nImport ? 'converged' : undefined),
          observe: (st) => ({ metrics: { objective: st.objective, imports: st.imports.length } }),
          serialise: (st) => stateToJson(st),
          deserialise: (j) => stateFromJson(j) as ImportVectorState,
        }
      },
      maxSteps: (h) => h.nImport + 1,
      params: (st, data, h) => {
        const s = classificationSetup(data, h, who)
        return {
          kernel: s.resolved,
          classes: s.classes,
          centres: st.imports.map((j) => s.rows[j] as number[]),
          coef: st.coef,
          intercept: st.intercept,
          nFeaturesIn: s.d,
        }
      },
      verbs: (p, h) => expansionVerbs(p, h, who),
      logs: { 'solver/objective': Log(), 'solver/imports': Log() },
    },
    { ...IVM_DEFAULTS, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Least-squares classifiers
// ---------------------------------------------------------------------------------------------------------------------

/** Targets `T` (`n * k`) of plus and minus one: one column for two classes (the second is plus), one per class otherwise. */
function signTargets(codes: ArrayLike<number>, k: number): { t: Float64Array; cols: number } {
  const cols = k === 2 ? 1 : k
  const t = new Float64Array(codes.length * cols).fill(-1)
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i] ?? 0
    if (cols === 1) t[i] = c === 1 ? 1 : -1
    else t[i * cols + c] = 1
  }
  return { t, cols }
}

/** The least-squares classifier objective on `(A, b)`: `1/2 tr(A' K A) + C/2 |T - K A - 1 b'|^2`. */
function leastSquaresSmooth(
  k: Float64Array,
  t: Float64Array,
  n: number,
  cols: number,
  c: number,
  bias: boolean,
): Smooth {
  const p = n * cols + (bias ? cols : 0)
  const residual = (x: ArrayLike<number>): Float64Array => {
    const r = new Float64Array(n * cols)
    for (let i = 0; i < n; i++)
      for (let q = 0; q < cols; q++) {
        let s = bias ? (x[n * cols + q] ?? 0) : 0
        for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * (x[j * cols + q] ?? 0)
        r[i * cols + q] = s - (t[i * cols + q] ?? 0)
      }
    return r
  }
  return {
    value: (x) => {
      const r = residual(x)
      let v = 0
      for (const e of r) v += 0.5 * c * e * e
      for (let q = 0; q < cols; q++)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) v += 0.5 * (x[i * cols + q] ?? 0) * (k[i * n + j] ?? 0) * (x[j * cols + q] ?? 0)
      return v
    },
    gradient: (x) => {
      const r = residual(x)
      const g = new Float64Array(p)
      for (let i = 0; i < n; i++)
        for (let q = 0; q < cols; q++) {
          let s = 0
          for (let j = 0; j < n; j++) s += (k[i * n + j] ?? 0) * ((x[j * cols + q] ?? 0) + c * (r[j * cols + q] ?? 0))
          g[i * cols + q] = s
        }
      if (bias)
        for (let q = 0; q < cols; q++)
          for (let i = 0; i < n; i++) g[n * cols + q] = (g[n * cols + q] ?? 0) + c * (r[i * cols + q] ?? 0)
      return g
    },
    hessian: () => {
      const h = new Float64Array(p * p)
      const kk = dgemm(k, k, n, n, n)
      for (let q = 0; q < cols; q++) {
        for (let i = 0; i < n; i++) {
          let ki = 0
          for (let j = 0; j < n; j++) {
            h[(i * cols + q) * p + (j * cols + q)] = (k[i * n + j] ?? 0) + c * (kk[i * n + j] ?? 0)
            ki += k[i * n + j] ?? 0
          }
          if (bias) {
            h[(i * cols + q) * p + (n * cols + q)] = c * ki
            h[(n * cols + q) * p + (i * cols + q)] = c * ki
          }
        }
        if (bias) h[(n * cols + q) * p + (n * cols + q)] = c * n
      }
      return h
    },
  }
}

/** The closed form of the least-squares classifier: `[[0, 1'], [1, K + I/C]] [b; A] = [0; T]` (or without the bias row). */
function leastSquaresClosed(
  k: Float64Array,
  t: Float64Array,
  n: number,
  cols: number,
  c: number,
  bias: boolean,
): Float64Array {
  const m = n + (bias ? 1 : 0)
  const a = new Float64Array(m * m)
  const rhs = new Float64Array(m * cols)
  const off = bias ? 1 : 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) a[(i + off) * m + (j + off)] = (k[i * n + j] ?? 0) + (i === j ? 1 / c : 0)
    if (bias) {
      a[(i + 1) * m] = 1
      a[i + 1] = 1
    }
    for (let q = 0; q < cols; q++) rhs[(i + off) * cols + q] = t[i * cols + q] ?? 0
  }
  const sol = solveDense(a, m, rhs, cols)
  const out = new Float64Array(n * cols + (bias ? cols : 0))
  for (let i = 0; i < n; i++) for (let q = 0; q < cols; q++) out[i * cols + q] = sol[(i + off) * cols + q] ?? 0
  if (bias) for (let q = 0; q < cols; q++) out[n * cols + q] = sol[q] ?? 0
  return out
}

/** The hyperparameters of {@link lssvm}. */
export interface LeastSquaresSvmHyper extends KernelOptions {
  /** The weight `C` of the squared errors against the margin term (default 1); `1 / C` is the ridge. */
  readonly regularisation: number
  /** `'closed-form'` (the linear system, default) or a value of `smoothSolvers.space`. */
  readonly solver: SolverChoice
  readonly tol: number
  readonly maxIter: number
}

/** The state of a least-squares classifier: the dual weights `[n * cols]` then the biases. */
export type LeastSquaresState = SlotState & {
  readonly theta: number[]
  readonly objective: number
  readonly gradient: number
}

/** The fitted parameters of a least-squares kernel classifier: plain JSON. */
export type LeastSquaresParams = {
  readonly kernel: ResolvedKernel
  readonly classes: number[]
  readonly centres: number[][]
  /** The dual weights `[n][cols]`. */
  readonly dualCoef: number[][]
  readonly intercept: number[]
  readonly nFeaturesIn: number
}

function leastSquaresVerbs(p: LeastSquaresParams, h: KernelOptions, who: string): KernelClassifierVerbs {
  const cols = p.intercept.length
  return classifierVerbs(p.classes, p.nFeaturesIn, who, (x) => {
    const q = queryView(x, p.nFeaturesIn, who)
    const m = p.centres.length
    const cross = crossKernel(h, p.kernel, x, p.centres)
    const values = new Float64Array(q.n * cols)
    for (let i = 0; i < q.n; i++)
      for (let c = 0; c < cols; c++) {
        let s = p.intercept[c] ?? 0
        for (let j = 0; j < m; j++) s += (cross[i * m + j] ?? 0) * (p.dualCoef[j]?.[c] ?? 0)
        values[i * cols + c] = s
      }
    return { m: q.n, k: cols, values }
  })
}

function leastSquaresEstimator<
  H extends KernelOptions & { readonly solver: SolverChoice; readonly tol: number; readonly maxIter: number },
>(
  name: string,
  who: string,
  defaults: H,
  bias: boolean,
  weightOf: (h: H) => number,
  label: string,
  hyper: Partial<H>,
): IterativeEstimator<H, LeastSquaresState, LeastSquaresParams, KernelClassifierVerbs> {
  const fitOf = (data: LearnData, h: H): { s: ClassificationSetup; t: Float64Array; cols: number; c: number } => {
    const s = classificationSetup(data, h, who)
    const c = weightOf(h)
    if (!(c > 0) || !Number.isFinite(c))
      throw new MdmlError('value', `${who}: the regularisation must be positive, got ${label}`)
    if (s.classes.length < 2) throw new MdmlError('value', `${who}: need at least two classes`)
    const { t, cols } = signTargets(s.codes, s.classes.length)
    return { s, t, cols, c }
  }
  return iterative<H, LeastSquaresState, LeastSquaresParams, KernelClassifierVerbs>(
    {
      name,
      kind: 'classifier',
      defaults,
      hyperSpace: (h) =>
        dict({
          [label]: real(1e-6, 1e6, { log: true, default: (h as Record<string, unknown>)[label] as number, label }),
          ...kernelSpace(h),
          ...(h.solver === 'closed-form' ? {} : { solver: smoothSolvers.space }),
        }),
      algorithm: (data, h, key) => {
        const { s, t, cols, c } = fitOf(data, h)
        const smooth = leastSquaresSmooth(s.gram, t, s.n, cols, c, bias)
        const p = s.n * cols + (bias ? cols : 0)
        return slotted<LeastSquaresState>({
          name,
          config: { [label]: (h as Record<string, unknown>)[label] as number, bias, kernel: s.resolved.kernel },
          schema: logisticSchema,
          problem: problemOf(smooth, new Array<number>(p).fill(0)),
          smooth,
          solver: h.solver,
          direct: () => leastSquaresClosed(s.gram, t, s.n, cols, c, bias),
          key,
          tol: h.tol,
          view: (r) => ({ theta: Array.from(r.x), objective: r.value, gradient: infNorm(r.grad) }),
          observe: (st) => ({ metrics: { objective: st.objective, gradient: st.gradient } }),
        })
      },
      maxSteps: (h) => (h.solver === 'closed-form' ? 1 : h.maxIter),
      params: (st, data, h) => {
        const { s, cols } = fitOf(data, h)
        return {
          kernel: s.resolved,
          classes: s.classes,
          centres: s.rows,
          dualCoef: Array.from({ length: s.n }, (_, i) => st.theta.slice(i * cols, (i + 1) * cols)),
          intercept: bias ? st.theta.slice(s.n * cols) : new Array<number>(cols).fill(0),
          nFeaturesIn: s.d,
        }
      },
      verbs: (p, h) => leastSquaresVerbs(p, h, who),
      logs: { 'solver/objective': Log(), 'solver/gradient': Log() },
    },
    { ...defaults, ...hyper },
  )
}

/**
 * The least-squares support vector machine (Suykens and Vandewalle): the margin term `1/2 |w|^2` with squared errors
 * `C/2 sum (1 - y_i f(x_i))^2`, whose optimum is the solution of one linear system in the dual, `[[0, 1'], [1, K + I/C]]
 * [b; alpha] = [0; y]`. That closed form is the default mode of the solver slot; any solver of `smoothSolvers` reaches the
 * same `(alpha, b)`. Several classes are one-versus-rest.
 *
 * @param hyper - The kernel options, `regularisation` (`C`), `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @concept kernel-classifiers
 * @reference J. A. K. Suykens and J. Vandewalle, Least squares support vector machine classifiers (Neural Processing
 *   Letters 9, 1999).
 * @example
 * ```ts
 * import { lssvm } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0.1, 0], [0, 0.2], [-0.1, 0], [2, 0], [0, 2.1], [-2, 0]]
 * const fit = lssvm({ gamma: 0.5, regularisation: 10 }).fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * toFlat(fit.predict([[0.05, 0.05], [1.9, 0.3]])) // => [0, 1]
 * ```
 */
export function lssvm(
  hyper: Partial<LeastSquaresSvmHyper> = {},
): IterativeEstimator<LeastSquaresSvmHyper, LeastSquaresState, LeastSquaresParams, KernelClassifierVerbs> {
  return leastSquaresEstimator<LeastSquaresSvmHyper>(
    'kernel-methods/lssvm',
    'lssvm',
    { ...KERNEL_DEFAULTS, regularisation: 1, solver: 'closed-form', tol: 1e-9, maxIter: 200 },
    true,
    (h) => h.regularisation,
    'regularisation',
    hyper,
  )
}

/** The hyperparameters of {@link kernelRidgeClassifier}. */
export interface KernelRidgeClassifierHyper extends KernelOptions {
  /** The ridge `alpha > 0` of `(K + alpha I)^(-1) T` (default 1). */
  readonly alpha: number
  /** `'closed-form'` (default) or a value of `smoothSolvers.space`. */
  readonly solver: SolverChoice
  readonly tol: number
  readonly maxIter: number
}

/**
 * Kernel ridge classification: kernel ridge regression on targets of plus one for the class and minus one for the rest
 * (one column per class; one column for two classes), the label the largest output. It is the least-squares support
 * vector machine without a bias term with `C = 1 / alpha`; a linear kernel gives scikit-learn's `RidgeClassifier`
 * without an intercept.
 *
 * @param hyper - The kernel options, `alpha`, `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @concept kernel-classifiers
 * @example
 * ```ts
 * import { kernelRidgeClassifier } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [0.2, 0.1], [3, 3], [3.1, 2.8], [-3, 3], [-2.9, 3.2]]
 * const fit = kernelRidgeClassifier({ gamma: 0.3, alpha: 0.1 }).fit({ x, y: [0, 0, 1, 1, 2, 2] })
 * toFlat(fit.predict([[0.1, 0.1], [3, 3.1], [-3, 3]])) // => [0, 1, 2]
 * ```
 */
export function kernelRidgeClassifier(
  hyper: Partial<KernelRidgeClassifierHyper> = {},
): IterativeEstimator<KernelRidgeClassifierHyper, LeastSquaresState, LeastSquaresParams, KernelClassifierVerbs> {
  return leastSquaresEstimator<KernelRidgeClassifierHyper>(
    'kernel-methods/kernel-ridge-classifier',
    'kernelRidgeClassifier',
    { ...KERNEL_DEFAULTS, alpha: 1, solver: 'closed-form', tol: 1e-9, maxIter: 200 },
    false,
    (h) => 1 / h.alpha,
    'alpha',
    hyper,
  )
}
