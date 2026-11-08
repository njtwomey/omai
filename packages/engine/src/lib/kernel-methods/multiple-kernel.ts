/**
 * Multiple kernel learning: a convex combination of base kernels chosen from the data. `alignmentWeights` learns the
 * weights by centred kernel-target alignment (Cortes, Mohri and Rostamizadeh), in closed form or by projected gradient on
 * the non-negative quadratic programme; `easyMkl` is EasyMKL (Aiolli and Donini), a margin criterion solved on two
 * simplices; `mklProblem` is the SimpleMKL-style criterion (the optimal ridge loss of the combined kernel) over a softmax
 * of the weights, a smooth problem for the solver slot, and `mklAlgorithm` runs any solver of `smoothSolvers` on it
 * logging the weights at every step. `fitMultipleKernel` runs one of them on base kernels and returns the combined kernel
 * as a `Kernel` object and its Gram matrix, ready for `kernelRidge`, the support vector machines or a Gaussian process.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import { type Kernel, scaleKernel, sumKernel } from '../kernels/index.ts'
import { eigvalsh, inv, solve } from '../numerics/linalg/index.ts'
import {
  type GradientPath,
  gradientContract,
  loggedSolver,
  projectSimplex,
  type SmoothProblem,
  type SolverPath,
  smoothSolverConfig,
  solverContract,
} from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  asTensor,
  div,
  exp,
  eye,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  stack,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, type Contract, defineContract, runAlgorithm, tensor } from '../trace/index.ts'

function gramsOf(grams: readonly TensorLike[], who: string): { flat: Float64Array[]; n: number } {
  if (grams.length === 0) throw new OmaiError('value', `${who}: need at least one kernel matrix`)
  const flat = grams.map((g) => {
    const t = primalOf(asTensor(g))
    if (t.shape.length !== 2 || t.shape[0] !== t.shape[1])
      throw new OmaiError('shape', `${who}: expected square Gram matrices, got [${t.shape.join(', ')}]`)
    return Float64Array.from(toFlat(t))
  })
  const n = Math.sqrt(flat[0]?.length ?? 0)
  if (flat.some((g) => g.length !== n * n)) throw new OmaiError('shape', `${who}: the Gram matrices differ in size`)
  return { flat, n }
}

function targets(y: TensorLike, n: number, who: string): Float64Array {
  const v = Float64Array.from(toFlat(primalOf(asTensor(y))))
  if (v.length !== n)
    throw new OmaiError('shape', `${who}: y has ${v.length} entries but the Gram matrices have ${n} rows`)
  return v
}

/** `H G H` for the centring matrix `H = I - 11'/n`. */
function centred(g: Float64Array, n: number): Float64Array {
  const rows = new Float64Array(n)
  const cols = new Float64Array(n)
  let all = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const v = g[i * n + j] ?? 0
      rows[i] = (rows[i] ?? 0) + v / n
      cols[j] = (cols[j] ?? 0) + v / n
      all += v / (n * n)
    }
  return g.map((v, idx) => v - (rows[Math.floor(idx / n)] ?? 0) - (cols[idx % n] ?? 0) + all)
}

const frobenius = (a: Float64Array, b: Float64Array): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0)
  return s
}

const projectedSimplex = (v: ArrayLike<number>): Float64Array =>
  Float64Array.from(toFlat(primalOf(projectSimplex(fromStorage(Float64Array.from(v), [v.length], 'float64')))))

const largestEigenvalue = (m: Float64Array, p: number): number =>
  maxOf(Array.from(toFlat(primalOf(eigvalsh(fromStorage(m, [p, p], 'float64'))))))

/** How {@link alignmentWeights} solves for the weights. */
export type AlignmentSolver = 'closed-form' | 'projected-gradient'

/** Options of {@link alignmentWeights}. */
export interface AlignmentWeightsOptions {
  /** `closed-form` (the unconstrained solution `M^(-1) a`, clipped at zero) or `projected-gradient` (the exact programme; default). */
  readonly method?: AlignmentSolver
  /** Centre the kernels first (default true). */
  readonly centred?: boolean
  /** Projected-gradient iterations (default 2000). */
  readonly steps?: number
}

/** The weights learned by alignment, with the alignment each kernel and the combination reach. */
export interface AlignmentWeights {
  /** Non-negative weights summing to one. */
  readonly weights: number[]
  /** The (centred) target alignment of the weighted combination. */
  readonly alignment: number
  /** The (centred) target alignment of each base kernel. */
  readonly individual: number[]
}

/**
 * Kernel weights by centred kernel-target alignment (ALIGNF). With centred kernels `K_k` and the target kernel `y y'`,
 * the combination `sum mu_k K_k` of greatest alignment has `mu = v / |v|_1` for `v` minimising `v'Mv - 2 a'v` over
 * `v >= 0`, where `M_kl = <K_k, K_l>_F` and `a_k = y' K_k y`. The programme is solved exactly by projected gradient, or,
 * as `closed-form`, by dropping the sign constraint (`v = (M + eps I)^(-1) a`) and clipping negative entries, which is
 * exact whenever that solution is non-negative.
 *
 * @param grams - Base Gram matrices, each `[n, n]`.
 * @param y - Targets `[n]` (labels in plus and minus one for classification).
 * @param options - The solver, whether to centre, the iteration budget.
 * @returns The weights and the alignments.
 * @throws OmaiError `value` for no kernels; `shape` for mismatched sizes.
 * @concept multiple-kernel-learning
 * @reference C. Cortes, M. Mohri and A. Rostamizadeh, Algorithms for learning kernels based on centered alignment
 *   (JMLR 13, 2012).
 * @example
 * ```ts
 * import { alignmentWeights } from 'omai-engine/kernel-methods'
 * const informative = [[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]]
 * const noise = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
 * alignmentWeights([informative, noise], [1, -1, -1, 1]).weights[0] > 0.9 // => true
 * ```
 */
export function alignmentWeights(
  grams: readonly TensorLike[],
  y: TensorLike,
  options: AlignmentWeightsOptions = {},
): AlignmentWeights {
  const { flat, n } = gramsOf(grams, 'alignmentWeights')
  const t = targets(y, n, 'alignmentWeights')
  const p = flat.length
  const k = options.centred === false ? flat : flat.map((g) => centred(g, n))
  const ideal = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) ideal[i * n + j] = (t[i] ?? 0) * (t[j] ?? 0)
  const target = options.centred === false ? ideal : centred(ideal, n)
  const m = new Float64Array(p * p)
  const a = new Float64Array(p)
  for (let r = 0; r < p; r++) {
    a[r] = frobenius(k[r] as Float64Array, target)
    for (let s = 0; s < p; s++) m[r * p + s] = frobenius(k[r] as Float64Array, k[s] as Float64Array)
  }
  const align = (mu: ArrayLike<number>): number => {
    const comb = new Float64Array(n * n)
    for (let r = 0; r < p; r++)
      for (let i = 0; i < comb.length; i++) comb[i] = (comb[i] ?? 0) + (mu[r] ?? 0) * ((k[r] as Float64Array)[i] ?? 0)
    return frobenius(comb, target) / Math.sqrt(frobenius(comb, comb) * frobenius(target, target))
  }
  const normalise = (v: ArrayLike<number>): number[] => {
    const clipped = Array.from(v, (x) => Math.max(x, 0))
    const total = clipped.reduce((s, x) => s + x, 0)
    return total > 0 ? clipped.map((x) => x / total) : new Array<number>(p).fill(1 / p)
  }
  let weights: number[]
  if ((options.method ?? 'projected-gradient') === 'closed-form') {
    const ridge = 1e-10 * (m.reduce((s, x, i) => s + (i % (p + 1) === 0 ? x : 0), 0) / p || 1)
    const mm = m.map((v, i) => v + (i % (p + 1) === 0 ? ridge : 0))
    const v = toFlat(primalOf(solve(fromStorage(mm, [p, p], 'float64'), fromStorage(a, [p, 1], 'float64'))))
    weights = normalise(v)
  } else {
    // Minimise v'Mv - 2a'v over v >= 0 with the step 1 / (2 lambda_max(M)), from the weights proportional to a.
    const step = 1 / (2 * largestEigenvalue(m, p))
    let v = Float64Array.from(a, (x) => Math.max(x, 0))
    for (let it = 0; it < (options.steps ?? 2000); it++) {
      const g = new Float64Array(p)
      for (let r = 0; r < p; r++) {
        let s = -2 * (a[r] ?? 0)
        for (let q = 0; q < p; q++) s += 2 * (m[r * p + q] ?? 0) * (v[q] ?? 0)
        g[r] = s
      }
      v = v.map((x, r) => Math.max(x - step * (g[r] ?? 0), 0))
    }
    weights = normalise(v)
  }
  return {
    weights,
    alignment: align(weights),
    individual: k.map((_, r) => align(Array.from({ length: p }, (_, s) => (s === r ? 1 : 0)))),
  }
}

/** Options of {@link easyMkl}. */
export interface EasyMklOptions {
  /** The regularisation `lambda >= 0` of the margin programme (default 0.1). */
  readonly lambda?: number
  /** Projected-gradient iterations (default 1000). */
  readonly steps?: number
}

/** The outcome of {@link easyMkl}. */
export interface EasyMklResult {
  /** Non-negative weights summing to one. */
  readonly weights: number[]
  /** The distribution over the examples `gamma` (one simplex per class). */
  readonly gamma: number[]
}

/**
 * EasyMKL (Aiolli and Donini) for two classes: with the base kernels normalised to unit trace, find the distribution
 * `gamma` over the examples, one simplex for each class, minimising `lambda |gamma|^2 + gamma' Y K Y gamma`, where `K` is
 * the sum of the base kernels and `Y = diag(y)`; the weight of kernel `r` is proportional to `gamma' Y K_r Y gamma`, its
 * margin-weighted similarity between the closest points of the two classes.
 *
 * @param grams - Base Gram matrices, each `[n, n]`.
 * @param y - Two-class labels `[n]` (any two values; the larger is positive).
 * @param options - `lambda` and the iteration budget.
 * @returns The weights and the example distribution.
 * @throws OmaiError `value` unless there are two classes.
 * @concept multiple-kernel-learning
 * @reference F. Aiolli and M. Donini, EasyMKL: a scalable multiple kernel learning algorithm (Neurocomputing 169, 2015).
 * @example
 * ```ts
 * import { easyMkl } from 'omai-engine/kernel-methods'
 * const informative = [[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]
 * const noise = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
 * easyMkl([informative, noise], [1, 1, -1, -1]).weights[0] > 0.5 // => true
 * ```
 */
export function easyMkl(grams: readonly TensorLike[], y: TensorLike, options: EasyMklOptions = {}): EasyMklResult {
  const { flat, n } = gramsOf(grams, 'easyMkl')
  const t = targets(y, n, 'easyMkl')
  const values = Array.from(new Set(t)).sort((x, z) => x - z)
  if (values.length !== 2) throw new OmaiError('value', `easyMkl: expected two classes, got ${values.length}`)
  const sign = Float64Array.from(t, (v) => (v === values[1] ? 1 : -1))
  const lambda = options.lambda ?? 0.1
  const normal = flat.map((g) => {
    let trace = 0
    for (let i = 0; i < n; i++) trace += g[i * n + i] ?? 0
    return g.map((v) => v / (trace || 1))
  })
  const total = new Float64Array(n * n)
  for (const g of normal) for (let i = 0; i < total.length; i++) total[i] = (total[i] ?? 0) + (g[i] ?? 0)
  const signed = (g: Float64Array): Float64Array =>
    g.map((v, idx) => v * (sign[Math.floor(idx / n)] ?? 0) * (sign[idx % n] ?? 0))
  const q = signed(total)
  const step = 1 / (2 * (lambda + largestEigenvalue(q, n)))
  const pos = Array.from({ length: n }, (_, i) => i).filter((i) => sign[i] === 1)
  const neg = Array.from({ length: n }, (_, i) => i).filter((i) => sign[i] === -1)
  let gamma = new Float64Array(n)
  for (const i of pos) gamma[i] = 1 / pos.length
  for (const i of neg) gamma[i] = 1 / neg.length
  for (let it = 0; it < (options.steps ?? 1000); it++) {
    const next = Float64Array.from(gamma)
    for (let i = 0; i < n; i++) {
      let s = 2 * lambda * (gamma[i] ?? 0)
      for (let j = 0; j < n; j++) s += 2 * (q[i * n + j] ?? 0) * (gamma[j] ?? 0)
      next[i] = (gamma[i] ?? 0) - step * s
    }
    for (const block of [pos, neg]) {
      const proj = projectedSimplex(block.map((i) => next[i] ?? 0))
      block.forEach((i, b) => {
        next[i] = proj[b] ?? 0
      })
    }
    gamma = next
  }
  const margin = normal.map((g) => {
    const s = signed(g)
    let v = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v += (gamma[i] ?? 0) * (s[i * n + j] ?? 0) * (gamma[j] ?? 0)
    return v
  })
  const sumMargin = margin.reduce((s, v) => s + v, 0)
  return { weights: margin.map((v) => (sumMargin > 0 ? v / sumMargin : 1 / margin.length)), gamma: Array.from(gamma) }
}

const softmaxOf = (u: ArrayLike<number>): number[] => {
  const top = maxOf(Array.from(u))
  const e = Array.from(u, (v) => Math.exp(v - top))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/**
 * The SimpleMKL-style criterion as a smooth problem for the solver slot. The kernel weights are `softmax(u)`, so the
 * simplex constraint disappears, and the objective is the optimal regularised ridge loss of the combined kernel,
 * `lambda y' (sum_k w_k K_k + lambda I)^(-1) y`, convex in the weights; its gradient is automatic differentiation.
 *
 * @param grams - Base Gram matrices, each `[n, n]`.
 * @param y - Targets `[n]`.
 * @param lambda - The ridge `lambda > 0` (default 1).
 * @returns The problem over `u`, starting at uniform weights.
 * @throws OmaiError `value` for a non-positive ridge.
 * @concept multiple-kernel-learning
 * @reference A. Rakotomamonjy, F. Bach, S. Canu and Y. Grandvalet, SimpleMKL (JMLR 9, 2008).
 * @example
 * ```ts
 * import { mklProblem } from 'omai-engine/kernel-methods'
 * mklProblem([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], [1, -1]).x0 // => [0, 0]
 * ```
 */
export function mklProblem(grams: readonly TensorLike[], y: TensorLike, lambda = 1): SmoothProblem {
  if (!(lambda > 0)) throw new OmaiError('value', `mklProblem: lambda must be positive, got ${lambda}`)
  const { flat, n } = gramsOf(grams, 'mklProblem')
  const t = targets(y, n, 'mklProblem')
  const stackK = stack(
    flat.map((g) => fromStorage(g, [n, n], 'float64')),
    0,
  )
  const yv = fromStorage(t, [n, 1], 'float64')
  return {
    objective: (u: unknown): Value => {
      const v = promoteFloat(u as Value)[0] as Value
      const e = exp(v)
      const w = div(e, sum(e))
      const k = sum(mul(reshape(w, [-1, 1, 1]), stackK), 0)
      const a = inv(add(k, mul(eye(n), lambda)))
      return mul(sum(mul(yv, matmul(a, yv))), lambda)
    },
    x0: new Array<number>(flat.length).fill(0),
  }
}

/** The paths of {@link mklAlgorithm}: the solver's and the weights. */
export type MklPath = SolverPath | GradientPath | 'mkl/weights'

/** The contract of {@link mklAlgorithm}. */
export const mklContract: Contract<unknown, MklPath> = defineContract({
  ...solverContract,
  ...gradientContract,
  'mkl/weights': tensor({ role: 'param', description: 'the kernel weights (softmax of the unknowns), summing to one' }),
})

/**
 * Multiple kernel learning as an algorithm over the solver slot: a solver of `smoothSolvers` (L-BFGS by default)
 * minimises {@link mklProblem}, logging the objective, gradient norm and step of the solver and `mkl/weights`, the
 * kernel weights, at every step.
 *
 * @param grams - Base Gram matrices.
 * @param y - Targets.
 * @param options - The ridge, the solver (a value of `smoothSolvers.space`) and the key.
 * @returns The algorithm; its state is the solver's, and its weights are `softmax(state.x)`.
 * @concept multiple-kernel-learning
 * @example
 * ```ts
 * import { mklAlgorithm } from 'omai-engine/kernel-methods'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = mklAlgorithm([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], [1, -1])
 * runAlgorithm(alg, { maxSteps: 20 }).steps > 0 // => true
 * ```
 */
export function mklAlgorithm(
  grams: readonly TensorLike[],
  y: TensorLike,
  options: { readonly lambda?: number; readonly solver?: SpaceValue; readonly key?: Key } = {},
): Algorithm<unknown, MklPath> {
  return loggedSolver<'mkl/weights'>(
    mklProblem(grams, y, options.lambda ?? 1),
    options.solver ?? smoothSolverConfig({ kind: 'lbfgs' }),
    options.key ?? makeKey(0),
    (x, ctx) => ctx.log('mkl/weights', softmaxOf(x)),
  )
}

/** How {@link fitMultipleKernel} chooses the weights. */
export type MklMethod = 'alignment' | 'alignment-closed-form' | 'easy' | 'simple'

/** Options of {@link fitMultipleKernel}. */
export interface MultipleKernelOptions {
  /** The criterion (default `alignment`). */
  readonly method?: MklMethod
  /** The ridge of the `simple` criterion and the `lambda` of `easy` (defaults 1 and 0.1). */
  readonly lambda?: number
  /** The solver of the `simple` criterion (default L-BFGS). */
  readonly solver?: SpaceValue
  readonly maxSteps?: number
}

/** A fitted combination of base kernels. */
export interface MultipleKernelFit {
  readonly method: MklMethod
  /** The weights, summing to one, in the order of the base kernels. */
  readonly weights: number[]
  /** The combined kernel `sum w_k k_k` as a `Kernel`, for any method that takes one. */
  readonly kernel: Kernel
  /** Its Gram matrix on the fitting rows, `[n, n]`. */
  readonly gram: Tensor
  /** The centred target alignment of the combined Gram matrix. */
  readonly alignment: number
}

/**
 * Learn the weights of base kernels from labelled data and return the combined kernel: the convex combination as a
 * `Kernel` object (so it plugs into `kernelRidge`, the `svm` and `gp` estimators) with its Gram matrix on the data.
 *
 * @param kernels - Base kernels.
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]` (labels for `easy`, which needs two classes).
 * @param options - The criterion and its knobs.
 * @returns The weights, the combined kernel and its Gram matrix.
 * @concept multiple-kernel-learning
 * @example
 * ```ts
 * import { fitMultipleKernel } from 'omai-engine/kernel-methods'
 * import { linearKernel, rbfKernel } from 'omai-engine/kernels'
 * const fit = fitMultipleKernel([linearKernel(), rbfKernel()], [[0], [1], [2], [3]], [1, 1, -1, -1])
 * fit.weights.length // => 2
 * ```
 */
export function fitMultipleKernel(
  kernels: readonly Kernel[],
  x: TensorLike,
  y: TensorLike,
  options: MultipleKernelOptions = {},
): MultipleKernelFit {
  if (kernels.length === 0) throw new OmaiError('value', 'fitMultipleKernel: need at least one kernel')
  const method = options.method ?? 'alignment'
  const grams = kernels.map((k) => primalOf(k.k(x)) as Tensor)
  let weights: number[]
  if (method === 'easy')
    weights = easyMkl(grams, y, { ...(options.lambda === undefined ? {} : { lambda: options.lambda }) }).weights
  else if (method === 'simple') {
    const run = runAlgorithm(
      mklAlgorithm(grams, y, {
        ...(options.lambda === undefined ? {} : { lambda: options.lambda }),
        ...(options.solver === undefined ? {} : { solver: options.solver }),
      }),
      {
        maxSteps: options.maxSteps ?? 200,
      },
    )
    weights = softmaxOf((run.state as { readonly x: Float64Array }).x)
  } else
    weights = alignmentWeights(grams, y, {
      method: method === 'alignment' ? 'projected-gradient' : 'closed-form',
    }).weights
  const combined = sumKernel(...kernels.map((k, i) => scaleKernel(k, Math.max(weights[i] ?? 0, 1e-300))))
  const gramValue = weights.reduce<Value | undefined>(
    (acc, w, i) => (acc === undefined ? mul(grams[i] as Tensor, w) : add(acc, mul(grams[i] as Tensor, w))),
    undefined,
  ) as Value
  const g = primalOf(gramValue) as Tensor
  return {
    method,
    weights,
    kernel: combined,
    gram: g,
    alignment: alignmentWeights([g], y, { method: 'closed-form' }).individual[0] ?? Number.NaN,
  }
}
