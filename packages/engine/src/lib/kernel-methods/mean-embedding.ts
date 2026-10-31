/**
 * Kernel mean embeddings of distributions and what is computed with them: the empirical mean embedding and its inner
 * products (`meanEmbedding`, `embeddingInnerProduct`), the witness function of the maximum mean discrepancy
 * (`mmdWitness`), the conditional mean embedding as a regularised operator (`conditionalMeanEmbedding`, whose
 * expectation of the output is kernel ridge regression), the kernel Bayes rule at its basic level (`kernelBayesRule`),
 * and the kernel Stein discrepancy (`steinKernelGram`, `kernelSteinDiscrepancy`) with the Stein kernel built by automatic
 * differentiation for any score function, and its goodness-of-fit test by the multinomial wild bootstrap
 * (`steinGoodnessOfFit`). Kernel herding, which picks points to match a mean embedding, is `learn/coresets.ts`.
 */

import { grad, jacfwd, vmap } from '../autodiff/index.ts'
import { MdmlError, type TestResult } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import type { Kernel } from '../kernels/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { testResult } from '../stats/tests/index.ts'
import {
  add,
  asTensor,
  diagonal,
  fromStorage,
  mul,
  primalOf,
  reshape,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'

function rowsOf(x: TensorLike, who: string): Tensor {
  const t = primalOf(asTensor(x))
  if (t.shape.length === 1) return reshape(t, [t.shape[0] ?? 0, 1]) as Tensor
  if (t.shape.length !== 2) throw new MdmlError('shape', `${who}: expected points [n, d], got [${t.shape.join(', ')}]`)
  return t
}

const gramOf = (kernel: Kernel, x: Tensor, y?: Tensor): Float64Array =>
  Float64Array.from(toFlat(primalOf(kernel.k(x, y ?? x))))

function solveDense(a: Float64Array, n: number, b: Float64Array, k: number): Float64Array {
  return Float64Array.from(
    toFlat(primalOf(solve(fromStorage(a, [n, n], 'float64'), fromStorage(b, [n, k], 'float64')))),
  )
}

/** An empirical kernel mean embedding `mu(.) = sum_i w_i k(x_i, .)`. */
export interface MeanEmbedding {
  readonly kernel: Kernel
  /** The points `[n, d]`. */
  readonly points: Tensor
  /** The weights `[n]` (uniform `1/n` for an empirical mean). */
  readonly weights: Float64Array
  /** The embedding evaluated at points `[m, d]`, `[m]`. */
  evaluate(z: TensorLike): Tensor
  /** `|mu|^2 = w' K w`, the squared RKHS norm. */
  normSquared(): number
}

/**
 * The empirical kernel mean embedding of a sample, `mu_P(.) = (1/n) sum_i k(x_i, .)` (or with the given weights).
 *
 * @param kernel - The kernel.
 * @param x - Sample points `[n, d]`.
 * @param weights - Weights `[n]` (default uniform).
 * @returns The embedding, evaluable at any point.
 * @throws MdmlError `shape` for weights of the wrong length.
 * @concept kernel-mean-embedding
 * @reference A. Smola, A. Gretton, L. Song and B. Scholkopf, A Hilbert space embedding for distributions (ALT 2007).
 * @example
 * ```ts
 * import { meanEmbedding } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(meanEmbedding(rbfKernel(), [[0], [0]]).evaluate([[0]])) // => [1]
 * ```
 */
export function meanEmbedding(kernel: Kernel, x: TensorLike, weights?: ArrayLike<number>): MeanEmbedding {
  const points = rowsOf(x, 'meanEmbedding')
  const n = points.shape[0] ?? 0
  if (weights !== undefined && weights.length !== n) {
    throw new MdmlError('shape', `meanEmbedding: ${weights.length} weights for ${n} points`)
  }
  const w = weights === undefined ? new Float64Array(n).fill(1 / n) : Float64Array.from(weights)
  return {
    kernel,
    points,
    weights: w,
    evaluate: (z) => {
      const q = rowsOf(z, 'meanEmbedding')
      const m = q.shape[0] ?? 0
      const k = gramOf(kernel, points, q)
      const out = new Float64Array(m)
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j] = (out[j] ?? 0) + (w[i] ?? 0) * (k[i * m + j] ?? 0)
      return fromStorage(out, [m], 'float64')
    },
    normSquared: () =>
      embeddingInnerProduct(
        { kernel, points, weights: w } as MeanEmbedding,
        { kernel, points, weights: w } as MeanEmbedding,
      ),
  }
}

/**
 * The RKHS inner product `<mu_a, mu_b> = sum_ij a_i b_j k(x_i, y_j)` of two embeddings (of the same kernel).
 *
 * @param a - An embedding.
 * @param b - Another embedding.
 * @returns The inner product.
 * @concept kernel-mean-embedding
 * @example
 * ```ts
 * import { embeddingInnerProduct, meanEmbedding } from 'mdml-engine/kernel-methods'
 * import { linearKernel } from 'mdml-engine/kernels'
 * embeddingInnerProduct(meanEmbedding(linearKernel(), [[1], [3]]), meanEmbedding(linearKernel(), [[2]])) // => 4
 * ```
 */
export function embeddingInnerProduct(a: MeanEmbedding, b: MeanEmbedding): number {
  const k = gramOf(a.kernel, a.points, b.points)
  const m = b.weights.length
  let s = 0
  for (let i = 0; i < a.weights.length; i++)
    for (let j = 0; j < m; j++) s += (a.weights[i] ?? 0) * (k[i * m + j] ?? 0) * (b.weights[j] ?? 0)
  return s
}

/** The MMD witness function of two samples. */
export interface MmdWitness {
  /** `f(z) = mu_P(z) - mu_Q(z)`, `[m]`: positive where the first sample has more mass (in the kernel's sense). */
  readonly witness: (z: TensorLike) => Tensor
  /** The biased squared maximum mean discrepancy `|mu_P - mu_Q|^2`. */
  readonly mmd2: number
}

/**
 * The witness function of the maximum mean discrepancy: `f = mu_P - mu_Q`, the function in the unit ball of the RKHS
 * (after normalising) on which the two samples' means differ most, whose value at `z` shows where the distributions
 * differ and in which direction; its squared norm is the biased MMD squared.
 *
 * @param kernel - The kernel.
 * @param x - First sample `[n, d]`.
 * @param y - Second sample `[m, d]`.
 * @returns The witness and the squared discrepancy.
 * @concept kernel-mean-embedding
 * @reference A. Gretton, K. Borgwardt, M. Rasch, B. Scholkopf and A. Smola, A kernel two-sample test (JMLR 13, 2012).
 * @example
 * ```ts
 * import { mmdWitness } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { toFlat } from 'mdml-engine/tensor'
 * const w = mmdWitness(rbfKernel(), [[0], [0.1]], [[3], [3.1]])
 * (toFlat(w.witness([[0], [3]]))[0] ?? 0) > 0 // => true
 * ```
 */
export function mmdWitness(kernel: Kernel, x: TensorLike, y: TensorLike): MmdWitness {
  const p = meanEmbedding(kernel, x)
  const q = meanEmbedding(kernel, y)
  return {
    witness: (z) => {
      const a = toFlat(p.evaluate(z))
      const b = toFlat(q.evaluate(z))
      return fromStorage(
        Float64Array.from(a, (v, i) => v - (b[i] ?? 0)),
        [a.length],
        'float64',
      )
    },
    mmd2: p.normSquared() + q.normSquared() - 2 * embeddingInnerProduct(p, q),
  }
}

/** A conditional mean embedding `mu_{Y|x} = sum_i beta_i(x) l(y_i, .)` as an operator estimated from a joint sample. */
export interface ConditionalMeanEmbedding {
  /** The weights `beta(x) = (K_X + n lambda I)^(-1) k_X(X, x)` for each query row, `[m, n]`. */
  weights(x: TensorLike): Tensor
  /** `E[f(Y) | X = x]` for a function given by its values at the training outputs, `[n]` (or `[n, k]`); `[m]` (or `[m, k]`). */
  expectation(f: TensorLike, x: TensorLike): Tensor
  /** The embedding of the conditional distribution at each query row, evaluated at output points `[q, dy]`: `[m, q]`. */
  evaluate(x: TensorLike, z: TensorLike): Tensor
}

/**
 * The conditional mean embedding of `Y` given `X` (Song, Huang, Smola and Fukumizu) as a regularised operator estimated
 * from a joint sample: `mu_{Y|x} = sum_i beta_i(x) l(y_i, .)` with `beta(x) = (K_X + n lambda I)^(-1) k_X(X, x)`. The
 * conditional expectation of any function of `Y` is `sum_i beta_i(x) f(y_i)`; for `f` the identity it is kernel ridge
 * regression of `y` on `x` with ridge `n lambda`.
 *
 * @param kx - The kernel on inputs.
 * @param ky - The kernel on outputs.
 * @param x - Input sample `[n, dx]`.
 * @param y - Output sample `[n, dy]`.
 * @param lambda - Regularisation `lambda > 0` (default 1e-3).
 * @returns The operator.
 * @throws MdmlError `value` for a non-positive `lambda`; `shape` for mismatched samples.
 * @concept kernel-mean-embedding
 * @reference L. Song, J. Huang, A. Smola and K. Fukumizu, Hilbert space embeddings of conditional distributions with
 *   applications to dynamical systems (ICML 2009).
 * @example
 * ```ts
 * import { conditionalMeanEmbedding } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2]]
 * const y = [0, 1, 2]
 * const cme = conditionalMeanEmbedding(rbfKernel(), rbfKernel(), x, y.map((v) => [v]), 1e-6)
 * toFlat(cme.expectation(y, [[1]]))[0]?.toFixed(3) // => '1.000'
 * ```
 */
export function conditionalMeanEmbedding(
  kx: Kernel,
  ky: Kernel,
  x: TensorLike,
  y: TensorLike,
  lambda = 1e-3,
): ConditionalMeanEmbedding {
  if (!(lambda > 0)) throw new MdmlError('value', `conditionalMeanEmbedding: lambda must be positive, got ${lambda}`)
  const inputs = rowsOf(x, 'conditionalMeanEmbedding')
  const outputs = rowsOf(y, 'conditionalMeanEmbedding')
  const n = inputs.shape[0] ?? 0
  if (outputs.shape[0] !== n)
    throw new MdmlError('shape', `conditionalMeanEmbedding: ${n} inputs but ${outputs.shape[0]} outputs`)
  const k = gramOf(kx, inputs)
  const regularised = k.map((v, i) => v + (i % (n + 1) === 0 ? n * lambda : 0))
  const weights = (q: TensorLike): Tensor => {
    const rows = rowsOf(q, 'conditionalMeanEmbedding')
    const m = rows.shape[0] ?? 0
    const cross = gramOf(kx, inputs, rows)
    const beta = solveDense(regularised, n, cross, m)
    const out = new Float64Array(m * n)
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j * n + i] = beta[i * m + j] ?? 0
    return fromStorage(out, [m, n], 'float64')
  }
  return {
    weights,
    expectation: (f, q) => {
      const fv = primalOf(asTensor(f))
      const cols = fv.shape[1] ?? 1
      if ((fv.shape[0] ?? 0) !== n)
        throw new MdmlError('shape', `conditionalMeanEmbedding: f has ${fv.shape[0]} values for ${n} outputs`)
      const beta = toFlat(weights(q))
      const m = beta.length / n
      const values = toFlat(fv)
      const out = new Float64Array(m * cols)
      for (let j = 0; j < m; j++)
        for (let i = 0; i < n; i++)
          for (let c = 0; c < cols; c++)
            out[j * cols + c] = (out[j * cols + c] ?? 0) + (beta[j * n + i] ?? 0) * (values[i * cols + c] ?? 0)
      return fromStorage(out, fv.shape.length === 1 ? [m] : [m, cols], 'float64')
    },
    evaluate: (q, z) => {
      const beta = toFlat(weights(q))
      const m = beta.length / n
      const zs = rowsOf(z, 'conditionalMeanEmbedding')
      const l = gramOf(ky, outputs, zs)
      const nz = zs.shape[0] ?? 0
      const out = new Float64Array(m * nz)
      for (let j = 0; j < m; j++)
        for (let i = 0; i < n; i++)
          for (let c = 0; c < nz; c++)
            out[j * nz + c] = (out[j * nz + c] ?? 0) + (beta[j * n + i] ?? 0) * (l[i * nz + c] ?? 0)
      return fromStorage(out, [m, nz], 'float64')
    },
  }
}

/** Options of {@link kernelBayesRule}. */
export interface KernelBayesOptions {
  /** The prior as weights on the sample's input rows (default uniform: the posterior is the conditional embedding). */
  readonly prior?: ArrayLike<number>
  /** The regularisation of the conditional operator, `epsilon > 0` (default 1e-3). */
  readonly epsilon?: number
  /** The regularisation of the Bayes step, `delta > 0` (default 1e-2). */
  readonly delta?: number
}

/** The posterior of the kernel Bayes rule. */
export interface KernelBayesPosterior {
  /** The weights `w(y*)` on the sample's input rows for each observation, `[m, n]`: the posterior embedding is `sum_i w_i k(x_i, .)`. */
  weights(y: TensorLike): Tensor
  /** The posterior expectation of a function of `X`, given by its values at the sample's input rows `[n]`; `[m]`. */
  expectation(f: TensorLike, y: TensorLike): Tensor
}

/**
 * The kernel Bayes rule (Fukumizu, Song, Gretton), at its basic level: a posterior embedding of `X` given an observation
 * of `Y`, from a joint sample and a prior embedding on the same input rows, without any density. With prior weights `m`
 * on the inputs, the embedding of the induced marginal of `Y` has weights `mu = (G_X + n eps I)^(-1) G_X m`, and with
 * `Lambda = diag(mu)` the posterior weights at `y*` are
 * `w = Lambda G_Y ((Lambda G_Y)^2 + delta I)^(-1) Lambda k_Y(Y, y*)`. A uniform prior returns the conditional embedding.
 *
 * @param kx - The kernel on inputs.
 * @param ky - The kernel on observations.
 * @param x - Input sample `[n, dx]`.
 * @param y - Observation sample `[n, dy]`.
 * @param options - The prior and the two regularisations.
 * @returns The posterior, evaluable at observations.
 * @throws MdmlError `value` for non-positive regularisation; `shape` for mismatched samples or prior.
 * @concept kernel-mean-embedding
 * @reference K. Fukumizu, L. Song and A. Gretton, Kernel Bayes' rule: Bayesian inference with positive definite kernels
 *   (JMLR 14, 2013).
 * @example
 * ```ts
 * import { kernelBayesRule } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * const x = [[0], [1], [2], [3]]
 * const post = kernelBayesRule(rbfKernel(), rbfKernel(), x, x)
 * post.weights([[1]]).shape // => [1, 4]
 * ```
 */
export function kernelBayesRule(
  kx: Kernel,
  ky: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: KernelBayesOptions = {},
): KernelBayesPosterior {
  const inputs = rowsOf(x, 'kernelBayesRule')
  const obs = rowsOf(y, 'kernelBayesRule')
  const n = inputs.shape[0] ?? 0
  const eps = options.epsilon ?? 1e-3
  const delta = options.delta ?? 1e-2
  if (!(eps > 0) || !(delta > 0)) throw new MdmlError('value', 'kernelBayesRule: epsilon and delta must be positive')
  if (obs.shape[0] !== n) throw new MdmlError('shape', `kernelBayesRule: ${n} inputs but ${obs.shape[0]} observations`)
  if (options.prior !== undefined && options.prior.length !== n)
    throw new MdmlError('shape', 'kernelBayesRule: the prior needs one weight per input row')
  const prior = options.prior === undefined ? new Float64Array(n).fill(1 / n) : Float64Array.from(options.prior)
  const gx = gramOf(kx, inputs)
  const gy = gramOf(ky, obs)
  const gxm = new Float64Array(n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) gxm[i] = (gxm[i] ?? 0) + (gx[i * n + j] ?? 0) * (prior[j] ?? 0)
  const mu = solveDense(
    gx.map((v, i) => v + (i % (n + 1) === 0 ? n * eps : 0)),
    n,
    gxm,
    1,
  )
  // A = Lambda G_Y; the posterior weights are A (A^2 + delta I)^(-1) Lambda k_Y(Y, y*).
  const a = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i * n + j] = (mu[i] ?? 0) * (gy[i * n + j] ?? 0)
  const a2 = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let l = 0; l < n; l++) {
      const v = a[i * n + l] ?? 0
      if (v !== 0) for (let j = 0; j < n; j++) a2[i * n + j] = (a2[i * n + j] ?? 0) + v * (a[l * n + j] ?? 0)
    }
  const system = a2.map((v, i) => v + (i % (n + 1) === 0 ? delta : 0))
  const weights = (q: TensorLike): Tensor => {
    const rows = rowsOf(q, 'kernelBayesRule')
    const m = rows.shape[0] ?? 0
    const cross = gramOf(ky, obs, rows)
    const rhs = cross.map((v, idx) => v * (mu[Math.floor(idx / m)] ?? 0))
    const sol = solveDense(system, n, rhs, m)
    const out = new Float64Array(m * n)
    for (let j = 0; j < m; j++)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let l = 0; l < n; l++) s += (a[i * n + l] ?? 0) * (sol[l * m + j] ?? 0)
        out[j * n + i] = s
      }
    return fromStorage(out, [m, n], 'float64')
  }
  return {
    weights,
    expectation: (f, q) => {
      const values = toFlat(primalOf(asTensor(f)))
      const w = toFlat(weights(q))
      const m = w.length / n
      return fromStorage(
        Float64Array.from({ length: m }, (_, j) => {
          let s = 0
          for (let i = 0; i < n; i++) s += (w[j * n + i] ?? 0) * (values[i] ?? 0)
          return s
        }),
        [m],
        'float64',
      )
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernel Stein discrepancy
// ---------------------------------------------------------------------------------------------------------------------

/** A score function `s(x) = grad log p(x)` of a single point `[d]`, built from differentiable tensor operations. */
export type ScoreFunction = (x: Value) => Value

/**
 * The Gram matrix of the Stein kernel of a kernel `k` and a score function `s = grad log p`:
 * `u(x, y) = s(x).s(y) k(x, y) + s(x).grad_y k + s(y).grad_x k + tr(grad_x grad_y k)`, built by automatic differentiation
 * of the kernel (nested reverse and forward mode, vectorised over the pairs) so that any differentiable kernel and score
 * works. Under `p`, `E u(X, X') = 0`; the distribution is known only through its score, so the normalising constant is not needed.
 *
 * @param kernel - A twice-differentiable kernel.
 * @param score - The score function of one point `[d]`.
 * @param x - Points `[n, d]`.
 * @returns The `[n, n]` matrix of `u(x_i, x_j)`.
 * @concept kernel-stein-discrepancy
 * @reference Q. Liu, J. Lee and M. Jordan, A kernelized Stein discrepancy for goodness-of-fit tests (ICML 2016).
 * @example
 * ```ts
 * import { steinKernelGram } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { neg, toArray } from 'mdml-engine/tensor'
 * toArray(steinKernelGram(rbfKernel({ lengthScale: 1 }), (x) => neg(x), [[0]])) // => [[1]]
 * ```
 */
export function steinKernelGram(kernel: Kernel, score: ScoreFunction, x: TensorLike): Tensor {
  const points = rowsOf(x, 'steinKernelGram')
  const d = points.shape[1] ?? 1
  const k = (a: Value, b: Value): Value => reshape(kernel.k(reshape(a, [1, d]), reshape(b, [1, d])), [])
  const pair = (xi: Value, yj: Value): Value => {
    const gx = grad((a: Value) => k(a, yj))(xi)
    const gy = grad((b: Value) => k(xi, b))(yj)
    const mixed = jacfwd((b: Value) => grad((a: Value) => k(a, b))(xi))(yj)
    const sx = score(xi)
    const sy = score(yj)
    return add(add(mul(sum(mul(sx, sy)), k(xi, yj)), sum(mul(sx, gy))), add(sum(mul(sy, gx)), sum(diagonal(mixed))))
  }
  return primalOf(vmap(vmap(pair, [null, 0]), [0, null])(points, points)) as Tensor
}

/**
 * The squared kernel Stein discrepancy of a sample against the distribution with score `s`: the mean of the Stein
 * kernel's Gram matrix, as the V-statistic (default, non-negative) or the unbiased U-statistic (diagonal left out,
 * zero in expectation under the model).
 *
 * @param kernel - A twice-differentiable kernel.
 * @param score - The score function of one point.
 * @param x - The sample `[n, d]`.
 * @param unbiased - Use the U-statistic (default false).
 * @returns The squared discrepancy.
 * @concept kernel-stein-discrepancy
 * @example
 * ```ts
 * import { kernelSteinDiscrepancy } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { neg } from 'mdml-engine/tensor'
 * kernelSteinDiscrepancy(rbfKernel(), (x) => neg(x), [[0], [0]]) > 0 // => true
 * ```
 */
export function kernelSteinDiscrepancy(kernel: Kernel, score: ScoreFunction, x: TensorLike, unbiased = false): number {
  const u = toFlat(steinKernelGram(kernel, score, x))
  const n = Math.sqrt(u.length)
  let s = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (!unbiased || i !== j) s += u[i * n + j] ?? 0
  return s / (unbiased ? n * (n - 1) : n * n)
}

/** Options of {@link steinGoodnessOfFit}. */
export interface SteinTestOptions {
  /** Bootstrap draws (default 500). */
  readonly nBootstrap?: number
  readonly key?: Key
}

type BootstrapState = { readonly done: number; readonly draws: number[] }

function bootstrapAlgorithm(u: Float64Array, n: number, total: number): Algorithm<BootstrapState> {
  return {
    name: 'stein-wild-bootstrap',
    config: { nBootstrap: total },
    schema: channels({ metrics: { draws: { description: 'bootstrap statistics drawn' } } }),
    init: () => ({ done: 0, draws: [] }),
    step: (s, ctx) => {
      const draws = [...s.draws]
      const end = Math.min(total, s.done + 50)
      for (let r = s.done; r < end; r++) {
        // Multinomial weights w = counts / n, from n uniform draws; the statistic sums (w_i - 1/n)(w_j - 1/n) u_ij.
        const counts = new Float64Array(n)
        const draw = toFlat(uniform(foldIn(ctx.key, r), [n]))
        for (const v of draw)
          counts[Math.min(n - 1, Math.floor(v * n))] = (counts[Math.min(n - 1, Math.floor(v * n))] ?? 0) + 1
        const c = counts.map((v) => v / n - 1 / n)
        let stat = 0
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) if (i !== j) stat += (c[i] ?? 0) * (c[j] ?? 0) * (u[i * n + j] ?? 0)
        draws.push(stat)
      }
      return { done: end, draws }
    },
    done: (s) => (s.done >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { draws: s.done } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BootstrapState,
  }
}

/**
 * The kernel Stein goodness-of-fit test (Liu, Lee and Jordan; Chwialkowski, Strathmann and Gretton): does the sample come
 * from the distribution whose score is `s`? The statistic is `n` times the off-diagonal (U-statistic) mean of the Stein kernel, and the null
 * distribution comes from the multinomial wild bootstrap, `(1/n) sum_{i != j} (w_i - 1/n)(w_j - 1/n) u(x_i, x_j)` with
 * `w` multinomial counts over `n`. The score needs no normalising constant.
 *
 * @param kernel - A twice-differentiable kernel (for example a Gaussian with a median-heuristic bandwidth).
 * @param score - The score function of one point.
 * @param x - The sample `[n, d]`.
 * @param options - The bootstrap size and the key.
 * @returns The statistic (`n` times the unbiased squared discrepancy) and the bootstrap p-value.
 * @concept kernel-stein-discrepancy
 * @reference K. Chwialkowski, H. Strathmann and A. Gretton, A kernel test of goodness of fit (ICML 2016).
 * @example
 * ```ts
 * import { steinGoodnessOfFit } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * import { neg } from 'mdml-engine/tensor'
 * const x = [[5], [5.2], [4.9], [5.1], [5.3], [4.8], [5.0], [5.2]]
 * steinGoodnessOfFit(rbfKernel(), (p) => neg(p), x, { nBootstrap: 199 }).pValue < 0.05 // => true
 * ```
 */
export function steinGoodnessOfFit(
  kernel: Kernel,
  score: ScoreFunction,
  x: TensorLike,
  options: SteinTestOptions = {},
): TestResult {
  const u = Float64Array.from(toFlat(steinKernelGram(kernel, score, x)))
  const n = Math.sqrt(u.length)
  let stat = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) stat += u[i * n + j] ?? 0
  stat /= n
  const total = options.nBootstrap ?? 500
  const key = options.key ?? resolveKey({}, 'steinGoodnessOfFit')
  const run = runAlgorithm(bootstrapAlgorithm(u, n, total), { key, maxSteps: Number.MAX_SAFE_INTEGER })
  const scaled = run.state.draws.map((v) => v * n)
  const exceed = scaled.filter((v) => v >= stat).length
  return testResult({
    test: 'kernel Stein goodness-of-fit test',
    statistic: stat,
    pValue: (exceed + 1) / (total + 1),
    alternative: 'greater',
    method: `${total} wild bootstrap draws`,
  })
}
