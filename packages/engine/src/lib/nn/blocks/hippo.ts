/**
 * HiPPO, the initialisations of state-space models: the matrices that make a linear system remember its input's
 * history (a projection of the past onto Legendre polynomials over a sliding window `legt`, over all the past
 * `legs`, or onto Laguerre functions with exponential forgetting `lagt`), their normal-plus-low-rank (NPLR) forms
 * `A = V diag(lambda) V^H - P P^T` that S4 diagonalises, and the diagonal approximations of S4D (`lin`, `inv`, `real`).
 * Every matrix is the state matrix of `x' = A x + B u`, so it is negative-definite in the sense that matters: the
 * state decays.
 */

import { OmaiError } from '../../base/index.ts'
import { eig } from '../../numerics/linalg/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'

/** The HiPPO measures: scaled Legendre (`legs`), translated Legendre (`legt`) and translated Laguerre (`lagt`). */
export type HippoMeasure = 'legs' | 'legt' | 'lagt'

/** The state matrix `[N, N]` and input vector `[N]` of `x' = A x + B u`. */
export interface HippoSystem {
  readonly a: Tensor
  readonly b: Tensor
}

/** Options of the HiPPO constructors. */
export interface HippoOptions {
  /** The window length of `legt` (default 1): the matrices scale as 1 / theta. */
  readonly theta?: number
}

interface Dense {
  readonly a: Float64Array
  readonly b: Float64Array
  /** The low-rank factor `[N, r]`, row-major, with `a + p p^T` normal. */
  readonly p: Float64Array
  readonly rank: number
}

const c = (n: number): number => Math.sqrt(2 * n + 1)

function dense(measure: HippoMeasure, n: number, theta: number): Dense {
  if (!Number.isInteger(n) || n < 1)
    throw new OmaiError('value', `hippo: the order must be a positive integer, got ${n}`)
  const a = new Float64Array(n * n)
  const b = new Float64Array(n)
  if (measure === 'legs') {
    // A_nk = -sqrt(2n+1) sqrt(2k+1) below the diagonal, -(n+1) on it; B_n = sqrt(2n+1). P_n = sqrt(n + 1/2).
    for (let i = 0; i < n; i++) {
      b[i] = c(i)
      for (let k = 0; k < i; k++) a[i * n + k] = -c(i) * c(k)
      a[i * n + i] = -(i + 1)
    }
    return { a, b, p: Float64Array.from({ length: n }, (_, i) => Math.sqrt(i + 0.5)), rank: 1 }
  }
  if (measure === 'legt') {
    // A_nk = -sqrt(2n+1) sqrt(2k+1) (1 for k <= n, (-1)^(n-k) for k > n) / theta, B_n = sqrt(2n+1) / theta. The
    // correction has rank two: the even and the odd entries of sqrt(2n+1) (over sqrt(theta)).
    const s = 1 / theta
    for (let i = 0; i < n; i++) {
      b[i] = c(i) * s
      for (let k = 0; k < n; k++) a[i * n + k] = -c(i) * c(k) * (k <= i ? 1 : (-1) ** (i - k)) * s
    }
    const p = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) p[2 * i + (i % 2)] = c(i) / Math.sqrt(theta)
    return { a, b, p, rank: 2 }
  }
  // lagt: A = I/2 - tril(ones), B = ones, so the impulse response of state n is L_n(s) exp(-s/2).
  for (let i = 0; i < n; i++) {
    b[i] = 1
    for (let k = 0; k < i; k++) a[i * n + k] = -1
    a[i * n + i] = -0.5
  }
  return { a, b, p: new Float64Array(n).fill(Math.SQRT1_2), rank: 1 }
}

const system = (d: Dense, n: number): HippoSystem => ({
  a: fromStorage(d.a, [n, n], 'float64'),
  b: fromStorage(d.b, [n], 'float64'),
})

/**
 * The HiPPO-LegS system (Gu et al. 2020): the state is the coefficient vector of the whole history projected onto the
 * orthonormal Legendre polynomials on `[0, t]`. `A_nk = -sqrt(2n+1) sqrt(2k+1)` below the diagonal, `-(n+1)` on it,
 * `B_n = sqrt(2n+1)`. The original dynamics are time-varying, `c' = (A c + B f) / t`; S4 uses the time-invariant
 * system with the same matrices.
 *
 * @param n - The order N (number of Legendre polynomials).
 * @returns A `[N, N]` and B `[N]`.
 * @concept state-space-models
 * @reference Gu, Dao, Ermon, Rudra and Re (2020), HiPPO: recurrent memory with optimal polynomial projections.
 * @example
 * ```ts
 * import { hippoLegS } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hippoLegS(2).a) // => [[-1, 0], [-1.7320508075688772, -2]]
 * ```
 */
export function hippoLegS(n: number): HippoSystem {
  return system(dense('legs', n, 1), n)
}

/**
 * The HiPPO-LegT system: the state is the projection of the last `theta` time units onto the orthonormal Legendre
 * polynomials of the window. `A_nk = -sqrt(2n+1) sqrt(2k+1)` for `k <= n` and `(-1)^(n-k)` times that above the
 * diagonal, `B_n = sqrt(2n+1)`, both divided by `theta`.
 *
 * @param n - The order N.
 * @param options - The window length `theta` (default 1).
 * @returns A `[N, N]` and B `[N]`.
 * @concept state-space-models
 * @reference Gu et al. (2020), HiPPO; Voelker, Kajic and Eliasmith (2019), Legendre memory units.
 * @example
 * ```ts
 * import { hippoLegT } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hippoLegT(2, { theta: 2 }).b) // => [0.5, 0.8660254037844386]
 * ```
 */
export function hippoLegT(n: number, options: HippoOptions = {}): HippoSystem {
  return system(dense('legt', n, options.theta ?? 1), n)
}

/**
 * The HiPPO-LagT system: the state is the projection of the past onto the Laguerre functions with exponential
 * forgetting. `A = I/2 - tril(1)`, `B = 1`; the response of state `n` to an impulse `s` time units ago is
 * `L_n(s) exp(-s/2)`.
 *
 * @param n - The order N.
 * @returns A `[N, N]` and B `[N]`.
 * @concept state-space-models
 * @reference Gu et al. (2020), HiPPO.
 * @example
 * ```ts
 * import { hippoLagT } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hippoLagT(2).a) // => [[-0.5, 0], [-1, -0.5]]
 * ```
 */
export function hippoLagT(n: number): HippoSystem {
  return system(dense('lagt', n, 1), n)
}

/** A complex vector or matrix as real and imaginary tensors. */
export interface ComplexPair {
  readonly re: Tensor
  readonly im: Tensor
}

/**
 * The normal-plus-low-rank form of a HiPPO matrix: `A = V diag(lambda) V^H - P P^T`, where `V` is unitary and
 * `V diag(lambda) V^H` is normal. In the eigenbasis (`A~ = V^H A V`) the matrix is `diag(lambda) - P~ P~^H`, diagonal
 * plus low rank, and the input map is `B~ = V^H B`.
 */
export interface NplrHippo {
  readonly measure: HippoMeasure
  /** The dense state matrix `[N, N]` and input `[N]` (as {@link hippoLegS} and the others). */
  readonly a: Tensor
  readonly b: Tensor
  /** The low-rank factor `[N, r]`: rank 1 for `legs` and `lagt`, 2 for `legt`. */
  readonly p: Tensor
  /** The eigenvalues `[N]` of the normal part, real part -1/2 for `legs` and 0 for the others, ordered by imaginary part. */
  readonly lambda: ComplexPair
  /** The unitary eigenvectors `[N, N]` (columns). */
  readonly v: ComplexPair
  /** `V^H P` `[N, r]` and `V^H B` `[N]`: the low-rank factor and input map in the eigenbasis. */
  readonly pTilde: ComplexPair
  readonly bTilde: ComplexPair
}

/**
 * The NPLR decomposition of a HiPPO matrix (S4's `make_NPLR_HiPPO` and `make_DPLR_HiPPO`): subtracting the low-rank
 * term makes the matrix normal, `A + P P^T = -(1/2) I + skew` for LegS (and plain skew for LegT and LagT), so its
 * eigenvectors are orthogonal and its eigenvalues are `-1/2 + i omega` (or `i omega`). Computed from the eigen
 * decomposition of the skew-symmetric part.
 *
 * @param measure - `legs`, `legt` or `lagt`.
 * @param n - The order N.
 * @param options - The window length of `legt`.
 * @returns The dense matrices, the low-rank factor, the eigen decomposition and the eigenbasis forms.
 * @concept state-space-models
 * @reference Gu, Goel and Re (2022), Efficiently modeling long sequences with structured state spaces.
 * @example
 * ```ts
 * import { nplrHippo } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(nplrHippo('legs', 4).lambda.re) // => [-0.5, -0.5, -0.5, -0.5]
 * ```
 */
export function nplrHippo(measure: HippoMeasure, n: number, options: HippoOptions = {}): NplrHippo {
  const d = dense(measure, n, options.theta ?? 1)
  const r = d.rank
  // K = A + P P^T: normal, a scalar multiple of the identity plus a skew-symmetric matrix.
  const k = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = d.a[i * n + j] ?? 0
      for (let q = 0; q < r; q++) s += (d.p[i * r + q] ?? 0) * (d.p[j * r + q] ?? 0)
      k[i * n + j] = s
    }
  let shift = 0
  for (let i = 0; i < n; i++) shift += (k[i * n + i] ?? 0) / n
  for (let i = 0; i < n; i++) k[i * n + i] = (k[i * n + i] ?? 0) - shift
  const { values, vectors } = eig(fromStorage(k, [n, n], 'float64'))
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (x, y) => (values.im.data[x] ?? 0) - (values.im.data[y] ?? 0),
  )
  const lr = new Float64Array(n).fill(shift)
  const li = Float64Array.from(order, (i) => values.im.data[i] ?? 0)
  const vr = new Float64Array(n * n)
  const vi = new Float64Array(n * n)
  order.forEach((src, col) => {
    let nrm = 0
    for (let row = 0; row < n; row++)
      nrm += (vectors.re.data[row * n + src] ?? 0) ** 2 + (vectors.im.data[row * n + src] ?? 0) ** 2
    const s = 1 / Math.sqrt(nrm)
    for (let row = 0; row < n; row++) {
      vr[row * n + col] = (vectors.re.data[row * n + src] ?? 0) * s
      vi[row * n + col] = (vectors.im.data[row * n + src] ?? 0) * s
    }
  })
  // V^H x for a real matrix x [N, m].
  const hermitian = (x: Float64Array, m: number, shape: number[]): ComplexPair => {
    const re = new Float64Array(n * m)
    const im = new Float64Array(n * m)
    for (let col = 0; col < n; col++)
      for (let q = 0; q < m; q++) {
        let sr = 0
        let si = 0
        for (let row = 0; row < n; row++) {
          sr += (vr[row * n + col] ?? 0) * (x[row * m + q] ?? 0)
          si -= (vi[row * n + col] ?? 0) * (x[row * m + q] ?? 0)
        }
        re[col * m + q] = sr
        im[col * m + q] = si
      }
    return { re: fromStorage(re, shape, 'float64'), im: fromStorage(im, shape, 'float64') }
  }
  return {
    measure,
    a: fromStorage(d.a, [n, n], 'float64'),
    b: fromStorage(d.b, [n], 'float64'),
    p: fromStorage(d.p, [n, r], 'float64'),
    lambda: { re: fromStorage(lr, [n], 'float64'), im: fromStorage(li, [n], 'float64') },
    v: { re: fromStorage(vr, [n, n], 'float64'), im: fromStorage(vi, [n, n], 'float64') },
    pTilde: hermitian(d.p, r, [n, r]),
    bTilde: hermitian(d.b, 1, [n]),
  }
}

/**
 * The dense matrix an NPLR decomposition stands for, `Re(V diag(lambda) V^H) - P P^T`: equal to the HiPPO matrix it
 * came from (the imaginary part of `V diag(lambda) V^H` vanishes).
 *
 * @param nplr - The decomposition.
 * @returns The `[N, N]` matrix.
 * @concept state-space-models
 * @example
 * ```ts
 * import { nplrHippo, nplrReconstruct } from 'omai-engine/nn/blocks'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(nplrReconstruct(nplrHippo('lagt', 2))).map((x) => Math.round(x * 1e9) / 1e9) // => [-0.5, 0, -1, -0.5]
 * ```
 */
export function nplrReconstruct(nplr: NplrHippo): Tensor {
  const n = nplr.lambda.re.shape[0] ?? 0
  const r = nplr.p.shape[1] ?? 1
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0
      // Re(V Lambda V^H)_ij = sum_k Re(V_ik lambda_k conj(V_jk)).
      for (let k = 0; k < n; k++) {
        const ar = nplr.v.re.data[i * n + k] ?? 0
        const ai = nplr.v.im.data[i * n + k] ?? 0
        const br = nplr.v.re.data[j * n + k] ?? 0
        const bi = nplr.v.im.data[j * n + k] ?? 0
        const lr = nplr.lambda.re.data[k] ?? 0
        const li = nplr.lambda.im.data[k] ?? 0
        const wr = ar * lr - ai * li
        const wi = ar * li + ai * lr
        s += wr * br + wi * bi
      }
      for (let q = 0; q < r; q++) s -= (nplr.p.data[i * r + q] ?? 0) * (nplr.p.data[j * r + q] ?? 0)
      out[i * n + j] = s
    }
  return fromStorage(out, [n, n], 'float64')
}

/** The diagonal state matrices of S4D. */
export type DiagonalHippo = 'lin' | 'inv' | 'real'

/**
 * The diagonal initialisations of S4D (Gu et al. 2022): complex eigenvalues for `count` states of conjugate pairs
 * (the layer takes twice the real part of the output, so the state size is `2 count`). `lin` is
 * `-1/2 + i pi k` (Fourier basis), `inv` is `-1/2 + i (N/pi)(N/(2k+1) - 1)` with `N = 2 count` (the inverse law that
 * approximates the LegS spectrum), `real` is `-(k+1)` (S4D-Real, no oscillation).
 *
 * @param kind - `lin`, `inv` or `real`.
 * @param count - The number of complex eigenvalues (N/2).
 * @returns Real and imaginary parts `[count]`.
 * @concept state-space-models
 * @reference Gu, Gupta, Goel and Re (2022), On the parameterization and initialization of diagonal state space models.
 * @example
 * ```ts
 * import { s4dEigenvalues } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(s4dEigenvalues('real', 3).re) // => [-1, -2, -3]
 * ```
 */
export function s4dEigenvalues(kind: DiagonalHippo, count: number): ComplexPair {
  if (!Number.isInteger(count) || count < 1) throw new OmaiError('value', `s4dEigenvalues: bad count ${count}`)
  const big = 2 * count
  const re = Float64Array.from({ length: count }, (_, k) => (kind === 'real' ? -(k + 1) : -0.5))
  const im = Float64Array.from({ length: count }, (_, k) =>
    kind === 'lin' ? Math.PI * k : kind === 'inv' ? (big / Math.PI) * (big / (2 * k + 1) - 1) : 0,
  )
  return { re: fromStorage(re, [count], 'float64'), im: fromStorage(im, [count], 'float64') }
}
