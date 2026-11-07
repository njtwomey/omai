/**
 * Bijectors onto structured spaces and the building blocks of normalising flows: maps onto ordered vectors, the
 * probability simplex (stick-breaking), Cholesky factors of correlation and covariance matrices and positive-definite
 * matrices (as NumPyro and TFP), plus the RealNVP affine coupling layer, masked autoregressive flow (MAF) layers with
 * MADE masks, and permutations between layers.
 */

import { MdmlError } from '../../base/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { logSigmoid } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  arange,
  broadcastTo,
  concat,
  cumsum,
  diagonal,
  div,
  exp,
  fromStorage,
  log,
  log1p,
  matmul,
  mul,
  neg,
  pad,
  reshape,
  sign,
  sliceAxis,
  softplus,
  sqrt,
  square,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  take,
  tanh,
  type Value,
} from '../../tensor/index.ts'
import { type Bijector, defineBijector } from './bijector.ts'

const lastSize = (x: Value): number => x.shape[x.shape.length - 1] ?? 0
const lastSlice = (x: Value, start: number, stop: number): Value => sliceAxis(x, -1, start, stop)
// log(1 - tanh(x)^2), stable for large |x|.
const logTanhDerivative = (x: Value): Value => mul(sub(sub(Math.LN2, x), softplus(mul(x, -2))), 2)

/**
 * The ordered bijector from R^d onto increasing vectors: y_0 = x_0 and y_i = y_{i-1} + exp(x_i) (NumPyro's
 * OrderedTransform), used for cutpoints and ordered location parameters.
 *
 * @returns The bijector (vector-valued: log-determinant sum_{i >= 1} x_i).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.ordered().forward([1, 0, 0])) // => [1, 2, 3]
 * ```
 */
export function ordered(): Bijector {
  return defineBijector({
    name: 'ordered',
    inputRank: 1,
    codomain: { kind: 'ordered' },
    forward: (x) => {
      const d = lastSize(x)
      return cumsum(concat([lastSlice(x, 0, 1), exp(lastSlice(x, 1, d))], -1), -1)
    },
    inverse: (y) => {
      const d = lastSize(y)
      return concat([lastSlice(y, 0, 1), log(sub(lastSlice(y, 1, d), lastSlice(y, 0, d - 1)))], -1)
    },
    forwardLogDetJacobian: (x) => sum(lastSlice(x, 1, lastSize(x)), -1),
  })
}

/**
 * The stick-breaking bijector from R^(K-1) onto the K-simplex (NumPyro's StickBreakingTransform): z_i =
 * sigmoid(x_i - log(K - 1 - i)) breaks off a fraction of the remaining stick, so x = 0 maps to the uniform vector.
 *
 * @returns The bijector (vector-valued; changes the last axis from K - 1 to K).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.stickBreaking().forward([0, 0])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.333333333333, 0.333333333333, 0.333333333333]
 * ```
 */
export function stickBreaking(): Bijector {
  const offsets = (n: number): Tensor =>
    fromStorage(
      Float64Array.from({ length: n }, (_, i) => Math.log(n - i)),
      [n],
      'float64',
    )
  // log z_i, log(1 - z_i) and the log remaining stick before each break.
  const parts = (x: Value): { logZ: Value; logRest: Value; logStick: Value } => {
    const t = sub(x, offsets(lastSize(x)))
    const logZ = logSigmoid(t)
    const logRest = logSigmoid(neg(t))
    const inclusive = cumsum(logRest, -1)
    return { logZ, logRest, logStick: sub(inclusive, logRest) }
  }
  return defineBijector({
    name: 'stickBreaking',
    inputRank: 1,
    codomain: { kind: 'simplex' },
    forward: (x) => {
      const { logZ, logRest, logStick } = parts(x)
      // y_i = z_i (remaining stick), and the last piece is what remains after every break.
      return exp(concat([add(logZ, logStick), sum(logRest, -1, true)], -1))
    },
    inverse: (y) => {
      const k = lastSize(y)
      // z_i / (1 - z_i) = y_i / (the pieces after i), read from a reversed cumulative sum: no cancellation in the tail.
      const after = lastSlice(cumsum(y, -1, true), 1, k)
      return add(sub(log(lastSlice(y, 0, k - 1)), log(after)), offsets(k - 1))
    },
    forwardLogDetJacobian: (x) => {
      const { logZ, logRest, logStick } = parts(x)
      return sum(add(add(logZ, logRest), logStick), -1)
    },
    forwardShape: (shape) => [...shape.slice(0, -1), (shape[shape.length - 1] ?? 0) + 1],
    inverseShape: (shape) => [...shape.slice(0, -1), (shape[shape.length - 1] ?? 0) - 1],
  })
}

/** The size d of the matrix whose strictly lower triangle has n entries. */
function strictSize(n: number): number {
  const d = (1 + Math.sqrt(1 + 8 * n)) / 2
  if (!Number.isInteger(d)) throw new MdmlError('shape', `a strictly lower triangle cannot have ${n} entries`)
  return d
}

/** The size d of the matrix whose lower triangle (with diagonal) has n entries. */
function triangleSize(n: number): number {
  const d = (Math.sqrt(1 + 8 * n) - 1) / 2
  if (!Number.isInteger(d)) throw new MdmlError('shape', `a lower triangle cannot have ${n} entries`)
  return d
}

/** Pad a row `[..., m]` with zeros to `[..., d]`. */
function padRow(row: Value, d: number): Value {
  const widths: [number, number][] = row.shape.map(() => [0, 0])
  widths[widths.length - 1] = [0, d - lastSize(row)]
  return pad(row, widths)
}

/**
 * The bijector from R^(d(d-1)/2) onto Cholesky factors of d x d correlation matrices (NumPyro's
 * CorrCholeskyTransform, the LKJ parameterisation): tanh maps to partial correlations z filled row by row into the
 * strict lower triangle, and L_ij = z_ij prod_{k<j} sqrt(1 - z_ik^2), L_ii = prod_{k<i} sqrt(1 - z_ik^2), so every row
 * has unit norm.
 *
 * @returns The bijector (vector to matrix: input rank 1, output rank 2).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.correlationCholesky().forward([0])) // => [[1, 0], [0, 1]]
 * ```
 */
export function correlationCholesky(): Bijector {
  // For row i: the partial correlations tanh(x), and the exclusive cumulative sums of log sqrt(1 - z^2) (log of the
  // products). log sqrt(1 - tanh(x)^2) = log 2 - |x| - log1p(exp(-2 |x|)) stays finite where tanh(x) rounds to 1.
  const rowParts = (x: Value, i: number): { zr: Value; logPrefix: Value; logDiag: Value } => {
    const off = (i * (i - 1)) / 2
    const xr = lastSlice(x, off, off + i)
    const zr = tanh(xr)
    const logs = sub(Math.LN2, add(abs(xr), log1p(exp(mul(abs(xr), -2)))))
    const inclusive = cumsum(logs, -1)
    return { zr, logPrefix: sub(inclusive, logs), logDiag: sum(logs, -1, true) }
  }
  return defineBijector({
    name: 'correlationCholesky',
    inputRank: 1,
    outputRank: 2,
    codomain: { kind: 'correlationCholesky' },
    forward: (x) => {
      const d = strictSize(lastSize(x))
      const first = fromStorage(
        Float64Array.from({ length: d }, (_, j) => (j === 0 ? 1 : 0)),
        [d],
        'float64',
      )
      const rows: Value[] = [broadcastTo(first, [...x.shape.slice(0, -1), d])]
      for (let i = 1; i < d; i++) {
        const { zr, logPrefix, logDiag } = rowParts(x, i)
        rows.push(padRow(concat([mul(zr, exp(logPrefix)), exp(logDiag)], -1), d))
      }
      return stack(rows, -2)
    },
    inverse: (l) => {
      const d = lastSize(l)
      const parts: Value[] = []
      for (let i = 1; i < d; i++) {
        const row = reshape(lastSlice(take(l, [i], -2), 0, i + 1), [...l.shape.slice(0, -2), i + 1])
        // The row has unit norm, so the radius left at j is S_j = sum_{k >= j} L_ik^2 (a reversed cumulative sum, with no
        // cancellation) and z_j = L_ij / sqrt(S_j); atanh(z) = log1p(|z|) + (log S_j - log S_{j+1}) / 2 for z > 0, odd.
        const s = cumsum(square(row), -1, true)
        const here = lastSlice(s, 0, i)
        const z = div(lastSlice(row, 0, i), sqrt(here))
        const half = mul(sub(log(here), log(lastSlice(s, 1, i + 1))), 0.5)
        parts.push(mul(sign(z), add(log1p(abs(z)), half)))
      }
      return concat(parts, -1)
    },
    forwardLogDetJacobian: (x) => {
      const d = strictSize(lastSize(x))
      let total: Value = sum(logTanhDerivative(x), -1)
      for (let i = 1; i < d; i++) total = add(total, sum(rowParts(x, i).logPrefix, -1))
      return total
    },
    forwardShape: (shape) => {
      const d = strictSize(shape[shape.length - 1] ?? 0)
      return [...shape.slice(0, -1), d, d]
    },
    inverseShape: (shape) => {
      const d = shape[shape.length - 1] ?? 0
      return [...shape.slice(0, -2), (d * (d - 1)) / 2]
    },
  })
}

/**
 * The bijector from R^(d(d+1)/2) onto lower-triangular matrices with a positive diagonal (Cholesky factors): entries
 * fill the lower triangle row by row, with the diagonal through exp.
 *
 * @returns The bijector (vector to matrix; log-determinant the sum of the raw diagonal entries).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.lowerCholesky().forward([0, 2, 0])) // => [[1, 0], [2, 1]]
 * ```
 */
export function lowerCholesky(): Bijector {
  const diagIndex = (i: number): number => (i * (i + 1)) / 2 + i
  return defineBijector({
    name: 'lowerCholesky',
    inputRank: 1,
    outputRank: 2,
    codomain: { kind: 'lowerCholesky' },
    forward: (x) => {
      const d = triangleSize(lastSize(x))
      const rows: Value[] = []
      for (let i = 0; i < d; i++) {
        const off = (i * (i + 1)) / 2
        rows.push(padRow(concat([lastSlice(x, off, off + i), exp(lastSlice(x, off + i, off + i + 1))], -1), d))
      }
      return stack(rows, -2)
    },
    inverse: (l) => {
      const d = lastSize(l)
      const parts: Value[] = []
      for (let i = 0; i < d; i++) {
        const row = reshape(lastSlice(take(l, [i], -2), 0, i + 1), [...l.shape.slice(0, -2), i + 1])
        parts.push(concat([lastSlice(row, 0, i), log(lastSlice(row, i, i + 1))], -1))
      }
      return concat(parts, -1)
    },
    forwardLogDetJacobian: (x) => {
      const d = triangleSize(lastSize(x))
      return sum(
        take(
          x,
          Array.from({ length: d }, (_, i) => diagIndex(i)),
          -1,
        ),
        -1,
      )
    },
    forwardShape: (shape) => {
      const d = triangleSize(shape[shape.length - 1] ?? 0)
      return [...shape.slice(0, -1), d, d]
    },
    inverseShape: (shape) => {
      const d = shape[shape.length - 1] ?? 0
      return [...shape.slice(0, -2), (d * (d + 1)) / 2]
    },
  })
}

/**
 * The map L -> L L^T from Cholesky factors to positive-definite matrices (TensorFlow Probability's
 * CholeskyOuterProduct), or, with `correlation`, from Cholesky factors of correlation matrices (unit-norm rows) to
 * correlation matrices. Log-determinants are over the free entries: d log 2 + sum_j (d - j) log L_jj, or
 * sum_j (d - 1 - j) log L_jj for correlations (0-based j).
 *
 * @param kind - `covariance` (default) or `correlation`.
 * @returns The bijector (matrix to matrix; the inverse is the Cholesky factorisation).
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.choleskyOuterProduct().forward([[2, 0], [1, 1]])) // => [[4, 2], [2, 2]]
 * ```
 */
export function choleskyOuterProduct(kind: 'covariance' | 'correlation' = 'covariance'): Bijector {
  const correlation = kind === 'correlation'
  return defineBijector({
    name: correlation ? 'correlationOuterProduct' : 'choleskyOuterProduct',
    inputRank: 2,
    domain: { kind: correlation ? 'correlationCholesky' : 'lowerCholesky' },
    codomain: { kind: correlation ? 'correlation' : 'positiveDefinite' },
    forward: (l) => matmul(l, swapaxes(l, -1, -2)),
    inverse: (s) => cholesky(s),
    forwardLogDetJacobian: (l) => {
      const d = lastSize(l)
      const weights = sub(correlation ? d - 1 : d, arange(d))
      const logDiag = mul(log(diagonal(l, 0, -2, -1)), weights)
      return correlation ? sum(logDiag, -1) : add(sum(logDiag, -1), d * Math.LN2)
    },
  })
}

/** The shift and log-scale a coupling or autoregressive conditioner produces. */
export interface ShiftLogScale {
  readonly shift: Value
  readonly logScale: Value
}

/**
 * A RealNVP affine coupling layer (Dinh et al., 2017): the first `split` coordinates pass through and condition an
 * elementwise affine map of the rest, y_b = x_b exp(s(x_a)) + t(x_a). Inverse and log-determinant (sum of s) cost one
 * conditioner call each.
 *
 * @param split - How many leading coordinates pass through.
 * @param conditioner - Maps x_a `[..., split]` to `{ shift, logScale }` for the remaining coordinates.
 * @returns The bijector (vector-valued).
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { mul, toArray } from 'mdml-engine/tensor'
 * const b = bijectors.affineCoupling(1, (xa) => ({ shift: xa, logScale: mul(xa, 0) }))
 * toArray(b.forward([2, 3])) // => [2, 5]
 * ```
 */
export function affineCoupling(split: number, conditioner: (xa: Value) => ShiftLogScale): Bijector {
  return defineBijector({
    name: 'affineCoupling',
    inputRank: 1,
    forward: (x) => {
      const xa = lastSlice(x, 0, split)
      const { shift, logScale } = conditioner(xa)
      return concat([xa, add(mul(lastSlice(x, split, lastSize(x)), exp(logScale)), shift)], -1)
    },
    inverse: (y) => {
      const ya = lastSlice(y, 0, split)
      const { shift, logScale } = conditioner(ya)
      return concat([ya, mul(sub(lastSlice(y, split, lastSize(y)), shift), exp(neg(logScale)))], -1)
    },
    forwardLogDetJacobian: (x) => sum(conditioner(lastSlice(x, 0, split)).logScale, -1),
    inverseLogDetJacobian: (y) => neg(sum(conditioner(lastSlice(y, 0, split)).logScale, -1)),
  })
}

/**
 * A permutation of the last axis, y = x[..., permutation] (volume preserving), used between coupling layers.
 *
 * @param permutation - The permutation of 0 .. d - 1.
 * @returns The bijector.
 * @throws MdmlError with code `value` when `permutation` is not a permutation.
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.permute([2, 0, 1]).forward([10, 20, 30])) // => [30, 10, 20]
 * ```
 */
export function permute(permutation: readonly number[]): Bijector {
  const d = permutation.length
  const inverse = new Array<number>(d).fill(-1)
  for (const [i, p] of permutation.entries()) {
    if (!(Number.isInteger(p) && p >= 0 && p < d) || inverse[p] !== -1)
      throw new MdmlError('value', 'permute: not a permutation')
    inverse[p] = i
  }
  return defineBijector({
    name: 'permute',
    inputRank: 1,
    forward: (x) => take(x, [...permutation], -1),
    inverse: (y) => take(y, inverse, -1),
    forwardLogDetJacobian: (x) => sum(mul(x, 0), -1),
  })
}

/**
 * A masked autoregressive flow layer (MAF, Papamakarios et al., 2017): y_i = x_i exp(s_i(y_{<i})) + m_i(y_{<i}),
 * where `conditioner` maps a vector to `{ shift, logScale }` with output i depending only on inputs before i (a MADE
 * network, see {@link madeMasks}). The inverse (density evaluation) is one conditioner call; the forward map (sampling)
 * takes d sequential passes.
 *
 * @param conditioner - The autoregressive conditioner on `[..., d]`.
 * @returns The bijector (vector-valued; log-determinant sum of s at y).
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { concat, mul, sliceAxis, toArray, zerosLike } from 'mdml-engine/tensor'
 * // m_0 = 0, m_1 = y_0: an autoregressive shift.
 * const shiftByPrevious = (y) => ({ shift: concat([zerosLike(sliceAxis(y, -1, 0, 1)), sliceAxis(y, -1, 0, 1)], -1), logScale: mul(y, 0) })
 * toArray(bijectors.autoregressive(shiftByPrevious).forward([1, 1])) // => [1, 2]
 * ```
 */
export function autoregressive(conditioner: (y: Value) => ShiftLogScale): Bijector {
  const forward = (x: Value): Value => {
    let y: Value = mul(x, 0)
    for (let i = 0; i < lastSize(x); i++) {
      const { shift, logScale } = conditioner(y)
      y = add(mul(x, exp(logScale)), shift)
    }
    return y
  }
  return defineBijector({
    name: 'autoregressive',
    inputRank: 1,
    forward,
    inverse: (y) => {
      const { shift, logScale } = conditioner(y)
      return mul(sub(y, shift), exp(neg(logScale)))
    },
    forwardLogDetJacobian: (x) => sum(conditioner(forward(x)).logScale, -1),
    inverseLogDetJacobian: (y) => neg(sum(conditioner(y).logScale, -1)),
  })
}

/**
 * The connectivity masks of a MADE network (Germain et al., 2015) for an autoregressive conditioner: input degrees
 * 1 .. d, hidden degrees assigned cyclically in 1 .. d - 1, a hidden unit seeing inputs of degree at most its own and
 * an output of degree k seeing hidden units of degree below k. Multiply each layer's weight matrix (`[in, out]`, for
 * y = x W) by its mask.
 *
 * @param inputDim - The dimension d.
 * @param hiddenSizes - Widths of the hidden layers.
 * @returns One mask per layer: `[d, h1]`, `[h1, h2]`, ..., `[h_last, d]`.
 * @example
 * ```ts
 * import { bijectors } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bijectors.madeMasks(2, [1])[1]) // => [[0, 1]]
 * ```
 */
export function madeMasks(inputDim: number, hiddenSizes: readonly number[]): Tensor[] {
  if (inputDim < 2) throw new MdmlError('value', 'madeMasks: needs at least two inputs')
  const degrees: number[][] = [Array.from({ length: inputDim }, (_, i) => i + 1)]
  for (const h of hiddenSizes) degrees.push(Array.from({ length: h }, (_, k) => (k % (inputDim - 1)) + 1))
  const masks: Tensor[] = []
  const layers = [...degrees.slice(1), Array.from({ length: inputDim }, (_, i) => i + 1)]
  for (const [l, outDeg] of layers.entries()) {
    const inDeg = degrees[l] as number[]
    const output = l === layers.length - 1
    const data = new Float64Array(inDeg.length * outDeg.length)
    for (const [i, a] of inDeg.entries())
      for (const [j, b] of outDeg.entries()) data[i * outDeg.length + j] = (output ? b > a : b >= a) ? 1 : 0
    masks.push(fromStorage(data, [inDeg.length, outDeg.length], 'float64'))
  }
  return masks
}
