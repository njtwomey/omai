/**
 * The singular value decomposition (numpy.linalg `svd`, reduced) by one-sided Jacobi rotations, which keeps small
 * singular values accurate, or by Golub-Kahan-Reinsch, as a differentiable primitive, with what is built on it:
 * singular values, the pseudo-inverse, best low-rank approximations (Eckart-Young), null spaces and the polar
 * decomposition.
 */

import {
  add,
  arg,
  batchLeading,
  bind,
  definePrimitive,
  div,
  expandDims,
  type FloatDType,
  greater,
  matmul,
  max,
  mul,
  type Primitive,
  primalOf,
  promoteFloat,
  sliceAxis,
  square,
  sub,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { svdKernel } from './_kernels.ts'

/** The SVD algorithms: one-sided Jacobi (accurate small singular values) or Golub-Kahan-Reinsch (the classic). */
export type SvdMethod = 'jacobi' | 'golub-kahan'

import {
  assemble,
  diagEmbed,
  diagPart,
  identity as identityOf,
  matrices,
  matrixShape,
  mT,
  namingNonFinite,
} from './_matrix.ts'
import { eigengapInverse } from './eigh.ts'

// Scale columns (right-multiply by diag(s)) and rows (left-multiply by diag(s)).
const scaleColumns = (x: Value, s: Value): Value => mul(x, expandDims(s, -2))
const scaleRows = (x: Value, s: Value): Value => mul(expandDims(s, -1), x)

interface SvdParams {
  readonly method: SvdMethod
}

const svdPrimitive: Primitive<SvdParams> = definePrimitive<SvdParams>({
  name: 'svd',
  abstractEval: (xs) => {
    const a = arg(xs, 0)
    const { batch, m, n } = matrixShape(a.shape, 'svd')
    const k = Math.min(m, n)
    return [
      { shape: [...batch, m, k], dtype: a.dtype },
      { shape: [...batch, k], dtype: a.dtype },
      { shape: [...batch, k, n], dtype: a.dtype },
    ]
  },
  impl: ([a], { method }) => {
    const av = a as NonNullable<typeof a>
    const { batch, m, n } = matrixShape(av.shape, 'svd')
    const k = Math.min(m, n)
    const results = matrices(av, 'svd').map((x) => svdKernel(x, m, n, method))
    const dtype = av.dtype as FloatDType
    const vh = results.map((r) => {
      const t = new Float64Array(k * n)
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) t[j * n + i] = r.v[i * k + j] ?? 0
      return t
    })
    return [
      assemble(
        av,
        dtype,
        batch,
        [m, k],
        results.map((r) => r.u),
      ),
      assemble(
        av,
        dtype,
        batch,
        [k],
        results.map((r) => r.s),
      ),
      assemble(av, dtype, batch, [k, n], vh),
    ]
  },
  // With dS = U^T dA V and F_ij = 1 / (s_j^2 - s_i^2): ds = diag(dS), dU = U (F o (dS S + S dS^T)) (+ the part of dA V
  // outside span(U) when m > k), dV = V (F o (S dS + dS^T S)) (+ the part of dA^T U outside span(V) when n > k).
  jvp: ([a], [ta], _p, [u, s, vh]) => {
    if (ta === null || ta === undefined) return [null, null, null]
    const uv = u as Value
    const sv = s as Value
    const v = mT(vh as Value)
    const { m, n } = matrixShape((a as Value).shape, 'svd')
    const k = Math.min(m, n)
    const f = eigengapInverse(square(sv))
    const av = matmul(ta, v)
    const ds = matmul(mT(uv), av)
    let du = matmul(uv, mul(f, add(scaleColumns(ds, sv), scaleRows(mT(ds), sv))))
    let dv = matmul(v, mul(f, add(scaleRows(ds, sv), scaleColumns(mT(ds), sv))))
    if (m > k) du = add(du, div(sub(av, matmul(uv, ds)), expandDims(sv, -2)))
    if (n > k) dv = add(dv, div(sub(matmul(mT(ta), uv), matmul(v, mT(ds))), expandDims(sv, -2)))
    return [du, diagPart(ds), mT(dv)]
  },
  // The adjoint: with J = F o (U^T dU) and H = F o (V^T dV), dA = U (diag(ds) + (J + J^T) S + S (H + H^T)) V^T, plus
  // (dU - U U^T dU) S^{-1} V^T when m > k and U S^{-1} (dV - V V^T dV)^T when n > k.
  vjp: ([gu, gs, gvh], [a], _p, [u, s, vh]) => {
    const hasU = gu !== null && gu !== undefined
    const hasS = gs !== null && gs !== undefined
    const hasV = gvh !== null && gvh !== undefined
    if (!hasU && !hasS && !hasV) return [null]
    const uv = u as Value
    const sv = s as Value
    const v = mT(vh as Value)
    const { m, n } = matrixShape((a as Value).shape, 'svd')
    const k = Math.min(m, n)
    const terms: Value[] = []
    if (hasS) terms.push(diagEmbed(gs))
    const f = hasU || hasV ? eigengapInverse(square(sv)) : undefined
    if (hasU) {
      const j = mul(f as Value, matmul(mT(uv), gu))
      terms.push(scaleColumns(add(j, mT(j)), sv))
    }
    const gv = hasV ? mT(gvh) : undefined
    if (gv !== undefined) {
      const h = mul(f as Value, matmul(mT(v), gv))
      terms.push(scaleRows(add(h, mT(h)), sv))
    }
    const inner = terms.reduce((x, y) => add(x, y))
    let ga = matmul(matmul(uv, inner), mT(v))
    if (hasU && m > k) {
      const outside = sub(gu, matmul(uv, matmul(mT(uv), gu)))
      ga = add(ga, matmul(div(outside, expandDims(sv, -2)), mT(v)))
    }
    if (gv !== undefined && n > k) {
      const outside = sub(gv, matmul(v, matmul(mT(v), gv)))
      ga = add(ga, matmul(uv, mT(div(outside, expandDims(sv, -2)))))
    }
    return [ga]
  },
  // Factorisations and solves already map over leading batch axes.
  batch: (xs, ds, p, n) => batchLeading(svdPrimitive, xs, ds, p, n),
})

/** Result of {@link svd}. */
export interface SvdResult {
  /** Left singular vectors `[..., m, k]`. */
  readonly u: Value
  /** Singular values, descending `[..., k]`. */
  readonly s: Value
  /** Right singular vectors as rows `[..., k, n]`. */
  readonly vh: Value
}

/**
 * Reduced singular value decomposition A = U diag(s) Vh (numpy.linalg.svd with `full_matrices=False`), k = min(m, n),
 * by one-sided Jacobi rotations (default; accurate small singular values) or Golub-Kahan bidiagonalisation with the
 * Golub-Reinsch implicit QR iteration. Signs are fixed (each column of U has its largest-magnitude entry positive).
 *
 * @param a - Matrix `[..., m, n]`.
 * @param options - The method.
 * @returns `{ u, s, vh }`, differentiable (singular vectors only where singular values are distinct and non-zero).
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @throws MdmlError with code `convergence` for `method: 'golub-kahan'` when the implicit QR sweeps do not deflate
 *   within 75 iterations of one singular value. The result is a bare value of a differentiable primitive, with no
 *   place for a status; the default `jacobi` method has no such limit, and `jacobiSvd` returns a record with `status`.
 * @concept linear-algebra
 * @index function Reduced singular value decomposition, differentiable.
 * @example
 * ```ts
 * import { svd } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(svd([[3, 0], [0, -2]]).s) // => [3, 2]
 * ```
 */
export function svd(a: TensorLike, options: { readonly method?: SvdMethod } = {}): SvdResult {
  const [av] = promoteFloat(a) as [Value]
  const [u, s, vh] = bind(svdPrimitive, [av], { method: options.method ?? 'jacobi' }) as [Value, Value, Value]
  return { u, s, vh }
}

/**
 * Singular values, descending (numpy.linalg.svd with `compute_uv=False`).
 *
 * @param a - Matrix `[..., m, n]`.
 * @returns The singular values, differentiable.
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @example
 * ```ts
 * import { svdvals } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(svdvals([[0, 2], [1, 0]])) // => [2, 1]
 * ```
 */
export function svdvals(a: TensorLike): Value {
  return namingNonFinite('svdvals', () => svd(a).s)
}

/**
 * The Moore-Penrose pseudo-inverse (numpy.linalg.pinv), from the SVD, discarding singular values below
 * `rtol` max(s) (default rtol = max(m, n) eps, numpy's default).
 *
 * @param a - Matrix `[..., m, n]`.
 * @param rtol - Relative cutoff.
 * @returns The pseudo-inverse `[..., n, m]`, differentiable where the rank is locally constant.
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @example
 * ```ts
 * import { pinv } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(pinv([[2, 0], [0, 0]])) // => [[0.5, 0], [0, 0]]
 * ```
 */
export function pinv(a: TensorLike, rtol?: number): Value {
  const [av] = promoteFloat(a) as [Value]
  const { m, n } = matrixShape(av.shape, 'pinv')
  const { u, s, vh } = namingNonFinite('pinv', () => svd(av))
  const cutoff = mul(max(s, -1, true), rtol ?? Math.max(m, n) * 2.220446049250313e-16)
  const keep = greater(s, cutoff)
  const sinv = where(keep, div(1, where(keep, s, 1)), 0)
  return matmul(scaleColumns(mT(vh), sinv), mT(u))
}

/**
 * The best rank-r approximation in the Frobenius and spectral norms (Eckart-Young): the leading r singular triplets.
 *
 * @param a - Matrix `[..., m, n]`.
 * @param rank - The rank r.
 * @returns U_r diag(s_r) Vh_r, differentiable.
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @example
 * ```ts
 * import { lowRankApproximation } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lowRankApproximation([[3, 0], [0, 1]], 1)) // => [[3, 0], [0, 0]]
 * ```
 */
export function lowRankApproximation(a: TensorLike, rank: number): Value {
  const { u, s, vh } = namingNonFinite('lowRankApproximation', () => svd(a))
  const ur = sliceAxis(u, -1, 0, rank)
  const sr = sliceAxis(s, -1, 0, rank)
  const vr = sliceAxis(vh, -2, 0, rank)
  return matmul(scaleColumns(ur, sr), vr)
}

/**
 * An orthonormal basis of the null space of a matrix (scipy.linalg.null_space): the orthogonal complement of the
 * right singular vectors whose singular values exceed `rcond` times the largest.
 *
 * @param a - Matrix `[m, n]`.
 * @param rcond - Relative cutoff (default max(m, n) eps).
 * @returns The basis `[n, nullity]`.
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @example
 * ```ts
 * import { nullSpace } from 'mdml-engine/numerics/linalg'
 * nullSpace([[1, 1]]).shape // => [2, 1]
 * ```
 */
export function nullSpace(a: TensorLike, rcond?: number): Value {
  const [av] = promoteFloat(a) as [Value]
  const { m, n } = matrixShape(av.shape, 'nullSpace')
  const { s, vh } = namingNonFinite('nullSpace', () => svd(av))
  const values = toFlat(primalOf(s))
  const cut = (rcond ?? Math.max(m, n) * 2.220446049250313e-16) * (values[0] ?? 0)
  const rank = values.filter((x) => x > cut).length
  if (rank === n) return sliceAxis(mT(vh), 1, 0, 0)
  // The complement of the row space: the range of the projector I - V_r V_r^T, whose singular values are 1 or 0.
  const vr = mT(sliceAxis(vh, 0, 0, rank))
  const projector = sub(identityOf(n, av), matmul(vr, mT(vr)))
  return sliceAxis(svd(projector).u, 1, 0, n - rank)
}

/**
 * The polar decomposition (scipy.linalg.polar): A = U P (`right`) or A = P U (`left`), with U having orthonormal
 * columns and P symmetric positive semi-definite, from the SVD.
 *
 * @param a - Matrix `[m, n]` with m >= n for `right`.
 * @param side - `right` (default) or `left`.
 * @returns `{ u, p }`, differentiable.
 * @throws MdmlError with code `value` when the input has a NaN or infinite entry.
 * @example
 * ```ts
 * import { polar } from 'mdml-engine/numerics/linalg'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(polar([[2, 0], [0, 3]]).p) // => [[2, 0], [0, 3]]
 * ```
 */
export function polar(a: TensorLike, side: 'right' | 'left' = 'right'): { readonly u: Value; readonly p: Value } {
  const { u, s, vh } = namingNonFinite('polar', () => svd(a))
  const unitary = matmul(u, vh)
  const p = side === 'right' ? matmul(mul(mT(vh), expandDims(s, -2)), vh) : matmul(mul(u, expandDims(s, -2)), mT(u))
  return { u: unitary, p }
}
