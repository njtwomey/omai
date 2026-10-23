import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  arg,
  bind1,
  concreteOrThrow,
  definePrimitive,
  type FloatDType,
  fromStorage,
  item,
  log,
  matmul,
  mul,
  type Primitive,
  promoteFloat,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tril,
  type Value,
} from '../../tensor/index.ts'
import { choleskyKernel } from './_kernels.ts'
import {
  asColumns,
  assemble,
  broadcastBatch,
  diagEmbed,
  diagPart,
  identity,
  matrices,
  mT,
  squareSize,
  sym,
} from './_matrix.ts'
import { triangularSolve } from './triangular.ts'

const lowerSolve = (l: Value, b: Value, transpose = false): Value =>
  triangularSolve(l, b, { lower: true, transpose, unitDiagonal: false })

// Phi(X): the lower triangle with the diagonal halved.
const phi = (x: Value): Value => sub(tril(x), mul(diagEmbed(diagPart(x)), 0.5))

const choleskyPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'cholesky',
  abstractEval: (xs) => [arg(xs, 0)],
  impl: ([a]) => {
    const x = a as NonNullable<typeof a>
    const n = squareSize(x.shape, 'cholesky')
    const parts = matrices(x).map((m) => choleskyKernel(m, n))
    return [assemble(x, x.dtype as FloatDType, x.shape.slice(0, -2), [n, n], parts)]
  },
  // dL = L Phi(L^{-1} sym(dA) L^{-T}) (Murray 2016), with the input tangent symmetrised.
  jvp: (_xs, [ta], _p, [l]) => {
    if (ta === null || ta === undefined) return [null]
    const lv = l as Value
    const m = lowerSolve(lv, mT(lowerSolve(lv, sym(ta))))
    return [matmul(lv, phi(m))]
  },
  // dA = sym(L^{-T} Phi(L^T dL) L^{-1}), the adjoint of the rule above.
  vjp: ([g], _xs, _p, [l]) => {
    if (g === null || g === undefined) return [null]
    const lv = l as Value
    const p = phi(matmul(mT(lv), g))
    const x1 = lowerSolve(lv, p, true)
    return [sym(mT(lowerSolve(lv, mT(x1), true)))]
  },
})

/**
 * The Cholesky factor L of a symmetric positive-definite matrix, A = L L^T (numpy.linalg.cholesky). A is used through
 * its symmetric part (A + A^T) / 2, the same as numpy for symmetric input.
 *
 * @param a - Symmetric positive-definite matrix `[..., n, n]`.
 * @returns Lower-triangular L, differentiable (the derivative treats A as symmetric).
 * @throws WhiteboxError with code `value` when A is not positive definite.
 * @concept linear-algebra
 * @example
 * ```ts
 * import { cholesky } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(cholesky([[4, 2], [2, 2]])) // => [[2, 0], [1, 1]]
 * ```
 */
export function cholesky(a: TensorLike): Value {
  const [av] = promoteFloat(a) as [Value]
  return bind1(choleskyPrimitive, [av], null)
}

/**
 * Solve A x = b given the Cholesky factor L of A (scipy.linalg.cho_solve), by two triangular solves.
 *
 * @param l - Lower Cholesky factor `[..., n, n]`.
 * @param b - Right-hand side `[..., n, k]` or `[n]`.
 * @returns The solution, differentiable in both arguments.
 * @example
 * ```ts
 * import { cholesky, choSolve } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(choSolve(cholesky([[4, 2], [2, 2]]), [6, 4])) // => [1, 1]
 * ```
 */
export function choSolve(l: TensorLike, b: TensorLike): Value {
  const [lv, bv] = promoteFloat(l, b) as [Value, Value]
  const cols = asColumns(lv, bv)
  const [lb, bb] = broadcastBatch(lv, cols.b)
  const x = lowerSolve(lb, lowerSolve(lb, bb), true)
  return cols.vector ? squeeze(x, -1) : x
}

/**
 * log det A from its Cholesky factor: 2 sum log diag(L).
 *
 * @param l - Lower Cholesky factor `[..., n, n]`.
 * @returns log det A, differentiable.
 * @example
 * ```ts
 * import { choleskyLogdet } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * item(choleskyLogdet([[2, 0], [1, 1]])) === 2 * Math.log(2) // => true
 * ```
 */
export function choleskyLogdet(l: TensorLike): Value {
  const [lv] = promoteFloat(l) as [Value]
  return mul(sum(log(diagPart(lv)), -1), 2)
}

/** Result of {@link choleskyJitter}. */
export interface JitteredCholesky {
  readonly l: Value
  /** The multiple of the identity that was added (0 when none was needed). */
  readonly jitter: number
}

/**
 * Cholesky with a jitter ladder: when A is not numerically positive definite, retry with A + j I for j = 1e-10, 1e-9,
 * ... (relative to the mean diagonal) up to `maxJitter`, and report the jitter used (as GP libraries do).
 *
 * @param a - Symmetric matrix `[n, n]`.
 * @param maxJitter - The largest relative jitter to try (default 1e-4).
 * @returns The factor and the jitter added.
 * @throws WhiteboxError with code `value` when even the largest jitter fails.
 * @example
 * ```ts
 * import { choleskyJitter } from 'whitebox-engine/numerics/linalg'
 * choleskyJitter([[1, 1], [1, 1]]).jitter > 0 // => true
 * ```
 */
export function choleskyJitter(a: TensorLike, maxJitter = 1e-4): JitteredCholesky {
  const [av] = promoteFloat(a) as [Value]
  const n = squareSize(av.shape, 'choleskyJitter')
  try {
    return { l: cholesky(av), jitter: 0 }
  } catch {
    // Not positive definite as given: walk up the ladder.
  }
  const scale = Math.abs(sumDiagonal(av)) / Math.max(1, n) || 1
  for (let rel = 1e-10; rel <= maxJitter * (1 + 1e-12); rel *= 10) {
    const jitter = rel * scale
    try {
      return { l: cholesky(add(av, mul(identity(n, av), jitter))), jitter }
    } catch {
      // Try the next rung.
    }
  }
  throw new WhiteboxError('value', `choleskyJitter: not positive definite even with relative jitter ${maxJitter}`)
}

function sumDiagonal(a: Value): number {
  return item(sum(diagPart(a)))
}

/**
 * The Cholesky factor in scipy's cho_factor form (`{ c, lower }`), for {@link choSolve}.
 *
 * @param a - Symmetric positive-definite matrix.
 * @returns The lower factor and `lower: true`.
 * @example
 * ```ts
 * import { choFactor } from 'whitebox-engine/numerics/linalg'
 * choFactor([[4, 2], [2, 2]]).lower // => true
 * ```
 */
export function choFactor(a: TensorLike): { readonly c: Value; readonly lower: true } {
  return { c: cholesky(a), lower: true }
}

function rankOne(l: TensorLike, x: TensorLike, sign: 1 | -1, name: string): Tensor {
  const factor = concreteOrThrow(promoteFloat(l)[0] as Tensor, name)
  const n = factor.shape[0] ?? 0
  const lv = Float64Array.from(factor.data)
  const v = Float64Array.from(concreteOrThrow(promoteFloat(x)[0] as Tensor, name).data)
  for (let k = 0; k < n; k++) {
    const lkk = lv[k * n + k] ?? 0
    const xk = v[k] ?? 0
    const r2 = lkk * lkk + sign * xk * xk
    if (!(r2 > 0)) throw new WhiteboxError('value', `${name}: the result is not positive definite`)
    const r = Math.sqrt(r2)
    const c = r / lkk
    const s = xk / lkk
    lv[k * n + k] = r
    for (let i = k + 1; i < n; i++) {
      lv[i * n + k] = ((lv[i * n + k] ?? 0) + sign * s * (v[i] ?? 0)) / c
      v[i] = c * (v[i] ?? 0) - s * (lv[i * n + k] ?? 0)
    }
  }
  return fromStorage(lv, [n, n], 'float64')
}

/**
 * The Cholesky factor of A + x x^T from that of A, in O(n^2) (the classic rank-one update).
 *
 * @param l - Lower Cholesky factor of A.
 * @param x - Update vector.
 * @returns The updated factor.
 * @example
 * ```ts
 * import { choleskyUpdate } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(choleskyUpdate([[1]], [Math.sqrt(3)])) // => [[2]]
 * ```
 */
export function choleskyUpdate(l: TensorLike, x: TensorLike): Tensor {
  return rankOne(l, x, 1, 'choleskyUpdate')
}

/**
 * The Cholesky factor of A - x x^T from that of A (the rank-one downdate).
 *
 * @param l - Lower Cholesky factor of A.
 * @param x - Downdate vector.
 * @returns The downdated factor.
 * @throws WhiteboxError with code `value` when A - x x^T is not positive definite.
 * @example
 * ```ts
 * import { choleskyDowndate } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(choleskyDowndate([[2]], [Math.sqrt(3)])) // => [[1]]
 * ```
 */
export function choleskyDowndate(l: TensorLike, x: TensorLike): Tensor {
  return rankOne(l, x, -1, 'choleskyDowndate')
}
