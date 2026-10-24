import { WhiteboxError } from '../base/index.ts'
import { eigh, qr, solve, solveContinuousLyapunov, solveDiscreteLyapunov } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  diagonal,
  equal,
  matmul,
  maximum,
  mul,
  primalOf,
  reshape,
  sign,
  sqrt,
  type TensorLike,
  transpose,
  type Value,
} from '../tensor/index.ts'

const equalZero = (x: Value): Value => mul(equal(x, 0), 1)

import { float } from './_shared.ts'
import { columns, complement, complementIn, controllabilityStaircase, fromColumns, intersect } from './_staircase.ts'
import { isDiscrete, type Lti, type StateSpace, toSs } from './representations.ts'
import { isStable } from './stability.ts'

/**
 * The controllability matrix [B, AB, ..., A^{n-1} B] (python-control ctrb), differentiable in A and B.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @returns `[n, n m]`.
 * @concept controllability
 * @example
 * ```ts
 * import { ctrb } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(ctrb([[0, 1], [0, 0]], [[0], [1]])) // => [[0, 1], [1, 0]]
 * ```
 */
export function ctrb(a: TensorLike, b: TensorLike): Value {
  const av = float(a)
  const n = av.shape[0] ?? 0
  const blocks: Value[] = [float(b)]
  for (let k = 1; k < n; k++) blocks.push(matmul(av, blocks[k - 1] as Value))
  return concat(blocks, 1)
}

/**
 * The observability matrix [C; CA; ...; C A^{n-1}] (python-control obsv), differentiable in A and C.
 *
 * @param a - A `[n, n]`.
 * @param c - C `[p, n]`.
 * @returns `[n p, n]`.
 * @concept observability
 * @example
 * ```ts
 * import { obsv } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(obsv([[0, 1], [0, 0]], [[1, 0]])) // => [[1, 0], [0, 1]]
 * ```
 */
export function obsv(a: TensorLike, c: TensorLike): Value {
  return transpose(ctrb(transpose(float(a)), transpose(float(c))))
}

/**
 * Whether a system is controllable: the dimension of its reachable subspace (from an orthogonal staircase reduction,
 * more reliable than the rank of the controllability matrix) is the number of states.
 *
 * @param sys - The system.
 * @param tol - Rank tolerance (default n^2 eps max(|A|, |B|), as SLICOT).
 * @returns `true` when every state is reachable.
 * @concept controllability
 * @example
 * ```ts
 * import { isControllable, ss } from 'whitebox-engine/systems'
 * isControllable(ss([[-1, 0], [0, -2]], [[1], [0]], [[1, 1]])) // => false
 * ```
 */
export function isControllable(sys: Lti, tol?: number): boolean {
  const s = toSs(sys)
  return controllabilityStaircase(s.a, s.b, tol).rank === (s.a.shape[0] ?? 0)
}

/**
 * Whether a system is observable (the dual staircase on A^T, C^T).
 *
 * @param sys - The system.
 * @param tol - Rank tolerance.
 * @returns `true` when every state is seen in the output.
 * @concept observability
 * @example
 * ```ts
 * import { isObservable, ss } from 'whitebox-engine/systems'
 * isObservable(ss([[-1, 0], [0, -2]], [[1], [1]], [[1, 1]])) // => true
 * ```
 */
export function isObservable(sys: Lti, tol?: number): boolean {
  const s = toSs(sys)
  return controllabilityStaircase(transpose(s.a), transpose(s.c), tol).rank === (s.a.shape[0] ?? 0)
}

/** Which Gramian {@link gram} computes: controllability, observability, or their Cholesky factors. */
export type GramianType = 'c' | 'o' | 'cf' | 'of'

/**
 * A Gramian of a stable system (python-control gram): Wc solves A Wc + Wc A^T + B B^T = 0 (A Wc A^T - Wc + B B^T = 0 in
 * discrete time), Wo the dual with C^T C; 'cf' and 'of' give the upper-triangular factor U with W = U^T U.
 * Differentiable in the matrices.
 *
 * @param sys - A stable system.
 * @param type - 'c', 'o', 'cf' or 'of'.
 * @returns The Gramian or its factor, `[n, n]`.
 * @throws WhiteboxError with code `value` for an unstable system.
 * @concept gramians
 * @example
 * ```ts
 * import { gram, ss } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(gram(ss([[-1]], [[2]], [[1]]), 'c')) // => [[2]]
 * ```
 */
export function gram(sys: Lti, type: GramianType): Value {
  const s = toSs(sys)
  if (!isStable(s)) throw new WhiteboxError('value', 'gram: the system must be stable')
  const controllability = type === 'c' || type === 'cf'
  const a = controllability ? s.a : transpose(s.a)
  const q = controllability ? matmul(s.b, transpose(s.b)) : matmul(transpose(s.c), s.c)
  const w = isDiscrete(s) ? solveDiscreteLyapunov(a, q) : solveContinuousLyapunov(a, mul(q, -1))
  if (type === 'c' || type === 'o') return w
  // W = L L^T with L from the symmetric eigendecomposition (semidefinite Gramians included); L^T = Q R gives
  // W = R^T R, with R's diagonal made non-negative.
  const { eigenvalues, eigenvectors } = eigh(w)
  const l = mul(eigenvectors, reshape(sqrt(maximum(eigenvalues, 0)), [1, -1]))
  const { r } = qr(transpose(l))
  const signs = sign(diagonal(primalOf(r)))
  const flip = add(signs, equalZero(signs))
  return mul(reshape(flip, [-1, 1]), r)
}

/** The Kalman decomposition of a state-space system. */
export interface KalmanDecomposition {
  /** The transformed system, states ordered [controllable observable, controllable unobservable, uncontrollable
   * observable, uncontrollable unobservable]. */
  readonly sys: StateSpace
  /** The change of coordinates x = T z. */
  readonly t: Value
  /** The sizes of the four blocks. */
  readonly dims: { readonly co: number; readonly cn: number; readonly uo: number; readonly un: number }
}

/**
 * The Kalman decomposition: coordinates that split the state into controllable-observable, controllable-unobservable,
 * uncontrollable-observable and uncontrollable-unobservable parts, from orthonormal bases of the reachable and the
 * unobservable subspaces (their intersection and complements). In the new coordinates A is block triangular, B has no
 * uncontrollable rows and C no unobservable columns; the controllable-observable block alone has the system's
 * transfer function.
 *
 * @param sys - The system.
 * @param tol - Rank tolerance of the staircase reductions.
 * @returns The transformed system, the transformation and the block sizes.
 * @concept kalman-decomposition
 * @example
 * ```ts
 * import { kalmanDecomposition, ss } from 'whitebox-engine/systems'
 * const sys = ss([[-1, 0, 0], [0, -2, 0], [0, 0, -3]], [[1], [1], [0]], [[1, 0, 1]])
 * kalmanDecomposition(sys).dims // => { co: 1, cn: 1, uo: 1, un: 0 }
 * ```
 */
export function kalmanDecomposition(sys: Lti, tol?: number): KalmanDecomposition {
  const s = toSs(sys)
  const n = s.a.shape[0] ?? 0
  const reach = controllabilityStaircase(s.a, s.b, tol)
  const seen = controllabilityStaircase(transpose(s.a), transpose(s.c), tol)
  const reachable = columns(primalOf(reach.t)).slice(0, reach.rank)
  const unobservable = columns(primalOf(seen.t)).slice(seen.rank)
  const cn = intersect(reachable, unobservable, n)
  const co = complementIn(reachable, cn, n)
  const un = complementIn(unobservable, cn, n)
  const uo = complement([...reachable, ...unobservable], n)
  const t = fromColumns([...co, ...cn, ...uo, ...un], n)
  const a = solve(t, matmul(s.a, t))
  return {
    sys: { kind: 'ss', a, b: solve(t, s.b), c: matmul(s.c, t), d: s.d, dt: s.dt },
    t,
    dims: { co: co.length, cn: cn.length, uo: uo.length, un: un.length },
  }
}
