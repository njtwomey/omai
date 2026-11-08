/**
 * Model reduction by balanced truncation: the Hankel singular values that say how much each state contributes to the
 * input-output behaviour, a balanced realisation in which the controllability and observability Gramians are equal and
 * diagonal, and `balred` (python-control), which drops the weakest states or residualises them to keep the DC gain.
 */

import { OmaiError } from '../base/index.ts'
import { blockDiag, eigh, eigvals, signm, solve, svd } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  eye,
  matmul,
  maximum,
  mul,
  primalOf,
  reciprocal,
  reshape,
  sliceAxis,
  sqrt,
  sub,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { numbers } from './_shared.ts'
import { gram } from './analysis.ts'
import { isDiscrete, type Lti, type StateSpace, toSs } from './representations.ts'

// Balanced realisation and truncation by the square-root method: Gramian factors Wc = Lc Lc^T, Wo = Lo Lo^T (from
// symmetric eigendecompositions, so semidefinite Gramians are fine), the SVD Lo^T Lc = U S V^T, and T = Lc V S^{-1/2}.

/** A symmetric positive semidefinite square-root factor L with W = L L^T. */
function factor(w: Value): Value {
  const { eigenvalues, eigenvectors } = eigh(w)
  return mul(eigenvectors, reshape(sqrt(maximum(eigenvalues, 0)), [1, -1]))
}

/**
 * The Hankel singular values of a stable system (python-control hsvd): the square roots of the eigenvalues of Wc Wo,
 * largest first; differentiable in the matrices.
 *
 * @param sys - A stable system.
 * @returns The values `[n]`.
 * @throws OmaiError with code `value` for an unstable system.
 * @concept balanced-truncation
 * @example
 * ```ts
 * import { hsvd, ss } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hsvd(ss([[-1]], [[2]], [[1]]))).map((v) => Math.round(v * 1e12) / 1e12) // => [1]
 * ```
 */
export function hsvd(sys: Lti): Value {
  const lc = factor(gram(sys, 'c'))
  const lo = factor(gram(sys, 'o'))
  return svd(matmul(transpose(lo), lc)).s
}

/** A balanced realisation. */
export interface BalancedRealization {
  /** The balanced system: both Gramians equal diag(hsv). */
  readonly sys: StateSpace
  /** Hankel singular values, largest first. */
  readonly hsv: Value
  /** The change of coordinates x = T z and its inverse. */
  readonly t: Value
  readonly tinv: Value
}

function balanceStable(s: StateSpace): BalancedRealization {
  const lc = factor(gram(s, 'c'))
  const lo = factor(gram(s, 'o'))
  const { u, s: hsv, vh } = svd(matmul(transpose(lo), lc))
  const sv = numbers(hsv)
  const top = sv[0] ?? 0
  if (sv.some((x) => x <= 1e-12 * Math.max(top, 1e-300))) {
    throw new OmaiError('value', 'balanced realisation: the system is not minimal (a Hankel singular value is zero)')
  }
  const inv = reshape(reciprocal(sqrt(hsv)), [1, -1])
  const t = mul(matmul(lc, transpose(vh)), inv)
  const tinv = mul(transpose(inv), matmul(transpose(u), transpose(lo)))
  return {
    sys: { kind: 'ss', a: matmul(tinv, matmul(s.a, t)), b: matmul(tinv, s.b), c: matmul(s.c, t), d: s.d, dt: s.dt },
    hsv,
    t,
    tinv,
  }
}

/**
 * A balanced realisation of a stable minimal system (Matlab balreal, by the square-root method): coordinates in which
 * the controllability and observability Gramians are equal and diagonal, the Hankel singular values.
 *
 * @param sys - A stable, minimal system.
 * @returns The balanced system, the Hankel singular values and the transformation.
 * @throws OmaiError with code `value` for an unstable or non-minimal system.
 * @concept balanced-truncation
 * @example
 * ```ts
 * import { balancedRealization, ss } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(balancedRealization(ss([[-1]], [[2]], [[1]])).sys.b) // => [[1.4142135623730951]]
 * ```
 */
export function balancedRealization(sys: Lti): BalancedRealization {
  return balanceStable(toSs(sys))
}

/** Options for {@link balred}. */
export interface BalredOptions {
  /** 'truncate' (default) drops the weakest balanced states; 'matchdc' residualises them (singular perturbation),
   * keeping the dc gain. */
  readonly method?: 'truncate' | 'matchdc'
}

/** The stable and unstable parts of a system, by the spectral projector (I - sign(M)) / 2. */
function splitStable(s: StateSpace): { stable: StateSpace; unstable: StateSpace } {
  const n = s.a.shape[0] ?? 0
  const { re, im } = eigvals(primalOf(s.a))
  const r = numbers(re)
  const iv = numbers(im)
  const discrete = isDiscrete(s)
  const unstableCount = r.filter((x, i) => (discrete ? Math.hypot(x, iv[i] ?? 0) >= 1 : x >= 0)).length
  if (unstableCount === 0) {
    const m = s.b.shape[1] ?? 0
    const p = s.c.shape[0] ?? 0
    return { stable: s, unstable: { ...s, a: zeros([0, 0]), b: zeros([0, m]), c: zeros([p, 0]) } }
  }
  // Discrete systems through the Cayley map (A - I)(A + I)^{-1}, which sends the unit disc to the left half-plane.
  const m = discrete ? transpose(solve(transpose(add(s.a, eye(n))), transpose(sub(s.a, eye(n))))) : s.a
  const sign = signm(m)
  const projector = mul(sub(eye(n), sign), 0.5)
  // Invariant subspaces: the range of the stable projector and the range of I - P.
  const range = (p: Value, k: number): Value => sliceAxis(svd(p).u, 1, 0, k)
  const vs = range(projector, n - unstableCount)
  const vu = range(sub(eye(n), projector), unstableCount)
  const t = concat([vs, vu], 1)
  const at = solve(t, matmul(s.a, t))
  const bt = solve(t, s.b)
  const ct = matmul(s.c, t)
  const k = n - unstableCount
  const block = (x: Value, r0: number, r1: number, c0: number, c1: number): Value =>
    sliceAxis(sliceAxis(x, 0, r0, r1), 1, c0, c1)
  return {
    stable: {
      kind: 'ss',
      a: block(at, 0, k, 0, k),
      b: sliceAxis(bt, 0, 0, k),
      c: sliceAxis(ct, 1, 0, k),
      d: s.d,
      dt: s.dt,
    },
    unstable: {
      kind: 'ss',
      a: block(at, k, n, k, n),
      b: sliceAxis(bt, 0, k, n),
      c: sliceAxis(ct, 1, k, n),
      d: mul(s.d, 0),
      dt: s.dt,
    },
  }
}

/**
 * Model reduction by balanced truncation (python-control balred): the stable part is balanced and its weakest states
 * (smallest Hankel singular values) are truncated, or residualised with 'matchdc' to keep the dc gain; an unstable
 * part (split off by the matrix sign function) is kept whole, so `order` counts it. Bounded error: the H-infinity
 * norm of the error of truncation is at most twice the sum of the discarded Hankel singular values.
 *
 * @param sys - The system.
 * @param order - The order of the reduced system (at least the number of unstable poles).
 * @param options - The method.
 * @returns The reduced state-space system.
 * @throws OmaiError with code `value` for an order out of range or poles on the stability boundary.
 * @concept balanced-truncation
 * @example
 * ```ts
 * import { balred, dimensions, ss } from 'omai-engine/systems'
 * dimensions(balred(ss([[-1, 0], [0, -10]], [[1], [1]], [[1, 1]]), 1)).states // => 1
 * ```
 */
export function balred(sys: Lti, order: number, options: BalredOptions = {}): StateSpace {
  const s = toSs(sys)
  const n = s.a.shape[0] ?? 0
  const { stable, unstable } = splitStable(s)
  const nu = unstable.a.shape[0] ?? 0
  const r = order - nu
  if (!Number.isInteger(order) || r < 0 || order > n) {
    throw new OmaiError('value', `balred: the order must be an integer from ${nu} (unstable poles) to ${n}`)
  }
  const ns = stable.a.shape[0] ?? 0
  let reduced: StateSpace
  if (ns === 0 || r === ns) reduced = stable
  else {
    const b = balanceStable(stable).sys
    const blk = (x: Value, r0: number, r1: number, c0: number, c1: number): Value =>
      sliceAxis(sliceAxis(x, 0, r0, r1), 1, c0, c1)
    const a11 = blk(b.a, 0, r, 0, r)
    const b1 = sliceAxis(b.b, 0, 0, r)
    const c1 = sliceAxis(b.c, 1, 0, r)
    if ((options.method ?? 'truncate') === 'truncate') {
      reduced = { kind: 'ss', a: a11, b: b1, c: c1, d: b.d, dt: s.dt }
    } else {
      // Residualise the weak states: set x2' = 0 (continuous) or x2[k+1] = x2[k] (discrete) and eliminate them.
      const a12 = blk(b.a, 0, r, r, ns)
      const a21 = blk(b.a, r, ns, 0, r)
      const a22 = blk(b.a, r, ns, r, ns)
      const b2 = sliceAxis(b.b, 0, r, ns)
      const c2 = sliceAxis(b.c, 1, r, ns)
      const m22 = isDiscrete(s) ? sub(eye(ns - r), a22) : mul(a22, -1)
      const x = solve(m22, concat([a21, b2], 1))
      const xa = sliceAxis(x, 1, 0, r)
      const xb = sliceAxis(x, 1, r, r + (b2.shape[1] ?? 0))
      reduced = {
        kind: 'ss',
        a: add(a11, matmul(a12, xa)),
        b: add(b1, matmul(a12, xb)),
        c: add(c1, matmul(c2, xa)),
        d: add(b.d, matmul(c2, xb)),
        dt: s.dt,
      }
    }
  }
  if (nu === 0) return reduced
  return {
    kind: 'ss',
    a: blockDiag(reduced.a, unstable.a),
    b: concat([reduced.b, unstable.b], 0),
    c: concat([reduced.c, unstable.c], 1),
    d: reduced.d,
    dt: s.dt,
  }
}
