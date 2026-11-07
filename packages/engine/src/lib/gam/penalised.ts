/**
 * The numerics of penalised least squares for additive models: the penalty `S_lambda = sum_r lambda_r S_r` assembled from
 * the term blocks, the penalised weighted least-squares solution `(X'WX + S_lambda)^(-1) X'Wz` with its effective degrees
 * of freedom (the trace of the influence matrix, per coefficient and per term), the smoothing-parameter criteria (GCV,
 * UBRE and the Laplace-approximate REML score) and their minimisation over `log lambda`. Everything here works on a
 * working response `z` with weights `w`, which is what one iteration of penalised IRLS presents.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { dgemm, dgemv, dpotrf, dpotrs, dsyev } from '../numerics/linalg/index.ts'
import { nelderMead, qpActiveSet, qpArrays, qpInteriorPoint } from '../optim/index.ts'
import { constraintNullSpace } from '../stats/smoothing/index.ts'
import { asTensor, fromStorage, toFlat, type Value } from '../tensor/index.ts'
import type { ConstraintBlock } from './shape.ts'
import type { GamDesign } from './terms.ts'

/** How the smoothing parameters are chosen. */
export type SmoothingMethod = 'fixed' | 'gcv' | 'ubre' | 'reml' | 'ml'

/** One penalty `lambda_r S_r` of the model, placed at the columns of its term. */
export interface PenaltyBlock {
  /** The first column of the term in the design. */
  readonly offset: number
  /** The width of the term. */
  readonly size: number
  /** The index of the term. */
  readonly term: number
  /** The `size x size` matrix, row-major. */
  readonly matrix: Float64Array
}

/** The penalties of a design, with the structural rank of each term's penalty. */
export interface PenaltyStructure {
  /** The number of coefficients. */
  readonly p: number
  readonly blocks: readonly PenaltyBlock[]
  /** For every term with a penalty: its offset, width, the indices of its blocks and the rank of `sum S_r`. */
  readonly terms: readonly {
    readonly offset: number
    readonly size: number
    readonly blocks: readonly number[]
    readonly rank: number
  }[]
}

/**
 * The penalty structure of a design: one block per penalty matrix of every term, in the order of the terms (the order
 * of the smoothing parameters), and the rank of each term's penalty at unit smoothing parameters.
 *
 * @param design - A built design.
 * @returns The structure.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, penaltyStructure, s } from 'mdml-engine/gam'
 * const design = buildDesign([[0], [1], [2], [3], [4], [5], [6]], [s(0, { k: 5 })])
 * penaltyStructure(design).terms[0]?.rank // => 3
 * ```
 */
export function penaltyStructure(design: GamDesign): PenaltyStructure {
  const blocks: PenaltyBlock[] = []
  const terms: { offset: number; size: number; blocks: number[]; rank: number }[] = []
  design.terms.forEach((t, term) => {
    if (t.penalties.length === 0) return
    const mine: number[] = []
    const total = new Float64Array(t.width * t.width)
    for (const pen of t.penalties) {
      mine.push(blocks.length)
      blocks.push({ offset: t.offset, size: t.width, term, matrix: pen.matrix })
      for (let i = 0; i < total.length; i++) total[i] = (total[i] ?? 0) + (pen.matrix[i] ?? 0)
    }
    const { w } = dsyev(total, t.width)
    const top = Math.max(maxOf(Array.from(w)), 0)
    const rank = Array.from(w).filter((v) => v > 1e-9 * Math.max(top, 1e-300)).length
    terms.push({ offset: t.offset, size: t.width, blocks: mine, rank })
  })
  return { p: design.p, blocks, terms }
}

/**
 * The penalty matrix `sum_r exp(logLambda_r) S_r` over all coefficients.
 *
 * @param structure - The penalty structure.
 * @param logLambda - The log smoothing parameters, one per block.
 * @returns The `p x p` matrix, row-major.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { assemblePenalty, buildDesign, penaltyStructure, s } from 'mdml-engine/gam'
 * const design = buildDesign([[0], [1], [2], [3], [4], [5]], [s(0, { k: 4 })])
 * assemblePenalty(penaltyStructure(design), [0]).length // => 16
 * ```
 */
export function assemblePenalty(structure: PenaltyStructure, logLambda: ArrayLike<number>): Float64Array {
  const p = structure.p
  const out = new Float64Array(p * p)
  structure.blocks.forEach((b, r) => {
    const lam = Math.exp(logLambda[r] ?? 0)
    for (let i = 0; i < b.size; i++)
      for (let j = 0; j < b.size; j++) {
        const at = (b.offset + i) * p + b.offset + j
        out[at] = (out[at] ?? 0) + lam * (b.matrix[i * b.size + j] ?? 0)
      }
  })
  return out
}

/** The sums that stay fixed while the smoothing parameters vary: `X'WX`, `X'Wz` and the data. */
export interface Gram {
  readonly n: number
  readonly p: number
  readonly X: Float64Array
  readonly w: Float64Array
  readonly z: Float64Array
  readonly xtwx: Float64Array
  readonly xtwz: Float64Array
  /** Linear inequality constraints `rows beta >= rhs` the solution must satisfy (shape-constrained terms). */
  readonly constraints?: ConstraintBlock
}

/**
 * Accumulate `X'WX` and `X'Wz` for a working response.
 *
 * @param X - The design `[n, p]`, row-major.
 * @param n - Rows.
 * @param p - Columns.
 * @param w - Working weights `[n]` (zero weights drop the row).
 * @param z - Working response `[n]`.
 * @param constraints - Inequality constraints on the coefficients, if any.
 * @returns The gram.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { gram } from 'mdml-engine/gam'
 * Array.from(gram(Float64Array.of(1, 1, 1, 2), 2, 2, Float64Array.of(1, 1), Float64Array.of(1, 3)).xtwz) // => [4, 7]
 * ```
 */
export function gram(
  X: Float64Array,
  n: number,
  p: number,
  w: Float64Array,
  z: Float64Array,
  constraints?: ConstraintBlock,
): Gram {
  const xtwx = new Float64Array(p * p)
  const xtwz = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    const wi = w[i] ?? 0
    if (wi === 0) continue
    const row = X.subarray(i * p, (i + 1) * p)
    for (let a = 0; a < p; a++) {
      const wa = wi * (row[a] ?? 0)
      xtwz[a] = (xtwz[a] ?? 0) + wa * (z[i] ?? 0)
      for (let b = a; b < p; b++) xtwx[a * p + b] = (xtwx[a * p + b] ?? 0) + wa * (row[b] ?? 0)
    }
  }
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) xtwx[a * p + b] = xtwx[b * p + a] ?? 0
  return { n, p, X, w, z, xtwx, xtwz, ...(constraints === undefined ? {} : { constraints }) }
}

/** A penalised weighted least-squares solution. */
export interface PenalisedSolution {
  readonly beta: Float64Array
  /** The linear predictor `X beta` at the data. */
  readonly eta: Float64Array
  /** The diagonal of the influence matrix `(X'WX + S)^(-1) X'WX`: the effective degrees of freedom of each coefficient. */
  readonly edf: Float64Array
  readonly edfTotal: number
  /** The weighted residual sum of squares `sum w (z - X beta)^2`. */
  readonly rss: number
  /** The penalty `beta' S beta`. */
  readonly penalty: number
  /** `log |X'WX + S|`. */
  readonly logDetA: number
  /** The upper Cholesky factor of `X'WX + S`. */
  readonly factor: Float64Array
  /**
   * `(X'WX + S)^(-1)` restricted to the free directions, when inequality constraints are active at the solution (the
   * Hessian is then inverted on the null space of the active constraints); undefined for an unconstrained solution.
   */
  readonly inverse?: Float64Array
}

/**
 * Solve the penalised weighted least-squares problem `min sum w (z - X beta)^2 + beta' S beta`.
 *
 * @param g - The gram of the working response.
 * @param S - The penalty `[p, p]`.
 * @returns The solution with its effective degrees of freedom.
 * @throws MdmlError `value` if `X'WX + S` is not positive definite (a rank-deficient design with too little penalty).
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { gram, solvePenalised } from 'mdml-engine/gam'
 * const g = gram(Float64Array.of(1, 0, 1, 1, 1, 2), 3, 2, Float64Array.of(1, 1, 1), Float64Array.of(1, 3, 5))
 * Array.from(solvePenalised(g, new Float64Array(4)).beta).map((v) => Math.round(v * 1e6) / 1e6) // => [1, 2]
 * ```
 */
export function solvePenalised(g: Gram, S: Float64Array): PenalisedSolution {
  const { p } = g
  const A = Float64Array.from(g.xtwx)
  for (let i = 0; i < A.length; i++) A[i] = (A[i] ?? 0) + (S[i] ?? 0)
  const { factor } = factorise(A, p)
  const unconstrained = finish(g, S, dpotrs(factor, g.xtwz, p), factor, undefined)
  const c = g.constraints
  if (c === undefined || satisfied(c, unconstrained.beta)) return unconstrained
  return solveConstrained(g, S, A, c)
}

/** The Cholesky factor of a symmetric matrix, with a small ridge added if it is not numerically positive definite. */
function factorise(A: Float64Array, p: number): { factor: Float64Array } {
  let factor = dpotrf(A, p)
  let jitter = 0
  let scale = 0
  for (let i = 0; i < p; i++) scale += A[i * p + i] ?? 0
  scale = scale / Math.max(p, 1) || 1
  for (let attempt = 0; factor.info !== 0 && attempt < 6; attempt++) {
    jitter = jitter === 0 ? 1e-10 * scale : jitter * 10
    const B = Float64Array.from(A)
    for (let i = 0; i < p; i++) B[i * p + i] = (B[i * p + i] ?? 0) + jitter
    factor = dpotrf(B, p)
  }
  if (factor.info !== 0) {
    throw new MdmlError('value', 'gam: the penalised normal equations are singular (the design is rank deficient)')
  }
  return { factor: factor.factor }
}

function satisfied(c: ConstraintBlock, beta: ArrayLike<number>): boolean {
  for (let r = 0; r < c.count; r++) {
    let a = 0
    let mag = 0
    for (let j = 0; j < c.width; j++) {
      a += (c.rows[r * c.width + j] ?? 0) * (beta[j] ?? 0)
      mag += Math.abs((c.rows[r * c.width + j] ?? 0) * (beta[j] ?? 0))
    }
    if (a < (c.rhs[r] ?? 0) - 1e-10 * (1 + mag)) return false
  }
  return true
}

/** The quantities of a solution at coefficients `beta`: the fit, the influence-matrix diagonal and the criteria's terms. */
function finish(
  g: Gram,
  S: Float64Array,
  beta: Float64Array,
  factor: Float64Array,
  inverse: Float64Array | undefined,
  logDet?: number,
): PenalisedSolution {
  const { n, p } = g
  const edf = new Float64Array(p)
  if (inverse === undefined) {
    const column = new Float64Array(p)
    for (let c = 0; c < p; c++) {
      for (let r = 0; r < p; r++) column[r] = g.xtwx[r * p + c] ?? 0
      edf[c] = dpotrs(factor, column, p)[c] ?? 0
    }
  } else {
    for (let i = 0; i < p; i++) {
      let a = 0
      for (let j = 0; j < p; j++) a += (inverse[i * p + j] ?? 0) * (g.xtwx[j * p + i] ?? 0)
      edf[i] = a
    }
  }
  const eta = dgemv(g.X, n, p, beta)
  let rss = 0
  for (let i = 0; i < n; i++) rss += (g.w[i] ?? 0) * ((g.z[i] ?? 0) - (eta[i] ?? 0)) ** 2
  const sb = dgemv(S, p, p, beta)
  let penalty = 0
  let logDetA = 0
  for (let i = 0; i < p; i++) {
    penalty += (beta[i] ?? 0) * (sb[i] ?? 0)
    logDetA += 2 * Math.log(factor[i * p + i] ?? 1)
  }
  return {
    beta,
    eta,
    edf,
    edfTotal: edf.reduce((a, b) => a + b, 0),
    rss,
    penalty,
    logDetA: logDet ?? logDetA,
    factor,
    ...(inverse === undefined ? {} : { inverse }),
  }
}

/** A linearly independent subset of constraint rows (modified Gram-Schmidt), in the order given. */
function independentRows(c: ConstraintBlock, candidates: readonly number[], p: number): number[] {
  const basis: Float64Array[] = []
  const keep: number[] = []
  for (const r of candidates) {
    const v = Float64Array.from(c.rows.subarray(r * p, (r + 1) * p))
    const norm0 = Math.hypot(...v)
    if (norm0 === 0) continue
    for (const b of basis) {
      let d = 0
      for (let j = 0; j < p; j++) d += (v[j] ?? 0) * (b[j] ?? 0)
      for (let j = 0; j < p; j++) v[j] = (v[j] ?? 0) - d * (b[j] ?? 0)
    }
    const norm = Math.hypot(...v)
    if (norm > 1e-8 * norm0) {
      basis.push(v.map((e) => e / norm))
      keep.push(r)
      // No more than p rows can be independent; beyond that the survivors are rounding error of near-parallel grid rows.
      if (keep.length === p) break
    }
  }
  return keep
}

/**
 * The penalised least-squares solution under linear inequality constraints `C beta >= d`: the quadratic program
 * `min (z - X beta)' W (z - X beta) + beta' S beta` by the active-set method. At the solution the active constraints are
 * equalities, so the Hessian is inverted on their null space `N`: `N (N'AN)^(-1) N'` is the posterior covariance (up
 * to the scale), its trace against `X'WX` the effective degrees of freedom, and `log|N'AN|` the determinant the
 * Laplace criterion uses.
 */
function solveConstrained(g: Gram, S: Float64Array, A: Float64Array, c: ConstraintBlock): PenalisedSolution {
  const { p } = g
  let scale = 0
  for (let i = 0; i < p; i++) scale += A[i * p + i] ?? 0
  scale = scale / p || 1
  const qp = qpArrays({
    P: Array.from({ length: p }, (_, i) => Array.from(A.subarray(i * p, (i + 1) * p), (v) => v / scale)),
    q: Array.from(g.xtwz, (v) => -v / scale),
    G: Array.from({ length: c.count }, (_, r) => Array.from(c.rows.subarray(r * p, (r + 1) * p), (v) => -v)),
    h: Array.from(c.rhs, (v) => -v),
  })
  const homogeneous = c.rhs.every((v) => v <= 0)
  // The active-set method is exact; many near-parallel grid rows can make it cycle, and then the interior-point method
  // (which does not mind) finds the optimum and the binding rows are read off the slacks.
  // Near-parallel grid rows make the working set change slowly: give the exact method room before giving up on it.
  let result = qpActiveSet(qp, {
    ...(homogeneous ? { x0: new Float64Array(p) } : {}),
    maxIterations: 20 * (p + c.count),
  })
  let viaInterior = false
  if (result.status !== 'optimal') {
    result = qpInteriorPoint(qp, { tol: 1e-10, maxIterations: 400 })
    viaInterior = true
  }
  if (result.status !== 'optimal') {
    throw new MdmlError('value', `gam: the shape-constrained fit is ${result.status} (are the constraints compatible?)`)
  }
  const beta = Float64Array.from(result.x)
  // The binding constraints: the working set of the active-set method, or the rows with no slack and a positive
  // multiplier; reduced to a linearly independent subset so that the null space has the right dimension.
  let candidates: number[]
  if (viaInterior) {
    candidates = []
    for (let r = 0; r < c.count; r++) {
      let a = 0
      for (let j = 0; j < p; j++) a += (c.rows[r * p + j] ?? 0) * (beta[j] ?? 0)
      if (a - (c.rhs[r] ?? 0) < 1e-6 && (result.z[r] ?? 0) > 1e-9) candidates.push(r)
    }
  } else candidates = [...result.active]
  const rows = independentRows(c, candidates, p)
  const active = Float64Array.from(rows.flatMap((r) => Array.from(c.rows.subarray(r * p, (r + 1) * p))))
  const N = Float64Array.from(toFlat(constraintNullSpace(fromStorage(active, [rows.length, p], 'float64'))))
  const q = N.length / p
  const AN = dgemm(A, N, p, p, q)
  const H = dgemm(N, AN, q, p, q, { transposeA: true })
  const { factor } = factorise(H, q)
  const inverse = new Float64Array(p * p)
  if (q > 0) {
    const e = new Float64Array(q)
    const cols = new Float64Array(q * q)
    for (let col = 0; col < q; col++) {
      e.fill(0)
      e[col] = 1
      const sol = dpotrs(factor, e, q)
      for (let r = 0; r < q; r++) cols[r * q + col] = sol[r] ?? 0
    }
    inverse.set(dgemm(dgemm(N, cols, p, q, q), N, p, q, p, { transposeB: true }))
  }
  let logDet = 0
  for (let i = 0; i < q; i++) logDet += 2 * Math.log(factor[i * q + i] ?? 1)
  return finish(g, S, beta, factor, inverse, logDet)
}

/**
 * The effective degrees of freedom of each term: the sum of the influence-matrix diagonal over the term's columns
 * (the intercept is the first entry of the coefficient-wise `edf`, not a term).
 *
 * @param design - The design.
 * @param edf - The effective degrees of freedom of the coefficients.
 * @returns One value per term, in order.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, linear, termEdf } from 'mdml-engine/gam'
 * termEdf(buildDesign([[1], [2], [3]], [linear(0)]), Float64Array.of(1, 1)) // => [1]
 * ```
 */
export function termEdf(design: GamDesign, edf: ArrayLike<number>): number[] {
  return design.terms.map((t) => {
    let s = 0
    for (let j = 0; j < t.width; j++) s += edf[t.offset + j] ?? 0
    return s
  })
}

/** What the criteria need beyond the solution. */
export interface CriterionContext {
  readonly method: Exclude<SmoothingMethod, 'fixed'>
  /** The observations with a nonzero weight. */
  readonly n: number
  /** The GCV and UBRE inflation of the degrees of freedom (default 1). */
  readonly gamma: number
  /** The known scale; `undefined` when it is estimated (REML profiles it out, UBRE refuses). */
  readonly scale: number | undefined
  readonly structure: PenaltyStructure
  readonly logLambda: ArrayLike<number>
}

/**
 * `log |S_lambda|_+`, the log pseudo-determinant of the assembled penalty, and the number of unpenalised dimensions,
 * from the structural ranks of the term penalties.
 *
 * @param structure - The penalty structure.
 * @param logLambda - The log smoothing parameters, one per block.
 * @returns The log pseudo-determinant and the dimension of the null space.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, penaltyLogDet, penaltyStructure, s } from 'mdml-engine/gam'
 * const design = buildDesign([[0], [1], [2], [3], [4], [5], [6]], [s(0, { k: 5 })])
 * penaltyLogDet(penaltyStructure(design), [0]).nullDim // => 2
 * ```
 */
export function penaltyLogDet(
  structure: PenaltyStructure,
  logLambda: ArrayLike<number>,
): { logDet: number; nullDim: number } {
  let logDet = 0
  let rank = 0
  for (const t of structure.terms) {
    const S = new Float64Array(t.size * t.size)
    for (const b of t.blocks) {
      const blk = structure.blocks[b]
      if (blk === undefined) continue
      const lam = Math.exp(logLambda[b] ?? 0)
      for (let i = 0; i < S.length; i++) S[i] = (S[i] ?? 0) + lam * (blk.matrix[i] ?? 0)
    }
    const w = Array.from(dsyev(S, t.size).w).sort((a, b) => b - a)
    for (let i = 0; i < t.rank; i++) logDet += Math.log(Math.max(w[i] ?? 0, 1e-300))
    rank += t.rank
  }
  return { logDet, nullDim: structure.p - rank }
}

/**
 * The smoothing-parameter criterion of a solution, to be minimised.
 *
 * GCV: `n rss / (n - gamma tau)^2` with `tau` the total effective degrees of freedom. UBRE (Mallows' Cp):
 * `rss / n + 2 scale gamma tau / n - scale`, for a known scale. REML (Wood 2011): the negative log marginal likelihood
 * of the working model with the coefficients integrated out under the improper prior `exp(-beta' S beta / 2 scale)`,
 * `(rss + pen) / (2 scale) + log|X'WX + S| / 2 - log|S|_+ / 2` (plus `(n - M)/2 log(2 pi scale)` when the scale is
 * profiled out as `(rss + pen)/(n - M)`, with `M` the unpenalised dimension).
 *
 * @param sol - The solution at `logLambda`.
 * @param ctx - The method, sample size, inflation, scale and penalties.
 * @returns The criterion (lower is better).
 * @throws MdmlError `value` for UBRE without a known scale.
 * @concept generalized-additive-models
 * @reference S. N. Wood, Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric
 *   generalized linear models (JRSS B 73, 2011).
 * @example
 * ```ts
 * import { gcvScore } from 'mdml-engine/gam'
 * gcvScore({ rss: 10, edfTotal: 5 }, 25, 1) // => 0.625
 * ```
 */
export function gcvScore(sol: Pick<PenalisedSolution, 'rss' | 'edfTotal'>, n: number, gamma = 1): number {
  const denom = n - gamma * sol.edfTotal
  return denom <= 0 ? Number.POSITIVE_INFINITY : (n * sol.rss) / (denom * denom)
}

/**
 * The UBRE (unbiased risk estimate) criterion for a known scale.
 *
 * @param sol - The solution.
 * @param n - The number of observations.
 * @param scale - The known scale.
 * @param gamma - The inflation of the degrees of freedom (default 1).
 * @returns The criterion (lower is better).
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { ubreScore } from 'mdml-engine/gam'
 * ubreScore({ rss: 10, edfTotal: 5 }, 20, 1) // => 0
 * ```
 */
export function ubreScore(
  sol: Pick<PenalisedSolution, 'rss' | 'edfTotal'>,
  n: number,
  scale: number,
  gamma = 1,
): number {
  return sol.rss / n - scale + (2 * scale * gamma * sol.edfTotal) / n
}

/**
 * The REML score of a solution (see {@link gcvScore} for the formula).
 *
 * @param sol - The solution.
 * @param ctx - The sample size, the known scale if any, and the penalties at `logLambda`.
 * @returns The criterion (lower is better).
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, penaltyStructure, remlScore, s, solvePenalised, gram, assemblePenalty } from 'mdml-engine/gam'
 * const x = Array.from({ length: 20 }, (_, i) => [i / 19])
 * const y = x.map((r) => Math.sin(6 * (r[0] ?? 0)))
 * const design = buildDesign(x, [s(0, { k: 6 })])
 * const structure = penaltyStructure(design)
 * const sol = solvePenalised(gram(design.X, 20, design.p, new Float64Array(20).fill(1), Float64Array.from(y)), assemblePenalty(structure, [0]))
 * Number.isFinite(remlScore(sol, { n: 20, scale: undefined, structure, logLambda: [0] })) // => true
 * ```
 */
export function remlScore(
  sol: PenalisedSolution,
  ctx: Pick<CriterionContext, 'n' | 'scale' | 'structure' | 'logLambda'> & { readonly ml?: boolean },
): number {
  const { logDet, nullDim } = penaltyLogDet(ctx.structure, ctx.logLambda)
  const dev = sol.rss + sol.penalty
  const half = 0.5 * (sol.logDetA - logDet)
  if (ctx.scale !== undefined) return dev / (2 * ctx.scale) + half
  const dof = ctx.n - (ctx.ml === true ? 0 : nullDim)
  const phi = Math.max(dev / dof, 1e-300)
  return dof / 2 + (dof / 2) * Math.log(2 * Math.PI * phi) + half
}

function criterionOf(sol: PenalisedSolution, ctx: CriterionContext): number {
  switch (ctx.method) {
    case 'gcv':
      return gcvScore(sol, ctx.n, ctx.gamma)
    case 'ubre':
      if (ctx.scale === undefined)
        throw new MdmlError('value', 'gam: UBRE needs a known scale (a binomial or Poisson family, or scale)')
      return ubreScore(sol, ctx.n, ctx.scale, ctx.gamma)
    case 'reml':
      return remlScore(sol, ctx)
    case 'ml':
      return remlScore(sol, { ...ctx, ml: true })
  }
}

/** The result of {@link selectLogLambda}. */
export interface LambdaSelection {
  readonly logLambda: number[]
  readonly value: number
  readonly evaluations: number
}

/** The box `log lambda` is searched in. */
const LOG_LAMBDA_BOUNDS = { lower: -25, upper: 30 } as const

/**
 * Choose the smoothing parameters for a working response: minimise the criterion over `log lambda` from a starting
 * point with the Nelder-Mead simplex (the dimension is the number of penalties, a handful).
 *
 * @param g - The gram of the working response.
 * @param structure - The penalties.
 * @param start - The starting `log lambda`, one per block.
 * @param method - `gcv`, `ubre` or `reml`.
 * @param options - `gamma` (default 1) and the known `scale`.
 * @returns The chosen `log lambda`, its criterion value and the number of evaluations.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, gram, penaltyStructure, s, selectLogLambda } from 'mdml-engine/gam'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const y = Float64Array.from(x, (r) => Math.sin(6 * (r[0] ?? 0)) + 0.01 * ((r[0] ?? 0) * 7 % 1 - 0.5))
 * const design = buildDesign(x, [s(0, { k: 8 })])
 * const g = gram(design.X, 30, design.p, new Float64Array(30).fill(1), y)
 * selectLogLambda(g, penaltyStructure(design), [0], 'gcv').logLambda.length // => 1
 * ```
 */
export function selectLogLambda(
  g: Gram,
  structure: PenaltyStructure,
  start: readonly number[],
  method: Exclude<SmoothingMethod, 'fixed'>,
  options: { readonly gamma?: number; readonly scale?: number } = {},
): LambdaSelection {
  const m = structure.blocks.length
  if (m === 0) return { logLambda: [], value: 0, evaluations: 0 }
  let n = 0
  for (let i = 0; i < g.n; i++) if ((g.w[i] ?? 0) > 0) n++
  const evaluate = (logLambda: ArrayLike<number>): number => {
    const sol = solvePenalised(g, assemblePenalty(structure, logLambda))
    return criterionOf(sol, { method, n, gamma: options.gamma ?? 1, scale: options.scale, structure, logLambda })
  }
  const x0 = Array.from({ length: m }, (_, r) =>
    Math.min(Math.max(start[r] ?? 0, LOG_LAMBDA_BOUNDS.lower), LOG_LAMBDA_BOUNDS.upper),
  )
  // A simplex of unit steps in log lambda (Nelder-Mead's default is 5% of the coordinate, nothing at zero).
  const simplex = [
    x0,
    ...x0.map((v, r) => x0.map((u, c) => (c === r ? v + (v + 2 > LOG_LAMBDA_BOUNDS.upper ? -2 : 2) : u))),
  ]
  const result = nelderMead(
    (x) => {
      const value = evaluate(Array.from(toFlat(x as Value)))
      return asTensor(Number.isFinite(value) ? value : 1e300)
    },
    x0,
    { xatol: 1e-4, fatol: 1e-9, initialSimplex: simplex, bounds: LOG_LAMBDA_BOUNDS, maxEvaluations: 400 * m },
  )
  return { logLambda: Array.from(toFlat(result.x as Value)), value: result.fun, evaluations: result.nfev }
}

/**
 * Solve with the penalised Hessian of a solution, `(X'WX + S)^(-1) b`.
 *
 * @param sol - A solution.
 * @param b - The right-hand side `[p]`.
 * @returns The solution of the linear system.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { gram, hessianSolve, solvePenalised } from 'mdml-engine/gam'
 * const g = gram(Float64Array.of(1, 1, 1), 3, 1, Float64Array.of(1, 1, 1), Float64Array.of(1, 2, 3))
 * Array.from(hessianSolve(solvePenalised(g, new Float64Array(1)), [3])) // => [1]
 * ```
 */
export function hessianSolve(sol: PenalisedSolution, b: ArrayLike<number>): Float64Array {
  if (sol.inverse === undefined) return dpotrs(sol.factor, b, b.length)
  return dgemv(sol.inverse, b.length, b.length, b)
}

/**
 * The Bayesian posterior covariance of the coefficients, `scale (X'WX + S)^(-1)`, as a `[p, p]` row-major array.
 *
 * @param sol - A solution.
 * @param p - The number of coefficients.
 * @param scale - The scale.
 * @returns The covariance.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { gram, posteriorCovariance, solvePenalised } from 'mdml-engine/gam'
 * const g = gram(Float64Array.of(1, 1, 1), 3, 1, Float64Array.of(1, 1, 1), Float64Array.of(1, 2, 3))
 * Array.from(posteriorCovariance(solvePenalised(g, new Float64Array(1)), 1, 1)) // => [0.333333]
 * ```
 */
export function posteriorCovariance(sol: PenalisedSolution, p: number, scale: number): Float64Array {
  const out = new Float64Array(p * p)
  const e = new Float64Array(p)
  for (let c = 0; c < p; c++) {
    e.fill(0)
    e[c] = 1
    const col = hessianSolve(sol, e)
    for (let r = 0; r < p; r++) out[r * p + c] = scale * (col[r] ?? 0)
  }
  return out
}
