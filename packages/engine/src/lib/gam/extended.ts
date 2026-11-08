/**
 * Additive models for extended families (Wood, Pya and Saefken 2016): responses whose log-likelihood is known with its
 * derivatives in the linear predictors but is not an exponential-dispersion family, so penalised IRLS does not reach
 * its stationary point (beta regression, ordered categorical, zero-inflated Poisson, a Gaussian with a modelled scale,
 * the multinomial, Cox's partial likelihood). The coefficients maximise `l(beta) - beta' S_lambda beta / 2` by Newton's
 * method on the observed Hessian (not Fisher scoring), made positive definite by flipping and flooring negative
 * eigenvalues, with step halving; the smoothing parameters (and the family's own aux parameters, the beta precision
 * or the ordinal cut points) minimise the Laplace-approximate marginal likelihood `-l + beta' S beta / 2 + log|H| / 2 -
 * log|S|_+ / 2` with `H` the penalised observed information, by Newton steps on `log lambda` with the analytic gradient
 * (the implicit derivative of the coefficients and the derivative of the information along it), a Hessian by
 * differences of gradients, an eigenvalue safeguard and step halving. Several predictors (a mean and a scale) are one
 * coefficient vector with a block per predictor; the effective degrees of freedom and the posterior covariance come
 * from the penalised information. The likelihood is a {@link CoefficientLikelihood}, the interface both the
 * predictor-based families and the Cox partial likelihood implement.
 */

import { maxOf, OmaiError } from '../base/index.ts'
import type { AuxParameter, ExtendedFamily, Link } from '../dists/families/index.ts'
import { matrixView, targetVector } from '../learn/index.ts'
import { ddot, dgemv, dpotrf, dpotrs, dsyev } from '../numerics/linalg/index.ts'
import { analyticObjective, smoothSolvers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { referenceTest } from './inference.ts'
import { logDetDerivative } from './outer.ts'
import {
  assemblePenalty,
  type PenaltyBlock,
  type PenaltyStructure,
  penaltyLogDet,
  penaltyStructure,
} from './penalised.ts'
import { buildDesign, evaluateDesign, type GamDesign, type GamTerm } from './terms.ts'

/** What a likelihood in the coefficients reports at a point. */
export interface CoefficientEvaluation {
  /** The log-likelihood; `-Infinity` where the coefficients are infeasible. */
  readonly logLik: number
  /** Its gradient `[p]`. */
  readonly gradient: Float64Array
  /** The observed information, minus the Hessian, row-major `[p, p]`. */
  readonly information: Float64Array
}

/**
 * A likelihood as a function of the coefficient vector, which is all the penalised Newton iteration and the Laplace
 * criterion need of a model: families with predictors (through the chain rule) and the Cox partial likelihood (a function
 * of the coefficients directly) implement it.
 */
export interface CoefficientLikelihood {
  /** The number of coefficients. */
  readonly p: number
  /** The log-likelihood, gradient and observed information at `beta` and the aux parameters. */
  evaluate(beta: Float64Array, aux: readonly number[]): CoefficientEvaluation
  /** The derivative of the observed information along a direction in the coefficients, row-major `[p, p]`. */
  informationDerivative(beta: Float64Array, aux: readonly number[], direction: Float64Array): Float64Array
}

/** A predictor of a model: its design and where its coefficients sit in the vector. */
export interface ExtendedPredictor {
  readonly name: string
  readonly design: GamDesign
  /** Whether the design's intercept column is left out (the partial likelihood of Cox has no intercept). */
  readonly dropIntercept: boolean
  /** The first coefficient of the predictor in the model's vector. */
  readonly offset: number
  /** The number of coefficients of the predictor. */
  readonly width: number
}

/** A term of the model in the coefficient vector. */
export interface ExtendedTerm {
  readonly predictor: string
  readonly label: string
  readonly offset: number
  readonly width: number
}

/** A model prepared on data: the likelihood, the penalties, the predictors and the starting point. */
export interface ExtendedModel {
  readonly n: number
  readonly likelihood: CoefficientLikelihood
  readonly structure: PenaltyStructure
  readonly predictors: readonly ExtendedPredictor[]
  readonly terms: readonly ExtendedTerm[]
  /** The starting coefficients. */
  readonly start: Float64Array
  readonly aux: readonly AuxParameter[]
  readonly family?: ExtendedFamily
}

/** `X_k' diag(d) X_l` as a `[pk, pl]` block written into `out` at `(r0, c0)` of a matrix with `P` columns. */
function crossBlock(
  out: Float64Array,
  P: number,
  r0: number,
  c0: number,
  Xk: Float64Array,
  pk: number,
  Xl: Float64Array,
  pl: number,
  d: ArrayLike<number>,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const di = d[i] ?? 0
    if (di === 0) continue
    for (let a = 0; a < pk; a++) {
      const da = di * (Xk[i * pk + a] ?? 0)
      if (da === 0) continue
      for (let b = 0; b < pl; b++) {
        const at = (r0 + a) * P + c0 + b
        out[at] = (out[at] ?? 0) + da * (Xl[i * pl + b] ?? 0)
      }
    }
  }
}

/**
 * The likelihood of an extended family with one design matrix per predictor, as a likelihood in the stacked
 * coefficients: the gradient and information by the chain rule through `X_k`, and the derivative of the information
 * along a direction from the family's third derivatives when it has them, else from differences of its Hessian along the
 * direction in the predictors.
 *
 * @param family - The family.
 * @param y - The response `[n]`.
 * @param weights - Prior weights `[n]`.
 * @param designs - For each predictor its design `[n, p_k]`, row-major, and `p_k`.
 * @returns The likelihood.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { etaLikelihood } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const lik = etaLikelihood(families.gaussianLocationScale(), Float64Array.of(1, 2, 3), Float64Array.of(1, 1, 1), [
 *   { X: Float64Array.of(1, 1, 1), p: 1 }, { X: Float64Array.of(1, 1, 1), p: 1 },
 * ])
 * lik.evaluate(Float64Array.of(2, 0), []).gradient.length // => 2
 * ```
 */
export function etaLikelihood(
  family: ExtendedFamily,
  y: Float64Array,
  weights: Float64Array,
  designs: readonly { readonly X: Float64Array; readonly p: number }[],
): CoefficientLikelihood {
  const K = designs.length
  const n = y.length
  const offsets: number[] = []
  let P = 0
  for (const d of designs) {
    offsets.push(P)
    P += d.p
  }
  const etaOf = (beta: ArrayLike<number>): Float64Array[] =>
    designs.map((d, k) =>
      dgemv(
        d.X,
        n,
        d.p,
        Array.from({ length: d.p }, (_, j) => beta[(offsets[k] ?? 0) + j] ?? 0),
      ),
    )
  const hessAt = (k: number, l: number, h: readonly Float64Array[]): Float64Array =>
    h[k <= l ? k * K - (k * (k - 1)) / 2 + (l - k) : l * K - (l * (l - 1)) / 2 + (k - l)] as Float64Array
  const assemble = (blockOf: (k: number, l: number) => ArrayLike<number>): Float64Array => {
    const out = new Float64Array(P * P)
    for (let k = 0; k < K; k++)
      for (let l = k; l < K; l++) {
        const dk = designs[k] as { X: Float64Array; p: number }
        const dl = designs[l] as { X: Float64Array; p: number }
        crossBlock(out, P, offsets[k] ?? 0, offsets[l] ?? 0, dk.X, dk.p, dl.X, dl.p, blockOf(k, l), n)
        if (l > k) {
          for (let a = 0; a < dk.p; a++)
            for (let b = 0; b < dl.p; b++)
              out[((offsets[l] ?? 0) + b) * P + (offsets[k] ?? 0) + a] =
                out[((offsets[k] ?? 0) + a) * P + (offsets[l] ?? 0) + b] ?? 0
        }
      }
    return out
  }
  return {
    p: P,
    evaluate: (beta, aux) => {
      const eta = etaOf(beta)
      const d = family.derivatives(y, weights, eta, aux, 2)
      if (!Number.isFinite(d.logLik)) {
        return { logLik: Number.NEGATIVE_INFINITY, gradient: new Float64Array(P), information: new Float64Array(P * P) }
      }
      const gradient = new Float64Array(P)
      designs.forEach((des, k) => {
        const g = dgemv(des.X, n, des.p, d.grad[k] as Float64Array, true)
        gradient.set(g, offsets[k] ?? 0)
      })
      const information = assemble((k, l) => Float64Array.from(hessAt(k, l, d.hess), (v) => -v))
      return { logLik: d.logLik, gradient, information }
    },
    informationDerivative: (beta, aux, direction) => {
      const eta = etaOf(beta)
      const delta = designs.map((des, k) =>
        dgemv(
          des.X,
          n,
          des.p,
          Array.from({ length: des.p }, (_, j) => direction[(offsets[k] ?? 0) + j] ?? 0),
        ),
      )
      let dh: (k: number, l: number) => Float64Array
      if (family.hasThird) {
        const third = family.derivatives(y, weights, eta, aux, 3).third as readonly Float64Array[]
        dh = (k, l) => {
          const out = new Float64Array(n)
          for (let m = 0; m < K; m++) {
            const t = third[(k * K + l) * K + m] as Float64Array
            const dm = delta[m] as Float64Array
            for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) + (t[i] ?? 0) * (dm[i] ?? 0)
          }
          return out
        }
      } else {
        let reach = 0
        for (const dm of delta) for (const v of dm) reach = Math.max(reach, Math.abs(v))
        let scale = 1
        for (const e of eta) for (const v of e) scale = Math.max(scale, Math.abs(v))
        const t = reach === 0 ? 0 : (1e-5 * scale) / reach
        const shifted = (sign: number): ReturnType<ExtendedFamily['derivatives']> =>
          family.derivatives(
            y,
            weights,
            eta.map((e, k) => Float64Array.from(e, (v, i) => v + sign * t * (delta[k]?.[i] ?? 0))),
            aux,
            2,
          )
        const up = t === 0 ? undefined : shifted(1)
        const dn = t === 0 ? undefined : shifted(-1)
        dh = (k, l) => {
          if (up === undefined || dn === undefined) return new Float64Array(n)
          const a = hessAt(k, l, up.hess)
          const b = hessAt(k, l, dn.hess)
          return Float64Array.from(a, (v, i) => (v - (b[i] ?? 0)) / (2 * t))
        }
      }
      return assemble((k, l) => Float64Array.from(dh(k, l), (v) => -v))
    },
  }
}

/**
 * A likelihood from its value, gradient and information in the coefficients, the derivative of the information along a
 * direction by central differences of the information.
 *
 * @param p - The number of coefficients.
 * @param evaluate - The log-likelihood, gradient and observed information.
 * @returns The likelihood.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { differencedLikelihood } from 'omai-engine/gam'
 * // l(b) = -b^2 / 2 - b^4 / 12: information 1 + b^2
 * const lik = differencedLikelihood(1, (b) => ({
 *   logLik: -(b[0] ** 2) / 2 - b[0] ** 4 / 12, gradient: Float64Array.of(-b[0] - b[0] ** 3 / 3),
 *   information: Float64Array.of(1 + b[0] ** 2),
 * }))
 * Math.round(1e4 * (lik.informationDerivative(Float64Array.of(1), [], Float64Array.of(1))[0] ?? 0)) // => 20000
 * ```
 */
export function differencedLikelihood(
  p: number,
  evaluate: (beta: Float64Array) => CoefficientEvaluation,
): CoefficientLikelihood {
  return {
    p,
    evaluate: (beta) => evaluate(beta),
    informationDerivative: (beta, _aux, direction) => {
      let reach = 0
      for (const v of direction) reach = Math.max(reach, Math.abs(v))
      let scale = 1
      for (const v of beta) scale = Math.max(scale, Math.abs(v))
      const t = reach === 0 ? 0 : (1e-4 * scale) / reach
      if (t === 0) return new Float64Array(p * p)
      const at = (sign: number): Float64Array =>
        evaluate(Float64Array.from(beta, (v, i) => v + sign * t * (direction[i] ?? 0))).information
      const up = at(1)
      const dn = at(-1)
      return Float64Array.from(up, (v, i) => (v - (dn[i] ?? 0)) / (2 * t))
    },
  }
}

/** The penalty structures of several designs as one over their stacked coefficients. */
function combineStructures(
  parts: readonly { readonly structure: PenaltyStructure; readonly shift: number }[],
  p: number,
): PenaltyStructure {
  const blocks: PenaltyBlock[] = []
  const terms: { offset: number; size: number; blocks: number[]; rank: number }[] = []
  let termBase = 0
  for (const { structure, shift } of parts) {
    const base = blocks.length
    let maxTerm = -1
    for (const b of structure.blocks) {
      blocks.push({ ...b, offset: b.offset + shift, term: b.term + termBase })
      maxTerm = Math.max(maxTerm, b.term)
    }
    for (const t of structure.terms) {
      terms.push({ offset: t.offset + shift, size: t.size, blocks: t.blocks.map((b) => b + base), rank: t.rank })
    }
    termBase += maxTerm + 1
  }
  return { p, blocks, terms }
}

/** The options of {@link extendedModel}. */
export interface ExtendedModelOptions {
  /** Prior weights `[n]` (default 1). */
  readonly weights?: TensorLike
}

/**
 * Prepare a model of an extended family: one design per predictor (each list of terms is a predictor's additive
 * formula), the stacked penalties and the starting coefficients (the intercepts at the family's starting predictors).
 *
 * @param x - Covariates `[n, d]`.
 * @param y - The response `[n]`.
 * @param family - The extended family.
 * @param terms - For each predictor of the family, its terms (default one smooth per column for each).
 * @param options - Prior weights.
 * @returns The model.
 * @throws OmaiError `shape` if the number of term lists differs from the number of predictors; `value` for fewer
 *   observations than coefficients or a shape-constrained term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const y = x.map((r) => 0.2 + 0.6 * (r[0] ?? 0))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 5 })]])
 * model.likelihood.p // => 5
 * ```
 */
export function extendedModel(
  x: TensorLike,
  y: TensorLike,
  family: ExtendedFamily,
  terms?: readonly (readonly GamTerm[] | undefined)[],
  options: ExtendedModelOptions = {},
): ExtendedModel {
  const m = matrixView(x, 'gam')
  const response = targetVector(y, 'gam', m.n)
  const K = family.predictors.length
  if (terms !== undefined && terms.length !== K) {
    throw new OmaiError('shape', `gam: ${family.name} has ${K} predictors but ${terms.length} term lists were given`)
  }
  const weights =
    options.weights === undefined ? new Float64Array(m.n).fill(1) : targetVector(options.weights, 'gam', m.n)
  const designs = Array.from({ length: K }, (_, k) => buildDesign(x, terms?.[k]))
  const constants = family.start(response, weights)
  return {
    ...modelFromDesigns(
      m.n,
      designs.map((design, k) => ({ name: family.predictors[k] as string, design, dropIntercept: false })),
      (columns) => etaLikelihood(family, response, weights, columns),
      family.aux(response, weights),
      constants,
    ),
    family,
  }
}

/**
 * Assemble a model from designs and a likelihood in the stacked coefficients: the predictors' columns (the intercept
 * column dropped where a predictor has none, as in Cox's partial likelihood), their stacked penalties, the term map and
 * the starting coefficients (each predictor's intercept at a constant).
 *
 * @param n - The number of observations.
 * @param parts - For each predictor its name, design and whether to drop the intercept column.
 * @param likelihood - The likelihood given the columns each predictor uses.
 * @param aux - The aux parameters.
 * @param constants - The starting value of each predictor's intercept (ignored where it is dropped).
 * @returns The model.
 * @throws OmaiError `value` for a shape-constrained term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, differencedLikelihood, linear, modelFromDesigns } from 'omai-engine/gam'
 * const design = buildDesign([[0], [1], [2]], [linear(0)])
 * const model = modelFromDesigns(3, [{ name: 'mean', design, dropIntercept: false }], (cols) =>
 *   differencedLikelihood(cols[0].p, () => ({ logLik: 0, gradient: new Float64Array(2), information: new Float64Array(4) })), [], [0])
 * model.likelihood.p // => 2
 * ```
 */
export function modelFromDesigns(
  n: number,
  parts: readonly { readonly name: string; readonly design: GamDesign; readonly dropIntercept: boolean }[],
  likelihood: (columns: readonly { readonly X: Float64Array; readonly p: number }[]) => CoefficientLikelihood,
  aux: readonly AuxParameter[],
  constants: readonly number[],
): ExtendedModel {
  const predictors: ExtendedPredictor[] = []
  const flatTerms: ExtendedTerm[] = []
  const shifts: number[] = []
  let P = 0
  for (const { name, design, dropIntercept } of parts) {
    if (design.constraints !== undefined) {
      throw new OmaiError('value', 'gam: shape-constrained terms are not supported by the extended models')
    }
    const drop = dropIntercept ? 1 : 0
    const width = design.p - drop
    predictors.push({ name, design, dropIntercept, offset: P, width })
    shifts.push(P - drop)
    for (const t of design.terms) {
      flatTerms.push({ predictor: name, label: t.label, offset: P + t.offset - drop, width: t.width })
    }
    P += width
  }
  const start = new Float64Array(P)
  predictors.forEach((pr, k) => {
    if (!pr.dropIntercept) start[pr.offset] = constants[k] ?? 0
  })
  const structure = combineStructures(
    parts.map(({ design }, k) => ({ structure: penaltyStructure(design), shift: shifts[k] ?? 0 })),
    P,
  )
  const columns = parts.map(({ design, dropIntercept }) => ({
    X: designColumns(design, dropIntercept),
    p: design.p - (dropIntercept ? 1 : 0),
  }))
  return { n, likelihood: likelihood(columns), structure, predictors, terms: flatTerms, start, aux }
}

/** The inverse and log determinant of a symmetric matrix and whether it had to be made positive definite. */
export interface Inverted {
  readonly inverse: Float64Array
  readonly logDet: number
  readonly modified: boolean
}

/**
 * The inverse and log determinant of a symmetric matrix by Cholesky or, if it is not positive definite, of the matrix
 * with its eigenvalues replaced by their absolute values floored at a fraction of the largest (the perturbation that
 * keeps Newton's step an ascent direction and the Laplace determinant defined).
 *
 * @param A - The matrix, row-major `[p, p]`.
 * @param p - Its order.
 * @returns The inverse, the log determinant and whether it was modified.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { invertSafely } from 'omai-engine/gam'
 * const r = invertSafely(Float64Array.of(2, 0, 0, -2), 2)
 * [r.modified, Array.from(r.inverse)] // => [true, [0.5, 0, 0, 0.5]]
 * ```
 */
export function invertSafely(A: Float64Array, p: number): Inverted {
  const chol = dpotrf(A, p)
  if (chol.info === 0) {
    const inverse = new Float64Array(p * p)
    const e = new Float64Array(p)
    let logDet = 0
    for (let i = 0; i < p; i++) logDet += 2 * Math.log(chol.factor[i * p + i] ?? 1)
    for (let c = 0; c < p; c++) {
      e.fill(0)
      e[c] = 1
      const col = dpotrs(chol.factor, e, p)
      for (let r = 0; r < p; r++) inverse[r * p + c] = col[r] ?? 0
    }
    return { inverse, logDet, modified: false }
  }
  const { w, v } = dsyev(A, p)
  const top = Math.max(1e-300, maxOf(Array.from(w, Math.abs)))
  const floor = 1e-7 * top
  const clipped = Array.from(w, (x) => Math.max(Math.abs(x), floor))
  const inverse = new Float64Array(p * p)
  let logDet = 0
  for (let c = 0; c < p; c++) {
    logDet += Math.log(clipped[c] ?? 1)
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++)
        inverse[i * p + j] = (inverse[i * p + j] ?? 0) + ((v[i * p + c] ?? 0) * (v[j * p + c] ?? 0)) / (clipped[c] ?? 1)
  }
  return { inverse, logDet, modified: true }
}

const quadratic = (S: Float64Array, beta: ArrayLike<number>, p: number): number => ddot(beta, dgemv(S, p, p, beta))

/** A penalised fit at fixed smoothing parameters and aux parameters. */
export interface ExtendedInnerFit {
  readonly beta: Float64Array
  readonly logLik: number
  /** `beta' S beta`. */
  readonly penalty: number
  readonly iterations: number
  /** The infinity norm of the penalised gradient at the end. */
  readonly gradNorm: number
}

/** The options of {@link innerExtendedFit}. */
export interface InnerOptions {
  readonly warm?: ArrayLike<number>
  /**
   * The solver of the slot: a configuration of `smoothSolvers` (gradient descent, conjugate gradient, Newton, BFGS,
   * L-BFGS, coordinate descent or the stochastic `gradient` mode) minimising the penalised negative log-likelihood from
   * the warm start. Without one, the safeguarded Newton iteration of this file (flipped eigenvalues, step halving).
   */
  readonly solver?: SpaceValue
  readonly key?: Key
  readonly tol?: number
  readonly maxIter?: number
}

/**
 * The coefficients maximising the penalised log-likelihood `l(beta) - beta' S beta / 2` at an assembled penalty `S` and
 * aux parameters: Newton's method on the observed Hessian with eigenvalue safeguarding and step halving on the penalised
 * log-likelihood, or any solver of the `smoothSolvers` slot.
 *
 * @param model - The model.
 * @param S - The assembled penalty `[p, p]`.
 * @param aux - The aux parameters.
 * @param options - The warm start, the solver, tolerance and iteration cap.
 * @returns The fit.
 * @throws OmaiError `value` if the log-likelihood is not finite at the starting point.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, innerExtendedFit, assemblePenalty, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 29])
 * const y = x.map((r) => 0.2 + 0.6 * (r[0] ?? 0))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 5 })]])
 * innerExtendedFit(model, assemblePenalty(model.structure, [0]), [2]).gradNorm < 1e-6 // => true
 * ```
 */
export function innerExtendedFit(
  model: ExtendedModel,
  S: Float64Array,
  aux: readonly number[],
  options: InnerOptions = {},
): ExtendedInnerFit {
  const { likelihood } = model
  const p = likelihood.p
  const tol = options.tol ?? 1e-13
  let beta = Float64Array.from(options.warm ?? model.start)
  const objective = (b: Float64Array): number => {
    const ev = likelihood.evaluate(b, aux)
    return Number.isFinite(ev.logLik) ? -ev.logLik + 0.5 * quadratic(S, b, p) : Number.POSITIVE_INFINITY
  }
  if (options.solver !== undefined) {
    const spec = analyticObjective({
      value: (b) => objective(b),
      gradient: (b) => {
        const ev = likelihood.evaluate(b, aux)
        const sb = dgemv(S, p, p, b)
        return Float64Array.from(ev.gradient, (g, i) => -g + (sb[i] ?? 0))
      },
      hessian: (b) => {
        const ev = likelihood.evaluate(b, aux)
        return Float64Array.from(ev.information, (v, i) => v + (S[i] ?? 0))
      },
    })
    const key = options.key ?? makeKey(0)
    const algorithm = smoothSolvers.make({ objective: spec, x0: Array.from(beta) }, options.solver, key)
    const result = runAlgorithm(algorithm, { key, maxSteps: options.maxIter ?? 500 })
    beta = Float64Array.from((result.state as { readonly x: Float64Array }).x)
    const ev = likelihood.evaluate(beta, aux)
    const sb = dgemv(S, p, p, beta)
    return {
      beta,
      logLik: ev.logLik,
      penalty: ddot(beta, sb),
      iterations: result.steps,
      gradNorm: Math.max(0, maxOf(Array.from(ev.gradient, (g, i) => Math.abs(g - (sb[i] ?? 0))))),
    }
  }
  let f = objective(beta)
  if (!Number.isFinite(f))
    throw new OmaiError('value', 'gam: the log-likelihood is not finite at the starting coefficients')
  let iterations = 0
  let gradNorm = Number.POSITIVE_INFINITY
  for (; iterations < (options.maxIter ?? 100); iterations++) {
    const ev = likelihood.evaluate(beta, aux)
    const sb = dgemv(S, p, p, beta)
    const g = Float64Array.from(ev.gradient, (v, i) => v - (sb[i] ?? 0))
    gradNorm = Math.max(0, maxOf(Array.from(g, Math.abs)))
    const A = Float64Array.from(ev.information, (v, i) => v + (S[i] ?? 0))
    const { inverse } = invertSafely(A, p)
    const delta = dgemv(inverse, p, p, g)
    const decrement = ddot(g, delta)
    if (decrement <= tol * (1 + Math.abs(f)) && gradNorm < 1e-9 * (1 + Math.abs(f))) break
    let a = 1
    let next = beta
    let fNext = Number.POSITIVE_INFINITY
    for (let tries = 0; tries < 40; tries++) {
      next = Float64Array.from(beta, (v, i) => v + a * (delta[i] ?? 0))
      fNext = objective(next)
      if (Number.isFinite(fNext) && fNext <= f - 1e-4 * a * decrement + 1e-14 * Math.abs(f)) break
      a /= 2
      fNext = Number.POSITIVE_INFINITY
    }
    if (!Number.isFinite(fNext)) break
    const done = Math.abs(f - fNext) <= 1e-14 * (1 + Math.abs(f)) && decrement <= 1e-8
    beta = next
    f = fNext
    if (done) {
      iterations++
      break
    }
  }
  const ev = likelihood.evaluate(beta, aux)
  const sb = dgemv(S, p, p, beta)
  gradNorm = Math.max(0, maxOf(Array.from(ev.gradient, (v, i) => Math.abs(v - (sb[i] ?? 0)))))
  return { beta, logLik: ev.logLik, penalty: ddot(beta, sb), iterations, gradNorm }
}

/** The criterion of the outer iteration at one point, with its gradient and the quantities inference needs. */
export interface ExtendedEvaluation {
  readonly logLambda: readonly number[]
  readonly aux: readonly number[]
  readonly fit: ExtendedInnerFit
  /** The Laplace-approximate marginal likelihood criterion (lower is better). */
  readonly value: number
  /** The gradient in `log lambda` then in the aux parameters (absent when not asked for). */
  readonly gradient: Float64Array | undefined
  /** `(X'DX + S)^(-1)`, the Bayesian posterior covariance of the coefficients. */
  readonly covariance: Float64Array
  /** The effective degrees of freedom of each coefficient: the diagonal of `H^-1 D`. */
  readonly edfCoef: Float64Array
  readonly edf: number
  /** Whether the penalised information had to be made positive definite. */
  readonly modified: boolean
}

/** The options of {@link evaluateExtended}. */
export interface EvaluateOptions extends InnerOptions {
  /** Compute the gradient (default true). */
  readonly gradient?: boolean
  /** Hold the aux parameters fixed (their gradient is then left out). */
  readonly fixedAux?: boolean
}

function valueOnly(
  model: ExtendedModel,
  logLambda: readonly number[],
  aux: readonly number[],
  warm: Float64Array,
  options: InnerOptions,
): number {
  const S = assemblePenalty(model.structure, logLambda)
  const fit = innerExtendedFit(model, S, aux, { ...options, warm })
  const ev = model.likelihood.evaluate(fit.beta, aux)
  const A = Float64Array.from(ev.information, (v, i) => v + (S[i] ?? 0))
  const { logDet } = invertSafely(A, model.likelihood.p)
  const { logDet: logS, nullDim } = penaltyLogDet(model.structure, logLambda)
  return -ev.logLik + 0.5 * fit.penalty + 0.5 * logDet - 0.5 * logS - (nullDim / 2) * Math.log(2 * Math.PI)
}

/**
 * The Laplace-approximate marginal likelihood (REML) criterion of an extended model at `log lambda` and the aux
 * parameters, after fitting the coefficients: `-l + beta' S beta / 2 + log|H| / 2 - log|S|_+ / 2 - (M/2) log 2 pi`.
 * The gradient in `log lambda` is analytic: `lambda_r beta' S_r beta / 2 + tr(H^-1 (lambda_r S_r + dD/d rho_r)) / 2 -
 * d log|S|_+ / 2` with `dD/d rho_r` the derivative of the observed information along `d beta / d rho_r = -H^-1
 * lambda_r S_r beta`; the gradient in an aux parameter is a central difference of the criterion with the coefficients
 * refitted (warm-started).
 *
 * @param model - The model.
 * @param logLambda - The log smoothing parameters, one per penalty.
 * @param aux - The aux parameters (on their unconstrained scale).
 * @param options - Warm start, solver, whether to compute the gradient, whether the aux parameters are fixed.
 * @returns The evaluation.
 * @concept generalized-additive-models
 * @reference S. N. Wood, N. Pya and B. Saefken, Smoothing parameter and model selection for general smooth models
 *   (JASA 111, 2016).
 * @example
 * ```ts
 * import { evaluateExtended, extendedModel, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r, i) => 0.5 + 0.3 * Math.sin(5 * (r[0] ?? 0)) + 0.05 * Math.sin(i * 7.3))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 6 })]])
 * evaluateExtended(model, [2], [3]).gradient?.length // => 2
 * ```
 */
export function evaluateExtended(
  model: ExtendedModel,
  logLambda: readonly number[],
  aux: readonly number[],
  options: EvaluateOptions = {},
): ExtendedEvaluation {
  const { likelihood, structure } = model
  const p = likelihood.p
  const S = assemblePenalty(structure, logLambda)
  const fit = innerExtendedFit(model, S, aux, options)
  const ev = likelihood.evaluate(fit.beta, aux)
  const A = Float64Array.from(ev.information, (v, i) => v + (S[i] ?? 0))
  const { inverse, logDet, modified } = invertSafely(A, p)
  const { logDet: logS, nullDim } = penaltyLogDet(structure, logLambda)
  const value = -ev.logLik + 0.5 * fit.penalty + 0.5 * logDet - 0.5 * logS - (nullDim / 2) * Math.log(2 * Math.PI)
  // The effective degrees of freedom: diag(H^-1 D) with D the observed information.
  const edfCoef = new Float64Array(p)
  for (let i = 0; i < p; i++) {
    let a = 0
    for (let j = 0; j < p; j++) a += (inverse[i * p + j] ?? 0) * (ev.information[j * p + i] ?? 0)
    edfCoef[i] = a
  }
  const base = {
    logLambda,
    aux,
    fit,
    value,
    covariance: inverse,
    edfCoef,
    edf: edfCoef.reduce((a, b) => a + b, 0),
    modified,
  }
  if (options.gradient === false) return { ...base, gradient: undefined }
  const m = structure.blocks.length
  const q = options.fixedAux === true ? 0 : aux.length
  const gradient = new Float64Array(m + q)
  const dlogdet = logDetDerivative(structure, logLambda)
  structure.blocks.forEach((blk, r) => {
    const lam = Math.exp(logLambda[r] ?? 0)
    const sb = new Float64Array(p)
    for (let i = 0; i < blk.size; i++) {
      let a = 0
      for (let j = 0; j < blk.size; j++) a += (blk.matrix[i * blk.size + j] ?? 0) * (fit.beta[blk.offset + j] ?? 0)
      sb[blk.offset + i] = lam * a
    }
    const betaSb = ddot(fit.beta, sb)
    const direction = dgemv(inverse, p, p, sb).map((v) => -v)
    const dInfo = likelihood.informationDerivative(fit.beta, aux, direction)
    let tr = 0
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) tr += (inverse[i * p + j] ?? 0) * (dInfo[j * p + i] ?? 0)
    for (let i = 0; i < blk.size; i++)
      for (let j = 0; j < blk.size; j++)
        tr += (inverse[(blk.offset + i) * p + blk.offset + j] ?? 0) * lam * (blk.matrix[j * blk.size + i] ?? 0)
    gradient[r] = 0.5 * betaSb + 0.5 * tr - 0.5 * (dlogdet[r] ?? 0)
  })
  const bounds = model.aux
  for (let a = 0; a < q; a++) {
    const h = 1e-3
    const lo = bounds[a]?.lower ?? -Infinity
    const hi = bounds[a]?.upper ?? Infinity
    const up = Math.min((aux[a] ?? 0) + h, hi)
    const dn = Math.max((aux[a] ?? 0) - h, lo)
    const at = (v: number): number =>
      valueOnly(
        model,
        logLambda,
        aux.map((u, i) => (i === a ? v : u)),
        fit.beta,
        { ...options },
      )
    gradient[m + a] = (at(up) - at(dn)) / Math.max(up - dn, 1e-12)
  }
  return { ...base, gradient }
}

/**
 * The Hessian of the criterion in `(log lambda, aux)` at an evaluation, by central differences of the gradient (each
 * refit warm-started), symmetrised.
 *
 * @param model - The model.
 * @param at - The evaluation, with its gradient.
 * @param options - The solver and whether the aux parameters are fixed.
 * @param h - The step (default 0.02).
 * @returns The Hessian, row-major.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { evaluateExtended, extendedHessian, extendedModel, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r, i) => 0.5 + 0.3 * Math.sin(5 * (r[0] ?? 0)) + 0.05 * Math.sin(i * 7.3))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 6 })]])
 * const at = evaluateExtended(model, [2], [3], { fixedAux: true })
 * extendedHessian(model, at, { fixedAux: true }).length // => 1
 * ```
 */
export function extendedHessian(
  model: ExtendedModel,
  at: ExtendedEvaluation,
  options: EvaluateOptions = {},
  h = 0.02,
): Float64Array {
  if (at.gradient === undefined) throw new OmaiError('value', 'extendedHessian: needs the gradient')
  const m = at.logLambda.length
  const d = at.gradient.length
  const out = new Float64Array(d * d)
  for (let s = 0; s < d; s++) {
    const move = (sign: number): Float64Array => {
      const ll = at.logLambda.map((v, r) => (r === s ? v + sign * h : v))
      const ax = at.aux.map((v, a) => (m + a === s ? v + sign * h : v))
      return evaluateExtended(model, ll, ax, { ...options, warm: at.fit.beta }).gradient as Float64Array
    }
    const up = move(1)
    const dn = move(-1)
    for (let r = 0; r < d; r++) out[r * d + s] = ((up[r] ?? 0) - (dn[r] ?? 0)) / (2 * h)
  }
  for (let r = 0; r < d; r++)
    for (let c = 0; c < r; c++) out[r * d + c] = out[c * d + r] = ((out[r * d + c] ?? 0) + (out[c * d + r] ?? 0)) / 2
  return out
}

/** The paths {@link extendedAlgorithm} logs. */
export type ExtendedPath =
  | 'extended/logLik'
  | 'extended/penalisedLogLik'
  | 'extended/criterion'
  | 'extended/gradNorm'
  | 'extended/edf'
  | 'extended/coef'
  | 'extended/logLambda'
  | 'extended/aux'

/** The state of {@link extendedAlgorithm}. */
export type ExtendedState = {
  readonly iteration: number
  readonly beta: number[]
  readonly logLambda: number[]
  readonly aux: number[]
  readonly logLik: number
  readonly penalisedLogLik: number
  readonly criterion: number
  readonly gradNorm: number
  readonly edf: number
  readonly edfCoef: number[]
  readonly converged: boolean
}

/** The contract of {@link extendedAlgorithm}. */
export const extendedContract: Contract<ExtendedState, ExtendedPath> = defineContract({
  'extended/logLik': scalar({ description: 'the log-likelihood of the model' }),
  'extended/penalisedLogLik': scalar({ description: 'the log-likelihood minus the wiggliness penalty / 2' }),
  'extended/criterion': scalar({
    description: 'the Laplace-approximate marginal likelihood (lower is better)',
    direction: 'lower',
  }),
  'extended/gradNorm': scalar({ description: 'the largest absolute gradient of the criterion' }),
  'extended/edf': scalar({ description: 'the total effective degrees of freedom' }),
  'extended/coef': tensorSource({ role: 'param', description: 'the coefficients' }),
  'extended/logLambda': tensorSource({ role: 'param', description: 'the log smoothing parameters' }),
  'extended/aux': tensorSource({ role: 'param', description: 'the family parameters that are not predictors' }),
})

/** The hyperparameters of the extended fit. */
export interface ExtendedHyper {
  /** `reml` selects the smoothing parameters and aux parameters; `fixed` uses `sp` and `aux` as given. */
  readonly method: 'reml' | 'fixed'
  /** Smoothing parameters `lambda`, one per penalty: used as given by `fixed`, as starting values by `reml`. */
  readonly sp: readonly number[] | undefined
  /** The aux parameters (unconstrained scale): given for `fixed`, a start for `reml` (default the family's). */
  readonly aux: readonly number[] | undefined
  /** The most outer iterations (default 60). */
  readonly maxIter: number
  /** The solver of the inner fit: a `smoothSolvers` configuration, or undefined for the safeguarded Newton. */
  readonly solver: SpaceValue | undefined
}

/** The defaults of {@link ExtendedHyper}. */
export const EXTENDED_DEFAULTS: ExtendedHyper = {
  method: 'reml',
  sp: undefined,
  aux: undefined,
  maxIter: 60,
  solver: undefined,
}

const LOG_LAMBDA = { lower: -20, upper: 25 } as const

function startingLogLambda(model: ExtendedModel, sp: readonly number[] | undefined, aux: readonly number[]): number[] {
  const ev = model.likelihood.evaluate(model.start, aux)
  const p = model.likelihood.p
  return model.structure.blocks.map((b, r) => {
    if (sp?.[r] !== undefined) return Math.log(Math.max(sp[r] ?? 1, 1e-300))
    let info = 0
    let pen = 0
    for (let i = 0; i < b.size; i++) {
      info += ev.information[(b.offset + i) * p + b.offset + i] ?? 0
      pen += b.matrix[i * b.size + i] ?? 0
    }
    return Math.min(Math.max(Math.log(Math.max(info, 1e-300) / Math.max(pen, 1e-300)), -6), 10)
  })
}

/**
 * The outer iteration for an extended model as an algorithm, one outer iteration per step: the criterion, its gradient
 * and the Hessian by differences at the converged fit, a Newton step with the eigenvalues of the Hessian made positive
 * (absolute values, floored) and the step capped at 4, halved until the criterion does not rise. It ends when the
 * gradient is below `1e-5` or no step improves the criterion. The state serialises, so a run resumes. With
 * `method: 'fixed'` the first step fits at the given smoothing parameters and ends.
 *
 * @param model - The model.
 * @param hyper - The method, starting or fixed smoothing and aux parameters, the iteration cap and the inner solver.
 * @param key - The run's key (for a stochastic inner solver).
 * @returns The algorithm.
 * @throws OmaiError `shape` if `sp` or `aux` has the wrong length.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { EXTENDED_DEFAULTS, extendedAlgorithm, extendedModel, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 50 }, (_, i) => [i / 49])
 * const y = x.map((r, i) => 0.5 + 0.3 * Math.sin(5 * (r[0] ?? 0)) + 0.04 * Math.sin(i * 7.3))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 6 })]])
 * runAlgorithm(extendedAlgorithm(model, EXTENDED_DEFAULTS), { maxSteps: 40 }).state.converged // => true
 * ```
 */
export function extendedAlgorithm(
  model: ExtendedModel,
  hyper: ExtendedHyper = EXTENDED_DEFAULTS,
  key: Key = makeKey(0),
): Algorithm<ExtendedState, ExtendedPath> {
  const m = model.structure.blocks.length
  const q = model.aux.length
  if (hyper.sp !== undefined && hyper.sp.length !== m) {
    throw new OmaiError('shape', `gam: sp has ${hyper.sp.length} entries for ${m} penalties`)
  }
  if (hyper.aux !== undefined && hyper.aux.length !== q) {
    throw new OmaiError('shape', `gam: aux has ${hyper.aux.length} entries for ${q} parameters`)
  }
  if (hyper.method === 'fixed' && m > 0 && hyper.sp === undefined)
    throw new OmaiError('value', 'gam: method "fixed" needs sp')
  const reml = hyper.method === 'reml'
  const fixedAux = !reml || q === 0
  const inner: InnerOptions = { ...(hyper.solver === undefined ? {} : { solver: hyper.solver }), key }
  const clampLambda = (v: readonly number[]): number[] =>
    v.map((x) => Math.min(Math.max(x, LOG_LAMBDA.lower), LOG_LAMBDA.upper))
  const clampAux = (v: readonly number[]): number[] =>
    v.map((x, a) => Math.min(Math.max(x, model.aux[a]?.lower ?? -Infinity), model.aux[a]?.upper ?? Infinity))
  const stateAt = (ev: ExtendedEvaluation, iteration: number, converged: boolean): ExtendedState => ({
    iteration,
    beta: Array.from(ev.fit.beta),
    logLambda: [...ev.logLambda],
    aux: [...ev.aux],
    logLik: ev.fit.logLik,
    penalisedLogLik: ev.fit.logLik - 0.5 * ev.fit.penalty,
    criterion: ev.value,
    gradNorm: ev.gradient === undefined ? Number.NaN : Math.max(0, maxOf(Array.from(ev.gradient, Math.abs))),
    edf: ev.edf,
    edfCoef: Array.from(ev.edfCoef),
    converged,
  })
  const startAux = (): number[] => clampAux(hyper.aux ?? model.aux.map((a) => a.start))
  return {
    name: 'gam-extended',
    config: {
      method: hyper.method,
      family: model.family?.name ?? 'custom',
      solver: hyper.solver ?? null,
    },
    init: () => {
      const aux = startAux()
      const start = clampLambda(startingLogLambda(model, hyper.sp, aux))
      return stateAt(evaluateExtended(model, start, aux, { ...inner, gradient: false }), 0, false)
    },
    step: (s, ctx) => {
      const aux0 = clampAux(s.aux)
      const current = clampLambda(s.logLambda)
      const warm = Float64Array.from(s.beta)
      let next: ExtendedState
      if (!reml) {
        next = stateAt(
          evaluateExtended(model, current, aux0, { ...inner, warm, gradient: false }),
          s.iteration + 1,
          true,
        )
      } else {
        const options: EvaluateOptions = { ...inner, warm, fixedAux }
        const ev0 = evaluateExtended(model, current, aux0, options)
        const g = ev0.gradient as Float64Array
        const d = g.length
        const gradTol = 1e-5 * Math.max(1, Math.abs(ev0.value) / 100)
        if (d === 0 || Math.max(0, maxOf(Array.from(g, Math.abs))) < gradTol) {
          next = stateAt(ev0, s.iteration + 1, true)
        } else {
          const H = extendedHessian(model, ev0, options)
          const { w, v } = dsyev(H, d)
          const floor = Math.max(1e-8, 1e-8 * maxOf(Array.from(w, Math.abs)))
          const delta = new Float64Array(d)
          for (let k = 0; k < d; k++) {
            let a = 0
            for (let r = 0; r < d; r++) a += (v[r * d + k] ?? 0) * (g[r] ?? 0)
            const c = -a / Math.max(Math.abs(w[k] ?? 0), floor)
            for (let r = 0; r < d; r++) delta[r] = (delta[r] ?? 0) + (v[r * d + k] ?? 0) * c
          }
          const big = Math.max(0, maxOf(Array.from(delta, Math.abs)))
          let a = big > 4 ? 4 / big : 1
          let best: ExtendedEvaluation | undefined
          for (let tries = 0; tries < 12; tries++) {
            const ll = clampLambda(current.map((c, r) => c + a * (delta[r] ?? 0)))
            const ax = clampAux(aux0.map((c, k) => c + a * (delta[m + k] ?? 0)))
            try {
              const ev1 = evaluateExtended(model, ll, ax, { ...options, warm: ev0.fit.beta })
              if (ev1.value <= ev0.value + 1e-10 * (Math.abs(ev0.value) + 1)) {
                best = ev1
                break
              }
            } catch {
              // an infeasible trial: halve the step
            }
            a /= 2
          }
          const gain = best === undefined ? 0 : ev0.value - best.value
          const flat = gain <= 1e-9 * (Math.abs(ev0.value) + 1) && Math.max(0, maxOf(Array.from(g, Math.abs))) < 1e-3
          next =
            best === undefined || gain <= 1e-12 * (Math.abs(ev0.value) + 1)
              ? stateAt(ev0, s.iteration + 1, true)
              : stateAt(best, s.iteration + 1, flat)
        }
      }
      ctx.log('extended/logLik', next.logLik)
      ctx.log('extended/penalisedLogLik', next.penalisedLogLik)
      ctx.log('extended/criterion', next.criterion)
      if (Number.isFinite(next.gradNorm)) ctx.log('extended/gradNorm', next.gradNorm)
      ctx.log('extended/edf', next.edf)
      ctx.log('extended/coef', next.beta)
      if (m > 0) ctx.log('extended/logLambda', next.logLambda)
      if (q > 0) ctx.log('extended/aux', next.aux)
      return next
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ExtendedState,
  }
}

/** A fitted extended model: the final state and the posterior quantities. */
export interface ExtendedFitResult {
  readonly state: ExtendedState
  readonly converged: boolean
  readonly evaluation: ExtendedEvaluation
  /** The Bayesian posterior covariance of the coefficients `[p, p]`. */
  readonly covariance: Float64Array
  /** The effective degrees of freedom of each term, in the order of `model.terms`. */
  readonly edfTerms: number[]
  /** The linear predictors at the data, one `[n]` array per predictor. */
  readonly eta: Float64Array[]
}

/**
 * Fit an extended model: run {@link extendedAlgorithm} to the end and read off the posterior covariance, the effective
 * degrees of freedom per term and the linear predictors.
 *
 * @param model - The model.
 * @param hyper - The hyperparameters (default REML).
 * @param key - The key for a stochastic inner solver.
 * @returns The fit.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, fitExtended, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 50 }, (_, i) => [i / 49])
 * const y = x.map((r, i) => 0.5 + 0.3 * Math.sin(5 * (r[0] ?? 0)) + 0.04 * Math.sin(i * 7.3))
 * const fit = fitExtended(extendedModel(x, y, families.betaRegression(), [[s(0, { k: 6 })]]))
 * fit.converged // => true
 * ```
 */
export function fitExtended(
  model: ExtendedModel,
  hyper: ExtendedHyper = EXTENDED_DEFAULTS,
  key: Key = makeKey(0),
): ExtendedFitResult {
  const result = runAlgorithm(extendedAlgorithm(model, hyper, key), { key, maxSteps: hyper.maxIter })
  const state = result.state
  const evaluation = evaluateExtended(model, state.logLambda, state.aux, {
    warm: Float64Array.from(state.beta),
    gradient: false,
    ...(hyper.solver === undefined ? {} : { solver: hyper.solver }),
    key,
  })
  const edfTerms = model.terms.map((t) => {
    let a = 0
    for (let j = 0; j < t.width; j++) a += evaluation.edfCoef[t.offset + j] ?? 0
    return a
  })
  return {
    state,
    converged: state.converged,
    evaluation,
    covariance: evaluation.covariance,
    edfTerms,
    eta: predictorsAt(model, Float64Array.from(state.beta)),
  }
}

/**
 * The linear predictors of a model at the training data for coefficients `beta`.
 *
 * @param model - The model.
 * @param beta - The coefficients.
 * @returns One `[n]` array per predictor.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, linear, predictorsAt } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const m = extendedModel([[0], [1], [2], [3]], [0.2, 0.4, 0.5, 0.8], families.betaRegression(), [[linear(0)]])
 * predictorsAt(m, Float64Array.of(0, 1))[0]?.length // => 4
 * ```
 */
export function predictorsAt(model: ExtendedModel, beta: Float64Array): Float64Array[] {
  return model.predictors.map((pr) => {
    const X = designColumns(pr.design, pr.dropIntercept)
    const w = pr.width
    return dgemv(X, model.n, w, beta.subarray(pr.offset, pr.offset + w))
  })
}

function designColumns(design: GamDesign, dropIntercept: boolean): Float64Array {
  if (!dropIntercept) return design.X
  const w = design.p - 1
  const out = new Float64Array(design.n * w)
  for (let i = 0; i < design.n; i++) for (let j = 0; j < w; j++) out[i * w + j] = design.X[i * design.p + j + 1] ?? 0
  return out
}

/** A fitted model's predictor, mean and standard error at new rows. */
export interface ExtendedPrediction {
  /** The linear predictors `[K][m]`. */
  readonly eta: Float64Array[]
  /** The standard errors of the linear predictors from the Bayesian posterior covariance `[K][m]`. */
  readonly se: Float64Array[]
  /** The parameters on their natural scale (the inverse links of the predictors) `[K][m]`. */
  readonly parameters: Float64Array[]
}

/**
 * Predict at new covariates: the linear predictors, their standard errors from the posterior covariance and the
 * parameters on their natural scale.
 *
 * @param model - The model.
 * @param fit - The fit.
 * @param x - New covariates `[m, d]`.
 * @returns The prediction.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { extendedModel, fitExtended, predictExtended, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r, i) => 0.5 + 0.2 * (r[0] ?? 0) + 0.03 * Math.sin(i * 7.3))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 5 })]])
 * predictExtended(model, fitExtended(model), [[0.5]]).parameters[0]?.length // => 1
 * ```
 */
export function predictExtended(model: ExtendedModel, fit: ExtendedFitResult, x: TensorLike): ExtendedPrediction {
  const P = model.likelihood.p
  const beta = Float64Array.from(fit.state.beta)
  const eta: Float64Array[] = []
  const se: Float64Array[] = []
  const parameters: Float64Array[] = []
  model.predictors.forEach((pr, k) => {
    const { X: full, n } = evaluateDesign(pr.design, x)
    const X = pr.dropIntercept ? stripIntercept(full, n, pr.design.p) : full
    const w = pr.width
    const e = dgemv(X, n, w, beta.subarray(pr.offset, pr.offset + w))
    eta.push(e)
    const s = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let a = 0
      for (let r = 0; r < w; r++)
        for (let c = 0; c < w; c++)
          a += (X[i * w + r] ?? 0) * (fit.covariance[(pr.offset + r) * P + pr.offset + c] ?? 0) * (X[i * w + c] ?? 0)
      s[i] = Math.sqrt(Math.max(a, 0))
    }
    se.push(s)
    const link = model.family?.links[k] as Link | undefined
    parameters.push(
      link === undefined ? e : Float64Array.from(toFlat(primalOf(link.inverse(fromStorage(e, [n], 'float64'))))),
    )
  })
  return { eta, se, parameters }
}

function stripIntercept(X: Float64Array, n: number, p: number): Float64Array {
  const out = new Float64Array(n * (p - 1))
  for (let i = 0; i < n; i++) for (let j = 0; j < p - 1; j++) out[i * (p - 1) + j] = X[i * p + j + 1] ?? 0
  return out
}

/** The test of one term of an extended model. */
export interface ExtendedTermTest {
  readonly predictor: string
  readonly label: string
  /** The effective degrees of freedom of the term. */
  readonly edf: number
  /** The reference degrees of freedom, `edf1 = 2 tr(F) - tr(F F)` clamped to the term's width. */
  readonly refDf: number
  readonly statistic: number
  /** The p-value under the null that the term's effect is zero (a chi-square reference: there is no scale to estimate). */
  readonly pValue: number
}

/**
 * Tests of the terms of a fitted extended model: Wood's (2013) reference test of each term's fitted effect against zero,
 * from the Bayesian posterior covariance of its coefficients, on the term's `edf1` reference degrees of freedom (a Wald
 * test for an unpenalised term).
 *
 * @param model - The model.
 * @param fit - The fit.
 * @returns One test per term.
 * @concept generalized-additive-models
 * @reference S. N. Wood, On p-values for smooth components of an extended empirical Bayesian GAM (Biometrika 100, 2013).
 * @example
 * ```ts
 * import { extendedModel, extendedTermTests, fitExtended, s } from 'omai-engine/gam'
 * import { families } from 'omai-engine/dists'
 * const x = Array.from({ length: 80 }, (_, i) => [i / 79])
 * const y = x.map((r, i) => Math.min(0.95, Math.max(0.05, 0.2 + 0.6 * (r[0] ?? 0) + 0.1 * Math.sin(i * 12.9898))))
 * const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 5 })]])
 * (extendedTermTests(model, fitExtended(model))[0]?.pValue ?? 1) < 0.01 // => true
 * ```
 */
export function extendedTermTests(model: ExtendedModel, fit: ExtendedFitResult): ExtendedTermTest[] {
  const P = model.likelihood.p
  const beta = Float64Array.from(fit.state.beta)
  const S = assemblePenalty(model.structure, fit.state.logLambda)
  const info = model.likelihood.evaluate(beta, fit.state.aux).information
  const A = Float64Array.from(info, (v, i) => v + (S[i] ?? 0))
  const { inverse } = invertSafely(A, P)
  // F = H^-1 D and edf1_j = 2 F_jj - (F F)_jj
  const F = new Float64Array(P * P)
  for (let i = 0; i < P; i++)
    for (let j = 0; j < P; j++) {
      let a = 0
      for (let k = 0; k < P; k++) a += (inverse[i * P + k] ?? 0) * (info[k * P + j] ?? 0)
      F[i * P + j] = a
    }
  return model.terms.map((t) => {
    const predictor = model.predictors.find((p) => p.name === t.predictor) as ExtendedPredictor
    const columns = designColumns(predictor.design, predictor.dropIntercept)
    const w = t.width
    const local = t.offset - predictor.offset
    const X = new Float64Array(model.n * w)
    for (let i = 0; i < model.n; i++)
      for (let j = 0; j < w; j++) X[i * w + j] = columns[i * predictor.width + local + j] ?? 0
    let edf = 0
    let edf1 = 0
    for (let j = 0; j < w; j++) {
      const at = t.offset + j
      let ff = 0
      for (let k = 0; k < P; k++) ff += (F[at * P + k] ?? 0) * (F[k * P + at] ?? 0)
      edf += F[at * P + at] ?? 0
      edf1 += 2 * (F[at * P + at] ?? 0) - ff
    }
    const V = new Float64Array(w * w)
    for (let a = 0; a < w; a++)
      for (let b = 0; b < w; b++) V[a * w + b] = fit.covariance[(t.offset + a) * P + t.offset + b] ?? 0
    const refDf = Math.min(Math.max(edf1, 1e-3), w)
    const test = referenceTest(X, model.n, w, beta.subarray(t.offset, t.offset + w), V, refDf, 0)
    return { predictor: t.predictor, label: t.label, edf, refDf, statistic: test.statistic, pValue: test.pValue }
  })
}
