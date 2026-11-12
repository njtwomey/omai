/**
 * The multiclass row: a categorical outcome `y` among `K` classes (the softmax, or the multinomial probit) whose `K` scores are
 * `z = A w + b`, a constant matrix and offset of one Gaussian vector `w`. It is the `multiclass-site` with one joint port in place of
 * `K` scalar ones, as `probit-row` is to `probit`: the factor reads the cavity of `w` once, takes the Gaussian cavity of the scores
 * (`N(A m + b, A V A')`, full covariance), integrates the likelihood over it, and sends back to `w` a message in natural parameters
 * whose precision is the full `K x K` matrix mapped through `A`. A site that sends each score its own Gaussian loses what the label
 * says of the scores together (the class slopes of one observation, or the coefficients that every alternative of a choice shares, are
 * correlated in the posterior), so the loop through the shared vector is too confident; this message carries the correlations.
 *
 * The integral runs over the standardised coordinates of the scores' cavity, whose number is its rank (at most the smaller of `K` and
 * the dimension of `w`, found by a pivoted Cholesky factor of `A V A'`), so a choice among eight alternatives with two shared
 * coefficients is a two-dimensional integral, and the rank is at most six (the product grid of `_multiclass-product.ts`).
 */

import { arg, errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, etaOf, massOf, vectorParam } from '../../core/args.ts'
import { defineOperator, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gaussianMvMoments } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, OperatorContext, Params } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
import { classesOf, MOST_PRODUCT, productTilt, scoreLog } from './_multiclass-product.ts'

/** The joint cavity of the scores in the standardised coordinates, and what the integral found there. */
interface RowTilt {
  readonly logMass: number
  /** The message to `w`, `[h, J]` in natural parameters, for the power-adjusted factor (before the division by the power). */
  readonly site: Eta
}

/** The matrix and the offset from the parameters, checked against the number of classes and the dimension of `w`. */
function scoresOf(
  params: Params,
  classes: number,
  d: number,
): { readonly a: readonly number[]; readonly b: readonly number[] } {
  const a = vectorParam(params, 'matrix')
  if (a.length !== classes * d)
    throw new ParameterError(`multiclass-row: matrix must have ${classes} x ${d} entries (a row per class)`)
  const b = params['offset'] === undefined ? new Array<number>(classes).fill(0) : vectorParam(params, 'offset')
  if (b.length !== classes) throw new ParameterError(`multiclass-row: offset must have ${classes} entries`)
  return { a, b }
}

/**
 * A factor `S = B B'` of a symmetric positive semi-definite matrix by pivoted Cholesky: the columns of `B` (`n x rank`, row-major) one
 * at a time, always on the largest remaining diagonal, until what is left is below `tol`. The rank is what the scores' cavity has.
 */
function pivotedFactor(s: readonly number[], n: number, tol: number): { readonly rank: number; readonly b: number[] } {
  const rest = Array.from({ length: n }, (_, i) => s[i * n + i] as number)
  const columns: number[][] = []
  for (let step = 0; step < n; step++) {
    let at = 0
    for (let i = 1; i < n; i++) if ((rest[i] as number) > (rest[at] as number)) at = i
    if (!((rest[at] as number) > tol)) break
    const column = Array.from({ length: n }, (_, i) => s[i * n + at] as number)
    for (const done of columns)
      for (let i = 0; i < n; i++) column[i] = (column[i] as number) - (done[i] as number) * (done[at] as number)
    const root = Math.sqrt(rest[at] as number)
    for (let i = 0; i < n; i++) {
      column[i] = (column[i] as number) / root
      rest[i] = (rest[i] as number) - (column[i] as number) ** 2
    }
    rest[at] = 0
    columns.push(column)
  }
  const rank = columns.length
  const b = new Array<number>(n * rank)
  for (const [j, column] of columns.entries()) for (let i = 0; i < n; i++) b[i * rank + j] = column[i] as number
  return { rank, b }
}

/** The tilted message for the cavity of `w` in the context, memoised under everything it depends on. */
function tilt(c: OperatorContext): RowTilt {
  const inputs = [c.args.map((a) => a.eta), c.params, c.power]
  return memoised(c, `multiclass-row:${keyOf(inputs)}`, () => integrate(c))
}

/** The integral: the scores' cavity, its standardised coordinates, the tilted moments there and the message to `w` they make. */
function integrate(c: OperatorContext): RowTilt {
  const { classes } = classesOf(c.params)
  const w = arg(c, 0)
  const d = w.family.dim
  const { a, b } = scoresOf(c.params, classes, d)
  const { mean: m, covariance: v } = gaussianMvMoments(etaOf(w), d)
  // The scores' cavity: mean A m + b, covariance A V A'.
  const mu = Array.from({ length: classes }, (_, k) => {
    let s = b[k] as number
    for (let e = 0; e < d; e++) s += (a[k * d + e] as number) * (m[e] as number)
    return s
  })
  const av = new Array<number>(classes * d).fill(0)
  for (let k = 0; k < classes; k++)
    for (let e = 0; e < d; e++) {
      let s = 0
      for (let f = 0; f < d; f++) s += (a[k * d + f] as number) * (v[f * d + e] as number)
      av[k * d + e] = s
    }
  const cov = new Array<number>(classes * classes).fill(0)
  for (let k = 0; k < classes; k++)
    for (let l = 0; l <= k; l++) {
      let s = 0
      for (let e = 0; e < d; e++) s += (av[k * d + e] as number) * (a[l * d + e] as number)
      cov[k * classes + l] = s
      cov[l * classes + k] = s
    }
  let scale = 0
  for (let k = 0; k < classes; k++) scale = Math.max(scale, cov[k * classes + k] as number)
  const { rank: r, b: factor } = pivotedFactor(cov, classes, 1e-10 * scale)
  const logf = scoreLog(c.params)
  const zeros = new Array<number>(d + d * d).fill(0)
  if (r === 0) return { logMass: c.power * logf(mu), site: zeros }
  if (r > MOST_PRODUCT)
    throw new ParameterError(`multiclass-row: the scores' cavity has rank ${r}, past the six the product grid reaches`)
  // The scores are mu + B xi with xi standard normal under the cavity: the grid is over xi.
  const z = mu.slice()
  const t = productTilt(
    r,
    {
      mean: new Array<number>(r).fill(0),
      covariance: Array.from({ length: r * r }, (_, i) => (i % (r + 1) === 0 ? 1 : 0)),
    },
    (x) => {
      let s = -0.5 * r * Math.log(2 * Math.PI)
      for (let i = 0; i < r; i++) s -= 0.5 * (x[i] as number) ** 2
      return s
    },
    (x) => {
      for (let k = 0; k < classes; k++) {
        let s = mu[k] as number
        for (let i = 0; i < r; i++) s += (factor[k * r + i] as number) * (x[i] as number)
        z[k] = s
      }
      return c.power * logf(z)
    },
  )
  // The site in the standardised coordinates: the tilted Gaussian over the standard cavity, (S^-1 a, S^-1 - I).
  const tiltedPrecision = spd(t.covariance, r)
  if (tiltedPrecision === undefined)
    throw new ImproperMessageError('multiclass-row: the tilted covariance of the scores is not positive definite')
  const precision = tiltedPrecision.inverse()
  const linear = Array.from(tiltedPrecision.solve(t.mean))
  const lambda = Array.from(precision)
  for (let i = 0; i < r; i++) lambda[i * r + i] = (lambda[i * r + i] as number) - 1
  // xi = G (w - m) with G = (B'B)^-1 B' A: the site as a function of w is exp(linear' G (w - m) - (w - m)' G' lambda G (w - m) / 2).
  const btb = new Array<number>(r * r).fill(0)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++) {
      let s = 0
      for (let k = 0; k < classes; k++) s += (factor[k * r + i] as number) * (factor[k * r + j] as number)
      btb[i * r + j] = s
    }
  const gram = spd(btb, r)
  if (gram === undefined) throw new ImproperMessageError('multiclass-row: the scores have no full-rank factor')
  const g = new Array<number>(r * d)
  for (let e = 0; e < d; e++) {
    const column = new Array<number>(r)
    for (let i = 0; i < r; i++) {
      let s = 0
      for (let k = 0; k < classes; k++) s += (factor[k * r + i] as number) * (a[k * d + e] as number)
      column[i] = s
    }
    const solved = gram.solve(column)
    for (let i = 0; i < r; i++) g[i * d + e] = solved[i] as number
  }
  // J = G' lambda G, h = J m + G' linear.
  const lg = new Array<number>(r * d).fill(0)
  for (let i = 0; i < r; i++)
    for (let e = 0; e < d; e++) {
      let s = 0
      for (let j = 0; j < r; j++) s += (lambda[i * r + j] as number) * (g[j * d + e] as number)
      lg[i * d + e] = s
    }
  const j = new Array<number>(d * d).fill(0)
  for (let e = 0; e < d; e++)
    for (let f = 0; f <= e; f++) {
      let s = 0
      for (let i = 0; i < r; i++) s += (g[i * d + e] as number) * (lg[i * d + f] as number)
      j[e * d + f] = s
      j[f * d + e] = s
    }
  const h = new Array<number>(d).fill(0)
  for (let e = 0; e < d; e++) {
    let s = 0
    for (let f = 0; f < d; f++) s += (j[e * d + f] as number) * (m[f] as number)
    for (let i = 0; i < r; i++) s += (g[i * d + e] as number) * (linear[i] as number)
    h[e] = s
  }
  return { logMass: t.logMass, site: [...h, ...j] }
}

/**
 * A categorical outcome whose `K` class scores are `z = A w + b` of a Gaussian vector `w`: the softmax (`link: 'logit'`) or the
 * multinomial probit (`link: 'probit'`) of the scores, one port (`w`), and the parameters `matrix` (`A`, row-major, a row per class),
 * `offset` (`b`, default zero), `y` (the observed class), `classes` and `link`. Expectation propagation: the scores' cavity is
 * `N(A m + b, A V A')` and the tilted moments are an integral over its rank (the coordinates the scores vary in), so the message to
 * `w` is a full natural-parameter message: the correlations between the scores of one observation reach the vector, which is what
 * the one-message-per-class site cannot do. It supports a power (fractional EP); there is no variational rule (the mean field of
 * `w` has no Gaussian expected log of the softmax in the vector's coordinates; compile the scores as scalars for it).
 *
 * @concept factor-operators
 * @index structure A categorical outcome with softmax or multinomial-probit probabilities of scores that are a matrix times a Gaussian vector.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, messageArg, multiclassRow } from 'omai-engine/message-passing'
 * const w = gaussianMv(3)
 * const args = [messageArg(w, gaussianMvEta([0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1]))]
 * const matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
 * multiclassRow.ep?.({ mode: 'ep', to: 0, args, params: { y: 0, classes: 3, matrix }, power: 1 }).message.length // => 12
 * ```
 */
export const multiclassRow: FactorOperator = defineOperator({
  name: 'multiclass-row',
  description:
    'y ~ Categorical(softmax(A w + b)) (or the multinomial probit) of a Gaussian vector w: one joint EP site over the K scores',
  ports: [{ name: 'w', accepts: ['gaussian-mv'] }],
  params: {
    classes: param.integer({ min: 2 }),
    y: param.integer({ min: 0 }),
    matrix: param.vector(),
    offset: param.vector({ optional: true }),
    link: param.text({ optional: true, choices: ['logit', 'probit'] }),
  },
  check: (params, families) => {
    try {
      const { classes } = classesOf(params)
      const d = families[0]?.dim ?? 0
      scoresOf(params, classes, d)
      if (Math.min(classes, d) > MOST_PRODUCT)
        return `multiclass-row: ${classes} classes of a vector of ${d} are past the six scores the product grid integrates`
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    ep: siteRule((c) => {
      const cavity = etaOf(arg(c, 0))
      const t = tilt(c)
      return { natural: cavity.map((v, i) => v + (t.site[i] ?? 0)) }
    }),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      return tilt(c).logMass + massOf(arg(c, 0))
    },
  },
  noRule: {
    vmp: 'the mean field of a vector has no Gaussian expected log of the softmax in its coordinates: compile the scores as scalar variables (packScalars off) for the variational rule of multiclass-site',
  },
  traits: { readsTarget: true, power: true },
})
