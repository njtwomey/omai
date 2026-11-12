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
 * the dimension of `w`), so a choice among eight alternatives with two shared coefficients is a two-dimensional integral, and the rank
 * is at most six (the product grid of `numerics/_product.ts`); the integration and the message are `numerics/_vector-site.ts`.
 */

import { arg, errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, etaOf, massOf, vectorParam } from '../../core/args.ts'
import { defineOperator, siteRule } from '../../core/define.ts'
import { ParameterError } from '../../core/errors.ts'
import { param } from '../../core/params.ts'
import type { FactorOperator, OperatorContext, Params } from '../../core/types.ts'
import { zeroArray } from '../../numerics/_arrays.ts'
import { MOST_PRODUCT } from '../../numerics/_product.ts'
import { type VectorSiteTilt, vectorSiteTilt } from '../../numerics/_vector-site.ts'
import { classesOf, scoreLog } from './_multiclass-scores.ts'

/** The matrix and the offset from the parameters, checked against the number of classes and the dimension of `w`. */
function scoresOf(
  params: Params,
  classes: number,
  d: number,
): { readonly a: readonly number[]; readonly b: readonly number[] } {
  const a = vectorParam(params, 'matrix')
  if (a.length !== classes * d)
    throw new ParameterError(`multiclass-row: matrix must have ${classes} x ${d} entries (a row per class)`)
  const b = params['offset'] === undefined ? zeroArray(classes) : vectorParam(params, 'offset')
  if (b.length !== classes) throw new ParameterError(`multiclass-row: offset must have ${classes} entries`)
  return { a, b }
}

/** The tilted message for the cavity of `w` in the context, memoised under everything it depends on. */
function tilt(c: OperatorContext): VectorSiteTilt {
  const inputs = [c.args.map((a) => a.eta), c.params, c.power]
  return memoised(c, `multiclass-row:${keyOf(inputs)}`, () => {
    const { classes } = classesOf(c.params)
    const w = arg(c, 0)
    const { a, b } = scoresOf(c.params, classes, w.family.dim)
    const log = scoreLog(c.params)
    return vectorSiteTilt(w, {
      coordinates: classes,
      matrix: a,
      offset: b,
      logf: (z) => c.power * log(z),
      name: 'multiclass-row',
    })
  })
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
