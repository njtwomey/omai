/**
 * The multiclass site: a categorical outcome `y` among `K` classes whose probabilities are a softmax (`link: 'logit'`, the
 * multinomial logit) or the multinomial probit (`link: 'probit'`: `y` is the class whose utility `z_k + e_k`, with standard normal
 * `e_k`, is the largest) of `K` Gaussian scores `z_k`, one port per class. The factor is a function of the `K` scores jointly, so
 * the tilted moments are a `K`-dimensional integral against the product of the cavities; it is found by adaptive Gauss-Hermite
 * quadrature with a full-covariance proposal (the cavity first, then the Gaussian of the tilted moments just found, three rounds),
 * which is exact for a smooth factor and well behaved when the cavities are much wider than the factor is, as in the first
 * sweeps. A class observed as a known score is a constant coordinate. Up to six free scores are integrated on that product grid
 * (the grid has a node for every combination of the scores, so it stops there); more are integrated by the auxiliary-variable
 * rule of `_multiclass-auxiliary.ts`, which writes the same integral so that the scores do not meet (the softmax's normaliser as
 * a gamma integral over one auxiliary variable, the probit as an integral over the observed class's score and its noise) and costs
 * the number of classes, not its power. The linear multiclass classifier is one such factor per row on the scores `a . w_k` of
 * the rows (the compiler makes each score a Gaussian variable).
 */

import { errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, numberParam } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult, Params, Port } from '../../core/types.ts'
import {
  naturalGradientMessage,
  numericExpectedLog,
  numericMessage,
  quadraturePort,
  quadraturePorts,
} from '../kernel/variational-numeric.ts'
import { probitAuxiliary, softmaxAuxiliary } from './_multiclass-auxiliary.ts'
import {
  classesOf,
  logProbit,
  logSoftmax,
  logSumExp,
  MOST_PRODUCT,
  productTilt,
  scoreLog,
} from './_multiclass-product.ts'
import {
  type ProbitBound,
  probitBound,
  probitScoreExpectation,
  type SoftmaxTable,
  softmaxExpectedLog,
  softmaxScoreExpectation,
  softmaxTable,
} from './_multiclass-variational.ts'

interface Tilt {
  readonly logMass: number
  /** The tilted mean and variance of the free scores, in the order of `free`. */
  readonly mean: readonly number[]
  readonly variance: readonly number[]
  readonly free: readonly number[]
}

/**
 * The tilted mean, covariance and log mass of the free scores against the cavities (the integral over normalised cavities). The
 * cavities of the scores of one factor are the same for every port (each port is a different variable), so the messages to the `K`
 * ports and the evidence of one sweep share one integral: the engine's cache keeps it under everything it depends on.
 */
function tilt(c: OperatorContext): Tilt {
  const inputs = [c.args.map((a) => (a.value !== undefined ? a.value : a.eta)), c.params, c.power, c.rule?.name]
  return memoised(c, `multiclass-tilt:${keyOf(inputs)}`, () => integrate(c))
}

/** The integral itself. */
function integrate(c: OperatorContext): Tilt {
  const { classes, y, probit } = classesOf(c.params)
  const free: number[] = []
  const fixed = new Array<number>(classes).fill(0)
  const cavity: { mean: number; variance: number }[] = []
  for (let k = 0; k < classes; k++) {
    const a = c.args[k] as Arg
    if (a.value !== undefined) fixed[k] = typeof a.value === 'number' ? a.value : Number.NaN
    else {
      free.push(k)
      cavity.push(gaussianOf(a))
    }
  }
  const r = free.length
  if (r === 0) {
    const lp = probit ? logProbit(fixed, y) : logSoftmax(fixed, y)
    return { logMass: c.power * lp, mean: [], variance: [], free }
  }
  const way = c.params['integration'] ?? 'auto'
  if (way === 'auxiliary' || (way !== 'product' && r > MOST_PRODUCT)) {
    const observed = free.indexOf(y)
    const problem = {
      cavities: cavity,
      observed,
      observedScore: fixed[y] as number,
      fixedLogSum: logSumExp(fixed.filter((_, k) => !free.includes(k))),
      fixedOthers: fixed.filter((_, k) => k !== y && !free.includes(k)),
      power: c.power,
    }
    const a = probit ? probitAuxiliary(problem) : softmaxAuxiliary(problem)
    return { logMass: a.logMass, mean: a.mean, variance: a.variance, free }
  }
  const z = fixed.slice()
  const logf = probit ? logProbit : logSoftmax
  // The cavity as a Gaussian: log density at a point of the free scores, its constants made once.
  const cavMean = cavity.map((g) => g.mean)
  const cavHalfPrecision = cavity.map((g) => 0.5 / g.variance)
  const cavConstant = -0.5 * cavity.reduce((a, g) => a + Math.log(2 * Math.PI * g.variance), 0)
  const logCav = (x: readonly number[]): number => {
    let s = cavConstant
    for (let i = 0; i < r; i++) s -= ((x[i] as number) - (cavMean[i] as number)) ** 2 * (cavHalfPrecision[i] as number)
    return s
  }
  const cov = new Array<number>(r * r).fill(0)
  for (let i = 0; i < r; i++) cov[i * r + i] = (cavity[i] as { variance: number }).variance
  const t = productTilt(r, { mean: cavMean, covariance: cov }, logCav, (x) => {
    for (let i = 0; i < r; i++) z[free[i] as number] = x[i] as number
    return c.power * logf(z, y)
  })
  return { logMass: t.logMass, mean: t.mean, variance: t.mean.map((_, i) => t.covariance[i * r + i] as number), free }
}

/** Whether the mean field uses the one-coordinate rule of `_multiclass-variational.ts` (more scores than the product grid reaches, or `integration: "auxiliary"`). */
function variationalAuxiliary(params: Params): boolean {
  const way = params['integration'] ?? 'auto'
  return way === 'auxiliary' || (way !== 'product' && numberParam(params, 'classes', 2) > MOST_PRODUCT)
}

/**
 * The mean-field message and expected log by the auxiliary rule: the beliefs of the scores (a known score has variance zero), the
 * table of the softmax's transforms or the probit's bound, shared by the messages and the bound of one sweep.
 */
function auxiliaryMean(c: OperatorContext): {
  readonly message: (to: number) => readonly number[]
  readonly expectedLog: () => number
} {
  const { y, probit } = classesOf(c.params)
  const beliefs = c.args.map((a) => gaussianOf(a))
  const inputs = [c.args.map((a) => (a.value !== undefined ? a.value : a.eta)), c.params]
  const shared = memoised<{ readonly bound?: ProbitBound; readonly table?: SoftmaxTable }>(
    c,
    `multiclass-vmp:${keyOf(inputs)}`,
    () => (probit ? { bound: probitBound(beliefs, y) } : { table: softmaxTable(beliefs) }),
  )
  return {
    message: (to) => {
      const arg = c.args[to] as Arg
      if (arg.value !== undefined) return [0, 0]
      const port = quadraturePort(arg, { points: 48 })
      const g =
        shared.bound !== undefined
          ? probitScoreExpectation(shared.bound, y, to, port.coords)
          : softmaxScoreExpectation(shared.table as SoftmaxTable, beliefs, y, to, port.coords)
      return naturalGradientMessage(port, g)
    },
    expectedLog: () =>
      shared.bound !== undefined ? shared.bound.value : softmaxExpectedLog(shared.table as SoftmaxTable, beliefs, y),
  }
}

/**
 * A categorical outcome with softmax (or multinomial probit) probabilities of Gaussian class scores: one port per class, the
 * parameters `y` (the observed class), `classes` (their number) and `link`. Expectation propagation: the tilted mean and variance of
 * every free score from an adaptive Gauss-Hermite integral over all of them, so the scores of the classes that were not observed
 * hear about each other (the product grid up to six free scores, the auxiliary-variable rule of `_multiclass-auxiliary.ts` for any
 * number; `integration` picks one, `auto` is the product grid while it reaches). It supports a power (the probit's auxiliary rule
 * takes the factor whole). Variational message passing is non-conjugate (the expected log of a softmax has no
 * closed form in Gaussian scores): the message to a score is the natural gradient of the expected log, by the product grid up to
 * six free scores and for any number by the auxiliary rule of `_multiclass-variational.ts` (the softmax's expected log is exact, an
 * integral over one auxiliary variable of a product of the classes' transforms; the multinomial probit's is a Jensen bound over the
 * other classes' scores, exact when they are known, `support` `numeric` for the softmax and a bound for the probit).
 *
 * @concept factor-operators
 * @index structure A categorical outcome with softmax or multinomial-probit probabilities of Gaussian class scores (adaptive Gauss-Hermite).
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, multiclassSite } from 'omai-engine/message-passing'
 * const args = [0, 1, 2].map(() => messageArg(gaussian, gaussianEta(0, 1)))
 * const m = multiclassSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 0, classes: 3 }, power: 1 })
 * m?.message.map((v) => v.toFixed(3)) // => ['0.000', '0.000']
 * ```
 */
export const multiclassSite: FactorOperator = defineOperator({
  name: 'multiclass-site',
  description:
    'y ~ Categorical(softmax(z)) (or the multinomial probit) of Gaussian class scores z_0..z_(K-1): an adaptive Gauss-Hermite EP site',
  ports: (params: Params): readonly Port[] =>
    Array.from({ length: Math.max(2, Math.floor(numberParam(params, 'classes', 2))) }, (_, k) => ({
      name: `z${k}`,
      accepts: ['gaussian'],
      constant: true,
    })),
  params: {
    classes: param.integer({ min: 2 }),
    y: param.integer({ min: 0 }),
    link: param.text({ optional: true, choices: ['logit', 'probit'] }),
    integration: param.text({ optional: true, choices: ['auto', 'product', 'auxiliary'] }),
  },
  check: (params) => {
    try {
      classesOf(params)
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const t = tilt(c)
      const at = t.free.indexOf(c.to)
      if (at < 0) return { message: [0, 0] }
      const own = etaOf(c.args[c.to] as Arg)
      return { message: gaussianSiteMessage(own, t.mean[at] as number, t.variance[at] as number) }
    }, 'numeric'),
    // The mean field: the non-conjugate message of log P(y | z) to a score under the beliefs of the others (the grid is a product over
    // the free scores, with fewer nodes the more there are). The softmax is log-concave in the scores, so the precisions are positive.
    vmp: messageRule(
      (c): OperatorResult => ({
        message: variationalAuxiliary(c.params)
          ? [...auxiliaryMean(c).message(c.to)]
          : numericMessage(quadraturePorts(c.args), c.to, scoreLog(c.params)),
      }),
      'numeric',
    ),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      let z = tilt(c).logMass
      for (const a of c.args) z += massOf(a)
      return z
    },
    vmp: (c) =>
      variationalAuxiliary(c.params)
        ? auxiliaryMean(c).expectedLog()
        : numericExpectedLog(quadraturePorts(c.args), scoreLog(c.params)),
  },
  traits: { readsTarget: true, power: true },
})
