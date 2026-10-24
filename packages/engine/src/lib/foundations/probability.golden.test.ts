import { goldenSuite } from '../../testing/suite.ts'
import { div, exp, log, mul, neg, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const die = F.probabilitySpace([1, 2, 3, 4, 5, 6])
const dice = F.powerSpace(die, 2)
const pmfTensor = (p: F.Pmf): Value => tensor([p.values, p.probabilities])
const at = (r: { readonly s: number[]; readonly density: number[] }, v: number): number => {
  const [s0 = 0, s1 = 1] = r.s
  return r.density[Math.round((v - s0) / (s1 - s0))] ?? Number.NaN
}
const normal = (x: Value): Value => div(exp(neg(mul(mul(x, x), 0.5))), Math.sqrt(2 * Math.PI))

goldenSuite('probability against sympy.stats and scipy.stats', 'foundations/probability', {
  diceEvent: () => F.eventProbability(dice, ([a = 0, b = 0]) => a + b > 7),
  diceConditional: () =>
    F.conditionalProbability(
      dice,
      ([a = 0, b = 0]) => a + b === 8,
      ([a]) => a === 3,
    ),
  diceConditionalEven: () =>
    F.conditionalProbability(
      dice,
      ([a = 0, b = 0]) => a + b > 8,
      ([a = 0]) => a > 3,
    ),
  diceExpectProduct: () => F.expectation(dice, ([a = 0, b = 0]) => a * b),
  diceVarianceSum: () => F.varianceOf(dice, ([a = 0, b = 0]) => a + b),
  diceCovariance: () =>
    F.covarianceOf(
      dice,
      ([a = 0]) => a,
      ([a = 0, b = 0]) => a + b,
    ),
  diceSum3: () => pmfTensor(F.diceSum(3)),
  diceIndependence: () => {
    const r = F.independent(
      dice,
      ([a = 0]) => a > 4,
      ([a = 0, b = 0]) => a + b > 9,
    )
    return tensor([r.pA, r.pB, r.pAB])
  },
  coinSum: () => {
    const coin = F.probabilitySpace([0, 1], [2, 1])
    return pmfTensor(F.distributionOf(F.powerSpace(coin, 4), (o) => o.reduce((s, v) => s + v, 0)))
  },
  binomialConvolution: () => {
    const bernoulli: F.Pmf = { values: [0, 1], probabilities: [0.7, 0.3] }
    let total: F.Pmf = { values: [0], probabilities: [1] }
    for (let i = 0; i < 8; i++) total = F.sumPmf(total, bernoulli)
    return tensor(total.probabilities)
  },
  bayes: ([prior, likelihood]) => tensor(F.bayesRule(toFlat(prior), toFlat(likelihood)).posterior),
  diagnostic: ([spec]) => {
    const [prevalence = 0, sensitivity = 0, specificity = 0] = toFlat(spec)
    const d = F.diagnosticTest({ prevalence, sensitivity, specificity })
    return tensor([d.positivePredictive, d.negativePredictive])
  },
  lognormal: ([y]) => F.transformDensity(normal, (v) => log(v))(y),
  scaledExponential: ([y]) =>
    F.transformDensity(
      (x) => exp(neg(x)),
      (v) => mul(v, 2),
    )(y),
  normalSum: ([s]) => {
    const r = F.sumDensity(normal, normal, [-10, 10], 2001)
    return tensor(toFlat(s).map((v) => at(r, v)))
  },
  exponentialSum: ([s]) => {
    const r = F.sumDensity(
      (x) => exp(neg(x)),
      (x) => exp(neg(x)),
      [0, 40],
      4001,
    )
    return tensor(toFlat(s).map((v) => at(r, v)))
  },
})
