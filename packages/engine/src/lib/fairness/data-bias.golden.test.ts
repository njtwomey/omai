import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { baseRates, proxyScores } from './data-bias.ts'

const rates = ([y, s]: readonly (Tensor | undefined)[]) => baseRates(y as Tensor, s as Tensor, { reference: 0 })

goldenSuite('dataset base-rate gaps against AIF360', 'fairness/data_bias', {
  statisticalParityDifference: (inputs) => rates(inputs).statisticalParityDifference,
  disparateImpact: (inputs) => rates(inputs).disparateImpact,
  baseRate1: (inputs) => rates(inputs).rates[1] ?? Number.NaN,
  proxyDependence: ([x, s]) =>
    proxyScores(
      toFlat(x as Tensor).map((v) => [v]),
      s as Tensor,
      { folds: 0 },
    ).dependence[0] ?? Number.NaN,
})
