import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { aipw, entropyBalancing, ipw, matching, regressionAdjustment, stratification, tmle } from './index.ts'

const data = (inputs: readonly unknown[]) => {
  const [x, t, y] = inputs as [Tensor, Tensor, Tensor]
  return { x, treatment: t, outcome: y }
}

learnGolden('Causal estimators against statsmodels, scikit-learn and scipy', 'causal/estimation', {
  regressionAdjustment: (inputs) => {
    const r = regressionAdjustment(data(inputs), { mode: 'linear' })
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
  ipw: (inputs, p) => {
    const r = ipw(data(inputs), { estimand: str(p, 'estimand') as 'ate' | 'att' })
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN, weights: Array.from(r.weights) }
  },
  aipw: (inputs, p) => {
    const r = aipw(data(inputs), { estimand: str(p, 'estimand') as 'ate' | 'att' })
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
  stratification: (inputs, p) => {
    const r = stratification(data(inputs), { strata: num(p, 'strata') })
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
  matching: (inputs, p) => {
    const r = matching(data(inputs), {
      on: str(p, 'on') as 'propensity' | 'covariates',
      neighbours: num(p, 'neighbours'),
    })
    return { value: r.estimate.value }
  },
  entropyBalancing: (inputs) => {
    const d = data(inputs)
    const r = entropyBalancing(d)
    const t = toFlat(d.treatment)
    return { value: r.estimate.value, controlWeights: Array.from(r.weights).filter((_, i) => t[i] === 0) }
  },
  tmle: (inputs) => {
    const r = tmle(data(inputs))
    return { epsilon: r.epsilon, value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
})
