import { learnGolden, num } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { differenceInDifferences, eventStudy, syntheticControl, twoWayFixedEffects } from './index.ts'

const panelOf = (inputs: readonly unknown[]) => {
  const [u, t, d, y] = inputs as [Tensor, Tensor, Tensor, Tensor]
  return { unit: toFlat(u), period: toFlat(t), treated: toFlat(d), outcome: toFlat(y) }
}

learnGolden('Panel designs against statsmodels dummy-variable regressions and scipy', 'causal/designs', {
  did: (inputs) => {
    const r = differenceInDifferences(panelOf(inputs))
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
  twfe: (inputs) => {
    const r = twoWayFixedEffects(panelOf(inputs))
    return { value: r.estimate.value, se: r.estimate.standardError ?? Number.NaN }
  },
  eventStudy: (inputs, p) => {
    const r = eventStudy(panelOf(inputs), { leads: num(p, 'leads'), lags: num(p, 'lags') })
    return { coefficients: r.coefficients as number[], standardErrors: r.standardErrors as number[] }
  },
  syntheticControl: (inputs, p) => {
    const [treated, pool] = inputs as [Tensor, Tensor]
    const r = syntheticControl({
      treated: toFlat(treated),
      donors: toFlat(pool),
      donorCount: pool.shape[1] ?? 0,
      start: num(p, 'start'),
    })
    let loss = 0
    for (let s = 0; s < num(p, 'start'); s++) loss += (r.gap[s] ?? 0) ** 2
    return { weights: Array.from(r.weights), gap: Array.from(r.gap), loss }
  },
})
