import { learnGolden, num } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  analyticGaussianScale,
  DEFAULT_RDP_ORDERS,
  exponentialProbabilities,
  gaussianScale,
  heterogeneousComposition,
  rdpToDp,
  subsampledGaussianRdp,
} from './privacy.ts'

learnGolden('differential privacy against diffprivlib and opacus', 'learn/privacy', {
  analytic: (_, p) => ({
    scale: analyticGaussianScale({ sensitivity: 1, epsilon: num(p, 'epsilon'), delta: num(p, 'delta') }),
  }),
  classic: (_, p) => ({
    scale: gaussianScale({ sensitivity: num(p, 'sensitivity'), epsilon: num(p, 'epsilon'), delta: num(p, 'delta') }),
  }),
  exponential: ([u], p) => ({
    probabilities: exponentialProbabilities(toFlat(u as Tensor), {
      epsilon: num(p, 'epsilon'),
      sensitivity: num(p, 'sensitivity'),
      monotonic: p['monotonic'] === true,
    }),
  }),
  composition: ([e, d], p) => {
    const ds = toFlat(d as Tensor)
    const t = heterogeneousComposition(
      toFlat(e as Tensor).map((epsilon, i) => ({ epsilon, delta: ds[i] ?? 0 })),
      num(p, 'slack'),
    )
    return { epsilon: t.epsilon, delta: t.delta }
  },
  rdp: (_, p) => {
    const rdp = subsampledGaussianRdp(num(p, 'q'), num(p, 'sigma'), num(p, 'steps'))
    const r = rdpToDp(DEFAULT_RDP_ORDERS, rdp, 1e-5)
    return { rdp, epsilon: r.epsilon, order: r.order }
  },
})
