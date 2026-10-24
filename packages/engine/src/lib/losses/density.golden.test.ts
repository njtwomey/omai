import { goldenSuite } from '../../testing/suite.ts'
import { normal, weibull } from '../dists/index.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('likelihood losses against torch.distributions and scipy.stats', 'losses/density', {
  nllNormal: ([loc, scale, x], p) => L.negativeLogLikelihood(normal(loc, scale), x as Tensor, p),
  heteroscedasticGaussianNll: ([m, s, y], p) => L.heteroscedasticGaussianNll(m, s, y as Tensor, p),
  mixtureDensityNll: ([z, mu, s, y], p) => L.mixtureDensityNll(z, mu, s as Tensor, y as Tensor, p),
  censoredNormal: ([loc, scale, lo, hi], p) => L.censoredNll(normal(loc, scale), lo as Tensor, hi as Tensor, p),
  censoredWeibull: ([k, scale, lo, hi], p) => L.censoredNll(weibull(k, scale), lo as Tensor, hi as Tensor, p),
})
