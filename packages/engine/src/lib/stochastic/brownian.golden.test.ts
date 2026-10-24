import { goldenSuite } from '../../testing/suite.ts'
import {
  bridgeMoments,
  firstPassageCdf,
  firstPassageDensity,
  firstPassageDistribution,
  maximumCdf,
} from './brownian.ts'

goldenSuite('stochastic/brownian against scipy.stats', 'stochastic/brownian', {
  'fpt-pdf': ([t], p) =>
    firstPassageDensity(t, p['level'] as number, { drift: p['drift'] as number, sigma: p['sigma'] as number }),
  'fpt-cdf': ([t], p) =>
    firstPassageCdf(t, p['level'] as number, { drift: p['drift'] as number, sigma: p['sigma'] as number }),
  'max-cdf': ([t], p) =>
    maximumCdf(p['level'] as number, t, { drift: p['drift'] as number, sigma: p['sigma'] as number }),
  'max-cdf-m': ([m], p) =>
    maximumCdf(m, p['t'] as number, { drift: p['drift'] as number, sigma: p['sigma'] as number }),
  'fpt-dist-logpdf': ([t], p) =>
    firstPassageDistribution(p['level'] as number, {
      drift: p['drift'] as number,
      sigma: p['sigma'] as number,
    }).logProb(t),
  'bridge-mean': ([s], p) =>
    bridgeMoments(s, {
      t0: p['t0'] as number,
      t1: p['t1'] as number,
      x0: p['x0'] as number,
      x1: p['x1'] as number,
      sigma: p['sigma'] as number,
    }).mean,
  'bridge-var': ([s], p) =>
    bridgeMoments(s, {
      t0: p['t0'] as number,
      t1: p['t1'] as number,
      x0: p['x0'] as number,
      x1: p['x1'] as number,
      sigma: p['sigma'] as number,
    }).variance,
})
