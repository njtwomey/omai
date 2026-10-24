import { goldenSuite } from '../../testing/suite.ts'
import { detrendedFluctuation, fbmCovariance, fgnAutocovariance, hurstRs } from './fractional.ts'
import { covarianceMatrix } from './gaussian-processes.ts'

goldenSuite('stochastic/fractional against nolds and closed forms', 'stochastic/fractional', {
  'fgn-acov': ([k], p) => fgnAutocovariance(k, p['hurst'] as number),
  'fbm-cov': ([t], p) => covarianceMatrix(fbmCovariance(p['hurst'] as number), t),
  'hurst-rs': ([x], p) => hurstRs(x, { windows: p['windows'] as number[], corrected: p['corrected'] as boolean }),
  dfa: ([x], p) =>
    detrendedFluctuation(x, {
      windows: p['windows'] as number[],
      overlap: p['overlap'] as boolean,
      order: p['order'] as number,
    }),
})
