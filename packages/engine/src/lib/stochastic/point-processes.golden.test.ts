import { goldenSuite } from '../../testing/suite.ts'
import { gamma } from '../dists/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { fitHawkes, hawkesLogLikelihood, renewalFunction } from './point-processes.ts'

goldenSuite('stochastic/point-processes against scipy and closed forms', 'stochastic/point-processes', {
  'hawkes-ll': ([events, params], p) => {
    const [mu = 0, alpha = 0, beta = 0] = toFlat(params)
    return hawkesLogLikelihood({ mu, alpha, beta }, events, p['horizon'] as number)
  },
  'hawkes-mle': ([events], p) => {
    const f = fitHawkes(events, p['horizon'] as number)
    return tensor([f.mu, f.alpha, f.beta])
  },
  'hawkes-mle-em': ([events], p) => {
    const f = fitHawkes(events, p['horizon'] as number, { polishBelow: 0 })
    return tensor([f.mu, f.alpha, f.beta])
  },
  'renewal-gamma': ([t], p) => {
    const rate = p['rate'] as number
    const times = toFlat(t)
    const horizon = times.reduce((a, b) => Math.max(a, b), 0)
    const r = renewalFunction(gamma(2, rate), horizon, 2000)
    // read the grid values at the requested times (they are grid points)
    return tensor(times.map((s) => r.values[Math.round((s / horizon) * 2000)] ?? Number.NaN))
  },
})
