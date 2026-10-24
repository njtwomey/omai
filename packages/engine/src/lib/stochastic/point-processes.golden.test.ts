import { goldenSuite } from '../../testing/suite.ts'
import { gamma } from '../dists/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { fitHawkes, hawkesLogLikelihood, type RipleyCorrection, renewalFunction, ripleysK } from './point-processes.ts'

goldenSuite('stochastic/point-processes against astropy, scipy and closed forms', 'stochastic/point-processes', {
  ripley: ([pts, radii], p) =>
    tensor(
      Array.from(
        ripleysK(pts, toFlat(radii), {
          correction: p['mode'] as RipleyCorrection,
          window: {
            xmin: p['xmin'] as number,
            xmax: p['xmax'] as number,
            ymin: p['ymin'] as number,
            ymax: p['ymax'] as number,
          },
        }),
      ),
    ),
  'hawkes-ll': ([events, params], p) => {
    const [mu = 0, alpha = 0, beta = 0] = toFlat(params)
    return hawkesLogLikelihood({ mu, alpha, beta }, events, p['horizon'] as number)
  },
  'hawkes-mle': ([events], p) => {
    const f = fitHawkes(events, p['horizon'] as number)
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
