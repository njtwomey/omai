import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { type RipleyCorrection, ripleysK } from './spatial-processes.ts'

goldenSuite('stochastic/spatial-processes against astropy', 'stochastic/spatial-processes', {
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
})
