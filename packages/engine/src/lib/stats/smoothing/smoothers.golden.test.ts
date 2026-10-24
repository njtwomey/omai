import { goldenSuite } from '../../../testing/suite.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import * as M from './index.ts'

goldenSuite('smoothers against statsmodels, pandas and scipy', 'stats/smoothing/smoothers', {
  lowess: ([x, y], p) =>
    M.lowess(x, y, { frac: Number(p['frac']), it: Number(p['it']), delta: Number(p['delta']) }).fitted,
  nadarayaWatson: ([x, y, at], p) => M.nadarayaWatson(x, y, at as Tensor, { bandwidth: Number(p['bandwidth']) }),
  localLinear: ([x, y, at], p) => M.localLinear(x, y, at as Tensor, { bandwidth: Number(p['bandwidth']) }),
  runningMean: ([x], p) => M.runningMean(x, Number(p['window'])),
  runningMedian: ([x], p) => M.runningMedian(x, Number(p['window'])),
  runningMeanTrailing: ([x], p) => M.runningMean(x, Number(p['window']), { center: false }),
  smoothingSpline: ([x, y], p) =>
    M.smoothingSpline(x, y, { lambda: Number(p['lam']) }).spline.evaluate(toFlat(p['at'] as Tensor)) as Tensor,
})
