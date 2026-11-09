import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { clipCorrelation, nearestCorrelation, nearestCovariance } from './nearest.ts'

goldenSuite('the nearest correlation and covariance matrices against statsmodels', 'covariance/nearest', {
  corrNearest: ([a]) => tensor(nearestCorrelation(a, { threshold: 1e-15, tol: 1e-14 }).matrix),
  corrClipped: ([a], o) => tensor(clipCorrelation(a, o['threshold'] as number).matrix),
  covNearest: ([a], o) =>
    tensor(
      nearestCovariance(a, { method: o['method'] as 'clipped' | 'nearest', threshold: o['threshold'] as number })
        .covariance,
    ),
})
