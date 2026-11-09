import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { nonparanormalCorrelation, nonparanormalTransform } from './rank.ts'

goldenSuite('the nonparanormal estimators against scipy', 'covariance/rank', {
  spearman: ([x]) => tensor(nonparanormalCorrelation(x, { method: 'spearman', positiveDefinite: false })),
  kendall: ([x]) => tensor(nonparanormalCorrelation(x, { method: 'kendall', positiveDefinite: false })),
  transform: ([x], o) => nonparanormalTransform(x, { winsorise: o['winsorise'] as boolean }),
})
