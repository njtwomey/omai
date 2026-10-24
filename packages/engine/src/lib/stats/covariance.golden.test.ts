import { goldenSuite } from '../../testing/suite.ts'
import * as S from './index.ts'

goldenSuite('covariance estimators against sklearn.covariance', 'stats/covariance', {
  empirical: ([x], p) => S.empiricalCovariance(x, { assumeCentered: p['assumeCentered'] as boolean }),
  ledoitWolf: ([x], p) => S.ledoitWolf(x, { assumeCentered: p['assumeCentered'] as boolean }).covariance,
  ledoitWolfShrinkage: ([x], p) => S.ledoitWolf(x, { assumeCentered: p['assumeCentered'] as boolean }).shrinkage,
  oas: ([x], p) => S.oas(x, { assumeCentered: p['assumeCentered'] as boolean }).covariance,
  oasShrinkage: ([x], p) => S.oas(x, { assumeCentered: p['assumeCentered'] as boolean }).shrinkage,
  shrunk: ([x], p) =>
    S.shrunkCovariance(x, { assumeCentered: p['assumeCentered'] as boolean, shrinkage: 0.3 }).covariance,
})
