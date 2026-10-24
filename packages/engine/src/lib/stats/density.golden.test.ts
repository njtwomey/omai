import { goldenSuite, type Params } from '../../testing/suite.ts'
import * as S from './index.ts'

const bw = (p: Params): 'scott' | 'silverman' | number => p['bandwidth'] as 'scott'

goldenSuite('kernel density estimation against scipy.stats and statsmodels', 'stats/density', {
  evaluate: ([x, pts], p) => S.gaussianKde(x, { bandwidth: bw(p) }).evaluate(pts),
  logpdf: ([x, pts], p) => S.gaussianKde(x, { bandwidth: bw(p) }).logpdf(pts),
  evaluateWeighted: ([x, pts, w], p) =>
    S.gaussianKde(x, { bandwidth: bw(p), ...(w === undefined ? {} : { weights: w }) }).evaluate(pts),
  factor: ([x], p) => S.gaussianKde(x, { bandwidth: bw(p) }).factor,
  covariance: ([x]) => S.gaussianKde(x).covariance,
  bandwidthScott: ([x]) => S.bandwidthScott(x),
  bandwidthSilverman: ([x]) => S.bandwidthSilverman(x),
  bandwidthCrossValidation: ([x], p) => S.bandwidthCrossValidation(x, { method: p['method'] as S.BandwidthCriterion }),
  bandwidthCriterion: ([x], p) =>
    S.bandwidthCriterion(x, Number(p['h']), { method: p['method'] as S.BandwidthCriterion }),
  reflection: ([y, grid]) => S.gaussianKde(y, { boundary: { lower: 0 } }).evaluate(grid),
  renormalisation: ([y, grid]) =>
    S.gaussianKde(y, { boundary: { lower: 0, method: 'renormalisation' } }).evaluate(grid),
})
