import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import {
  conditionDpp,
  dppLogProbability,
  elementarySymmetric,
  expectedSize,
  inclusionProbability,
  kDppLogProbability,
  marginalKernel,
} from './dpp.ts'

goldenSuite('stochastic/dpp against DPPy and numpy', 'stochastic/dpp', {
  marginal: ([l]) => marginalKernel(l),
  esp: ([eig], p) => tensor(Array.from(elementarySymmetric(toFlat(eig), p['k'] as number))),
  'log-prob': ([l], p) => dppLogProbability(l, p['subset'] as number[]),
  inclusion: ([l], p) => inclusionProbability(l, p['subset'] as number[]),
  'k-log-prob': ([l], p) => kDppLogProbability(l, p['subset'] as number[]),
  size: ([l]) => expectedSize(l),
  'condition-include': ([l], p) => conditionDpp(l, { include: p['include'] as number[] }).kernel,
})
