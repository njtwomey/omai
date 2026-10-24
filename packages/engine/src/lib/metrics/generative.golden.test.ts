import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

goldenSuite('generative metrics from features against pytorch-fid, torchmetrics and prdc', 'metrics/generative', {
  inceptionScore: ([p], o) => M.inceptionScore(p, o).estimate.value,
  inceptionScoreStd: ([p], o) => M.inceptionScore(p, o).breakdown?.['std'] ?? Number.NaN,
  frechetDistance: ([a, b]) => M.frechetDistance(a, b).estimate.value,
  kernelInceptionDistance: ([a, b], o) => M.kernelInceptionDistance(a, b, o).estimate.value,
  clipScore: ([a, b]) => M.clipScore(a, b).estimate.value,
  vendiScore: ([x], o) => M.vendiScore(x, o).estimate.value,
  prdc: ([a, b], o) =>
    M.precisionRecallDensityCoverage(a, b, { nearestK: o['nearestK'] as number })[o['field'] as keyof M.Prdc],
})
