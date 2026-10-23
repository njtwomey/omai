import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as M from './index.ts'

goldenSuite('distances against scipy.spatial', 'metrics/distances', {
  minkowskiDistance: ([u, v, w], o) =>
    M.minkowskiDistance(u, v, {
      ...(o['p'] === undefined ? {} : { p: o['p'] === 'inf' ? Number.POSITIVE_INFINITY : (o['p'] as number) }),
      ...(w === undefined ? {} : { w }),
    }).estimate.value,
  cosineSimilarity: ([u, v]) => M.cosineSimilarity(u, v).estimate.value,
  mahalanobisDistance: ([u, v, vi]) => M.mahalanobisDistance(u, v, vi as Tensor).estimate.value,
  procrustesDisparity: ([a, b]) => M.procrustesDisparity(a, b).estimate.value,
  hausdorffDistance: ([a, b], o) => M.hausdorffDistance(a, b, o).estimate.value,
  chamferDistance: ([a, b], o) => M.chamferDistance(a, b, o).estimate.value,
})
