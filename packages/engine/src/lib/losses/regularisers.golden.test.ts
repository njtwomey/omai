import { goldenSuite } from '../../testing/suite.ts'
import { toFlat } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('regularisation penalties against numpy and scipy', 'losses/regularisers', {
  l1Penalty: ([w], p) => L.l1Penalty(w, p),
  l2Penalty: ([w], p) => L.l2Penalty(w, p),
  elasticNet: ([w], p) => L.elasticNet(w, p),
  groupLasso: ([w, g], p) => L.groupLasso(w, toFlat(g), p),
  nuclearNorm: ([m], p) => L.nuclearNorm(m, p),
  orthogonality: ([m], p) => L.orthogonality(m, p),
  entropyPenalty: ([x], p) => L.entropyPenalty(x, p),
})
