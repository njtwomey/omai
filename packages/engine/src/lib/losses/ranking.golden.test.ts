import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('ranking losses against torch, optax and their formulas', 'losses/ranking', {
  marginRanking: ([a, b, y], p) => L.marginRanking(a, b, y as Tensor, p),
  bpr: ([pos, neg], p) => L.bpr(pos, neg, p),
  rankNet: ([s, r], p) => L.rankNet(s, r, p),
  pairwiseHinge: ([s, r], p) => L.pairwiseHinge(s, r, p),
  lambdaRank: ([s, r], p) => L.lambdaRank(s, r, p),
  listNet: ([s, r], p) => L.listNet(s, r, p),
  softmaxRanking: ([s, r], p) => L.softmaxRanking(s, r, p),
  listMle: ([s, r], p) => L.listMle(s, r, p),
  approxNdcg: ([s, r], p) => L.approxNdcg(s, r, p),
})
