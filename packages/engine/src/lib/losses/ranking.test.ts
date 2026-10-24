import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const s = tensor([
  [0.3, -1.2, 2, 0.7],
  [0.5, 0.15, -0.7, 1.9],
])
const r = tensor([
  [2, 0, 1, 1],
  [0, 3, 1, 0],
])

describe('ranking losses', () => {
  for (const [name, f] of [
    ['RankNet', (a: Value) => L.rankNet(a, r)],
    ['pairwise hinge', (a: Value) => L.pairwiseHinge(a, r, { margin: 5 })],
    ['LambdaRank', (a: Value) => L.lambdaRank(a, r, { k: 3 })],
    ['ListNet', (a: Value) => L.listNet(a, r)],
    ['softmax ranking', (a: Value) => L.softmaxRanking(a, r)],
    ['ListMLE', (a: Value) => L.listMle(a, r)],
    ['ApproxNDCG', (a: Value) => L.approxNdcg(a, r, { alpha: 2 })],
    ['BPR', (a: Value) => L.bpr(a, tensor([[0.1], [-0.4]]))],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f, [s]))
  }

  it('ignores masked items as if the lists were shorter', () => {
    const mask = [[true, true, true, false]]
    const short = { scores: [[0.3, -1.2, 2]], rel: [[2, 0, 1]] }
    const padded = { scores: [[0.3, -1.2, 2, 9]], rel: [[2, 0, 1, 5]] }
    for (const f of [L.rankNet, L.pairwiseHinge, L.lambdaRank, L.listNet, L.softmaxRanking, L.listMle, L.approxNdcg]) {
      expect(item(f(padded.scores, padded.rel, { mask }))).toBeCloseTo(item(f(short.scores, short.rel)), 13)
    }
  })

  it('makes ApproxNDCG approach minus the true NDCG for sharp ranks', () => {
    // Scores in the ideal order give NDCG 1.
    expect(item(L.approxNdcg([[3, 2, 1, 0]], [[3, 2, 1, 0]], { alpha: 1000 }))).toBeCloseTo(-1, 12)
  })

  it('treats a cutoff at the list length as no cutoff', () => {
    expect(item(L.lambdaRank(s, r, { k: 4 }))).toBe(item(L.lambdaRank(s, r)))
  })
})
