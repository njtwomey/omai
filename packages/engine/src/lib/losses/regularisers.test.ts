import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const w = tensor([0.3, -1.2, 2, 0.7, -0.4])
const m = tensor([
  [0.5, -1, 0.3],
  [0.2, 0.8, -0.4],
  [1.1, 0.1, 0.6],
  [-0.3, 0.5, 0.9],
])

describe('regularisers', () => {
  for (const [name, f, at] of [
    ['L1', (v: Value) => L.l1Penalty(v), w],
    ['L2', (v: Value) => L.l2Penalty(v, { strength: 0.1 }), w],
    ['elastic net', (v: Value) => L.elasticNet(v, { l1Ratio: 0.3 }), w],
    ['group lasso', (v: Value) => L.groupLasso(v, [0, 0, 1, 1, 2], { l1Ratio: 0.2 }), w],
    ['nuclear norm', (v: Value) => L.nuclearNorm(v), m],
    ['orthogonality', (v: Value) => L.orthogonality(v, { double: true }), m],
    ['entropy', (v: Value) => L.entropyPenalty(v), m],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f, [at], { rtol: 1e-4 }))
  }

  it('sums penalties over every leaf of a pytree', () => {
    const tree = { a: [1, -2], b: { c: tensor([0.5]) } }
    expect(item(L.l1Penalty(tree))).toBe(3.5)
    expect(item(L.l2Penalty([tensor([3]), tensor([4])]))).toBe(12.5)
  })

  it('gives zero groups a finite (zero) gradient', () => {
    expectConformant((v: Value) => L.groupLasso(v, [0, 0, 1]), [tensor([0.5, -1, 0])], { wrt: [0] })
    expect(() => L.groupLasso([1, 2], [0])).toThrow()
  })
})
