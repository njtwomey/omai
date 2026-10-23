import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const zp = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.15, -0.7],
])
const zq = tensor([
  [1.1, 0.4, 0.2],
  [-0.6, 0.9, 0.3],
])
const x = tensor([
  [0.2, -0.5],
  [1, 0.3],
  [-0.4, 0.8],
])
const y = tensor([
  [0.9, 0.1],
  [-1, 0.2],
])

describe('divergence losses', () => {
  for (const [name, f, inputs] of [
    ['KL from logits', (a: Value, b: Value) => L.klDivergence(a, b, { logits: true }), [zp, zq]],
    ['Jensen-Shannon from logits', (a: Value, b: Value) => L.jensenShannon(a, b, { logits: true }), [zp, zq]],
    ['distillation', (s: Value) => L.distillation(s, zq, { temperature: 2, target: [0, 2] }), [zp]],
    ['Gaussian MMD', (a: Value, b: Value) => L.mmd(a, b, { bandwidths: [0.5, 1] }), [x, y]],
    [
      'unbiased Laplacian MMD',
      (a: Value, b: Value) => L.mmd(a, b, { kernel: 'laplacian', bandwidths: [1], unbiased: true }),
      [x, y],
    ],
    ['energy distance', (a: Value, b: Value) => L.mmd(a, b, { kernel: 'energy' }), [x, y]],
    ['Sinkhorn', (a: Value, b: Value) => L.sinkhorn(a, b, { epsilon: 0.5, maxIter: 50 }), [x, y]],
  ] as const) {
    it(`${name} passes the conformance check`, () =>
      expectConformant(f as (...xs: Value[]) => Value, inputs, { rtol: 1e-4 }))
  }

  it('gives the KL losses their identities', () => {
    // klDiv with batchmean is the per-sample KL of klDivergence.
    const p = tensor([
      [0.2, 0.5, 0.3],
      [0.6, 0.3, 0.1],
    ])
    const q = tensor([
      [0.1, 0.1, 0.8],
      [0.3, 0.3, 0.4],
    ])
    const viaTorch = L.klDiv(
      tensor([
        [Math.log(0.1), Math.log(0.1), Math.log(0.8)],
        [Math.log(0.3), Math.log(0.3), Math.log(0.4)],
      ]),
      p,
      { reduction: 'batchmean' },
    )
    expect(item(viaTorch)).toBeCloseTo(item(L.klDivergence(p, q)), 14)
    expect(item(L.jensenShannon(p, p))).toBe(0)
    expect(item(L.jensenShannon(p, q))).toBeCloseTo(item(L.jensenShannon(q, p)), 15)
  })

  it('makes the debiased Sinkhorn divergence vanish between equal clouds', () => {
    expect(Math.abs(item(L.sinkhorn(x, x, { epsilon: 0.5, debiased: true })))).toBeLessThan(1e-9)
  })
})
