import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { key, normal } from '../random/index.ts'
import { add, item, tensor, toArray, type Value } from '../tensor/index.ts'
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

describe('sliced Wasserstein', () => {
  const x = normal(key(1), [200, 2])

  it('is exactly the squared shift in one dimension (the directions are plus or minus one)', () => {
    const d = normal(key(2), [50, 1])
    expect(item(L.slicedWasserstein(d, add(d, 0.7), key(3), { projections: 5 }))).toBeCloseTo(0.49, 13)
    expect(item(L.slicedWasserstein(d, add(d, 0.7), key(3), { projections: 5, root: true }))).toBeCloseTo(0.7, 13)
  })

  it('averages the squared projected shift over directions in two dimensions (shift squared over the dimension)', () => {
    const shifted = add(x, tensor([0.8, 0]))
    expect(item(L.slicedWasserstein(x, shifted, key(4), { projections: 3000 }))).toBeCloseTo(0.64 / 2, 1)
  })

  it('is zero between a sample and a permutation of itself and passes the conformance check', () => {
    const rows = toArray(x) as number[][]
    expect(item(L.slicedWasserstein(x, [...rows].reverse(), key(5)))).toBe(0)
    expectConformant(
      (a, b) => L.slicedWasserstein(a, b, key(6), { projections: 4 }),
      [
        tensor([
          [0.3, 1.1],
          [-0.2, 0.5],
          [1.4, -0.6],
        ]),
        tensor([
          [0.9, 0.2],
          [0.1, -0.3],
          [-0.5, 0.8],
        ]),
      ],
    )
  })
})
