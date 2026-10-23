import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const a = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.15, -0.7],
  [1.1, 0.4, 0.2],
  [-0.6, 0.9, 0.3],
])
const b = tensor([
  [0.2, -1, 1.5],
  [0.9, 0.1, -0.2],
  [1, -0.3, 0.6],
  [-0.2, 1.4, -0.5],
])
const labels = [0, 1, 0, 1]
const centres = tensor([
  [1, 0.2, -0.3],
  [0.1, -1, 0.5],
])

describe('contrastive losses', () => {
  for (const [name, f, inputs] of [
    ['InfoNCE', (x: Value, y: Value) => L.infoNce(x, y, { temperature: 0.5 }), [a, b]],
    ['InfoNCE with negatives', (x: Value, y: Value) => L.infoNce(x, y, { negatives: centres }), [a, b]],
    ['NT-Xent', (x: Value) => L.ntXent(x, labels, { temperature: 0.5 }), [a]],
    [
      'triplet',
      (x: Value, y: Value) =>
        L.tripletMargin(
          x,
          y,
          tensor([
            [0, 0, 0],
            [1, 1, 1],
            [2, 0, 1],
            [0, 2, 0],
          ]),
          { margin: 3 },
        ),
      [a, b],
    ],
    ['contrastive', (x: Value, y: Value) => L.contrastive(x, y, [1, 0, 1, 0], { margin: 5 }), [a, b]],
    ['cosine embedding', (x: Value, y: Value) => L.cosineEmbedding(x, y, [1, -1, 1, -1], { margin: -1 }), [a, b]],
    ['supervised contrastive', (x: Value) => L.supervisedContrastive(x, labels, { temperature: 0.5 }), [a]],
    ['CLIP', (x: Value, y: Value) => L.clipLoss(x, y, { logitScale: 3 }), [a, b]],
    ['ArcFace', (x: Value, w: Value) => L.arcFace(x, w, [0, 1, 0, 1], { scale: 4, margin: 0.3 }), [a, centres]],
    ['CosFace', (x: Value, w: Value) => L.cosFace(x, w, [0, 1, 0, 1], { scale: 4 }), [a, centres]],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f as (...xs: Value[]) => Value, inputs))
  }

  it('reduces ArcFace and CosFace with no margin to the scaled cosine softmax', () => {
    const plain = L.crossEntropy(L.cosFaceLogits(a, centres, [0, 1, 0, 1], { scale: 3, margin: 0 }), [0, 1, 0, 1])
    expect(item(L.arcFace(a, centres, [0, 1, 0, 1], { scale: 3, margin: 0 }))).toBeCloseTo(item(plain), 13)
  })

  it('gives NT-Xent with two views the SimCLR value', () => {
    // Two pairs of identical views: each anchor's positive has similarity 1/T, its two negatives 0.
    const views = [
      [1, 0],
      [0, 1],
      [1, 0],
      [0, 1],
    ]
    expect(item(L.ntXent(views, [0, 1, 0, 1], { temperature: 0.5 }))).toBeCloseTo(Math.log(1 + 2 * Math.exp(-2)), 14)
  })
})
