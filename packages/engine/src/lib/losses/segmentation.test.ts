import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const logits = tensor([
  [
    [0.3, -1.2, 2],
    [0.5, 0.15, -0.7],
  ],
  [
    [1.1, 0.4, 0.2],
    [-0.6, 0.9, 0.3],
  ],
])
const labels = tensor([
  [2, 0],
  [0, 1],
])

describe('segmentation losses', () => {
  for (const [name, f] of [
    ['Dice', (x: Value) => L.dice(x, labels, { activation: 'softmax' })],
    ['squared Dice', (x: Value) => L.dice(x, labels, { activation: 'sigmoid', squared: true })],
    ['Tversky', (x: Value) => L.tversky(x, labels, { activation: 'softmax', alpha: 0.2, beta: 0.8 })],
    ['focal Tversky', (x: Value) => L.focalTversky(x, labels, { activation: 'softmax' })],
    ['Lovasz-softmax', (x: Value) => L.lovaszSoftmax(x, labels)],
    [
      'Lovasz hinge',
      (x: Value) =>
        L.lovaszHinge(
          x,
          tensor([
            [
              [1, 0, 1],
              [0, 0, 1],
            ],
            [
              [1, 1, 0],
              [0, 1, 0],
            ],
          ]),
        ),
    ],
  ] as const) {
    it(`${name} passes the conformance check`, () => expectConformant(f, [logits]))
  }

  it('reduces Tversky with alpha = beta = 0.5 to Dice', () => {
    expect(item(L.tversky(logits, labels, { activation: 'softmax', smooth: 0 }))).toBeCloseTo(
      item(L.dice(logits, labels, { activation: 'softmax', smooth: 0 })),
      14,
    )
  })

  it('drops ignored pixels', () => {
    const ignored = L.dice(
      logits,
      [
        [2, 9],
        [0, 9],
      ],
      { activation: 'softmax', ignoreIndex: 9, batch: true },
    )
    const kept = L.dice(tensor([[[0.3, -1.2, 2]], [[1.1, 0.4, 0.2]]]), [[2], [0]], {
      activation: 'softmax',
      batch: true,
    })
    expect(item(ignored)).toBeCloseTo(item(kept), 14)
    const hinge = L.lovaszHinge([[2, -1, 0.5]], [[1, 7, 0]], { ignoreIndex: 7 })
    expect(item(hinge)).toBeCloseTo(item(L.lovaszHinge([[2, 0.5]], [[1, 0]])), 14)
    const soft = L.lovaszSoftmax(
      [
        [
          [0.2, 0.8],
          [0.6, 0.4],
          [0.1, 0.9],
        ],
      ],
      [[1, 5, 0]],
      { ignoreIndex: 5 },
    )
    expect(item(soft)).toBeCloseTo(
      item(
        L.lovaszSoftmax(
          [
            [
              [0.2, 0.8],
              [0.1, 0.9],
            ],
          ],
          [[1, 0]],
        ),
      ),
      14,
    )
  })
})
