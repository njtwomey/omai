import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const x = tensor([
  [0.3, -1.2, 2],
  [0.5, 0.1, -0.7],
])
const bits = tensor([
  [1, 0, 1],
  [0, 0, 1],
])

describe('classification losses', () => {
  for (const [name, f] of [
    [
      'bce with logits',
      (a: Value) => L.binaryCrossEntropyWithLogits(a, bits, { labelSmoothing: 0.1, posWeight: [1, 2, 3] }),
    ],
    ['bce', (a: Value) => L.binaryCrossEntropy(a, bits)],
    ['focal', (a: Value) => L.focal(a, bits)],
    ['categorical focal', (a: Value) => L.categoricalFocal(a, tensor([2, 0]), { gamma: 2, alpha: [1, 2, 0.5] })],
    ['multi-label soft margin', (a: Value) => L.multiLabelSoftMargin(a, bits)],
    ['cross-entropy', (a: Value) => L.crossEntropy(a, tensor([2, 0]), { labelSmoothing: 0.1, classWeight: [1, 2, 3] })],
    ['nll', (a: Value) => L.nllLoss(a, tensor([2, 0]), { ignoreIndex: 0 })],
  ] as const) {
    const at =
      name === 'bce'
        ? tensor([
            [0.2, 0.5, 0.9],
            [0.4, 0.1, 0.7],
          ])
        : x
    it(`${name} passes the conformance check`, () => expectConformant(f, [at]))
  }

  it('reduces focal loss with gamma 0 and no alpha to the binary cross-entropy', () => {
    expect(item(L.focal(x, bits, { gamma: 0, alpha: -1 }))).toBeCloseTo(
      item(L.binaryCrossEntropyWithLogits(x, bits)),
      14,
    )
    expect(item(L.categoricalFocal(x, [2, 0], { gamma: 0 }))).toBeCloseTo(item(L.crossEntropy(x, [2, 0])), 14)
  })

  it('ignores targets in the categorical focal mean', () => {
    const one = item(L.categoricalFocal([[0.3, -1.2, 2]], [2]))
    expect(item(L.categoricalFocal(x, [2, 1], { ignoreIndex: 1 }))).toBeCloseTo(one, 14)
  })
})
