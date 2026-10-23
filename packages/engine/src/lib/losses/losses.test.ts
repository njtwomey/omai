import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { checkPrimitive } from '../autodiff/index.ts'
import { type Tensor, tensor, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

type Params = Readonly<Record<string, unknown>>
const opts = (p: Params): Params => p

const FNS: Readonly<Record<string, (xs: Tensor[], p: Params) => Value>> = {
  mse: ([x, y], p) => L.mse(x as Tensor, y as Tensor, opts(p)),
  mae: ([x, y], p) => L.mae(x as Tensor, y as Tensor, opts(p)),
  huber: ([x, y], p) => L.huber(x as Tensor, y as Tensor, opts(p)),
  poissonNll: ([x, y], p) => L.poissonNll(x as Tensor, y as Tensor, opts(p)),
  gaussianNll: ([x, y, v], p) => L.gaussianNll(x as Tensor, y as Tensor, v as Tensor, opts(p)),
  binaryCrossEntropyWithLogits: ([x, y], p) => L.binaryCrossEntropyWithLogits(x as Tensor, y as Tensor, opts(p)),
  binaryCrossEntropy: ([x, y], p) => L.binaryCrossEntropy(x as Tensor, y as Tensor, opts(p)),
  crossEntropy: ([z, y, w], p) =>
    L.crossEntropy(z as Tensor, y as Tensor, { ...opts(p), ...(w === undefined ? {} : { classWeight: w }) }),
  nllLoss: ([lp, y]) => L.nllLoss(lp as Tensor, y as Tensor),
}

describe('losses against torch.nn.functional', () => {
  for (const c of loadFixture('losses/losses').cases) {
    it(c.name, () => {
      const f = FNS[c.fn]
      if (f === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(
        f(
          c.inputs.map((x) => decode(x) as Tensor),
          c.params,
        ),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})

describe('loss gradients', () => {
  const x = tensor([
    [0.3, -1.2, 2],
    [0.5, 0.1, -0.7],
  ])
  const y = tensor([
    [1, 0, 0.4],
    [-0.2, 0.8, 0],
  ])
  for (const [name, f] of [
    ['mse', (a: Value) => L.mse(a, y)],
    ['huber', (a: Value) => L.huber(a, y, { delta: 0.5 })],
    ['logCosh', (a: Value) => L.logCosh(a, y)],
    ['pinball', (a: Value) => L.pinball(a, y, { tau: 0.3 })],
    [
      'bce with logits',
      (a: Value) =>
        L.binaryCrossEntropyWithLogits(
          a,
          tensor([
            [1, 0, 1],
            [0, 0, 1],
          ]),
        ),
    ],
    ['cross-entropy', (a: Value) => L.crossEntropy(a, tensor([2, 0]), { labelSmoothing: 0.1 })],
  ] as const) {
    it(`${name} passes the conformance check`, () => {
      expect(checkPrimitive(f, [x], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    })
  }
})
