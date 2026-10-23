import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { type Tensor, tensor, type Value } from '../../tensor/index.ts'
import * as A from './activations.ts'

type Params = Readonly<Record<string, unknown>>

const TABLE: Readonly<Record<string, (x: Tensor, p: Params, w?: Tensor) => Value>> = {
  relu6: (x) => A.relu6(x),
  hardswish: (x) => A.hardswish(x),
  celu: (x, p) => A.celu(x, p['alpha'] as number),
  softsign: (x) => A.softsign(x),
  logsigmoid: (x) => A.logsigmoid(x),
  tanhshrink: (x) => A.tanhshrink(x),
  softshrink: (x, p) => A.softshrink(x, p['lambd'] as number),
  hardshrink: (x, p) => A.hardshrink(x, p['lambd'] as number),
  threshold: (x, p) => A.threshold(x, p['threshold'] as number, p['value'] as number),
  prelu: (x, _p, w) => A.prelu(x, w as Tensor),
  swiglu: (x) => A.swiglu(x),
  geglu: (x) => A.geglu(x),
  reglu: (x) => A.reglu(x),
}

describe('activations against torch', () => {
  for (const c of loadFixture('nn/functional/activations').cases) {
    it(c.name, () => {
      const [x, w] = c.inputs.map((v) => decode(v) as Tensor)
      const f = TABLE[c.fn] as (x: Tensor, p: Params, w?: Tensor) => Value
      expectMatches(f(x as Tensor, c.params, w), c.output, 1e-12, 1e-14, c.name)
    })
  }
})

describe('activation gradients', () => {
  it('pass the conformance check away from kinks', () => {
    const x = tensor([-2.3, -0.7, 0.4, 1.9])
    for (const f of [A.celu, A.softsign, A.logsigmoid, A.tanhshrink, A.hardswish]) {
      expect(checkPrimitive((v: Value) => f(v), [x], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    }
    const g = tensor([
      [0.3, -1.2, 0.8, 2.1],
      [-0.4, 0.9, -1.5, 0.6],
    ])
    for (const f of [A.swiglu, A.geglu, A.reglu, A.glu]) {
      expect(checkPrimitive((v: Value) => f(v), [g], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    }
    expect(
      checkPrimitive(
        (v: Value, w: Value) => A.prelu(v, w),
        [
          tensor([
            [
              [-1, 2],
              [3, -4],
            ],
          ]),
          tensor([0.2, 0.3]),
        ],
        {
          rtol: 1e-5,
        },
      ).failures,
    ).toEqual([])
  })
  it('activation names cover every listed function', () => {
    for (const name of ['relu6', 'celu', 'geluTanh', 'softsign', 'logsigmoid', 'hardswish', 'tanhshrink'] as const) {
      expect(A.activation(name)([0.5]).shape).toEqual([1])
    }
  })
})
