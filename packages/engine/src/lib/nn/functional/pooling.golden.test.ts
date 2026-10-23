import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { adaptiveAvgPool, adaptiveMaxPool, avgPool, lpPool, maxPool, maxPoolWithIndices, maxUnpool } from './pooling.ts'

type Size = number | number[]

const kernelOf = (p: Readonly<Record<string, unknown>>): Size => {
  const k = p['kernelSize'] as Size
  const d = p['dims'] as number | undefined
  return typeof k === 'number' && d !== undefined ? new Array<number>(d).fill(k) : k
}

describe('pooling against torch', () => {
  for (const c of loadFixture('nn/functional/pooling').cases) {
    it(c.name, () => {
      const [x, extra] = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const k = kernelOf(p)
      const maxOptions = {
        stride: p['stride'] as Size,
        padding: p['padding'] as Size,
        dilation: p['dilation'] as Size,
        ceilMode: p['ceilMode'] as boolean,
      }
      let y: Value
      switch (c.fn) {
        case 'maxPool':
          y = maxPool(x as Tensor, k, maxOptions)
          break
        case 'maxPoolIndices': {
          const got = maxPoolWithIndices(x as Tensor, k, maxOptions).indices
          const want = decode(c.output) as Tensor
          expect(got.shape).toEqual(want.shape)
          expect(toFlat(got)).toEqual(toFlat(want))
          return
        }
        case 'maxUnpool':
          y = maxUnpool(x as Tensor, extra as Tensor, k)
          break
        case 'avgPool':
          y = avgPool(x as Tensor, k, {
            stride: p['stride'] as Size,
            padding: p['padding'] as Size,
            ceilMode: p['ceilMode'] as boolean,
            countIncludePad: p['countIncludePad'] as boolean,
            ...(p['divisorOverride'] === null ? {} : { divisorOverride: p['divisorOverride'] as number }),
          })
          break
        case 'adaptiveAvgPool':
          y = adaptiveAvgPool(x as Tensor, p['outputSize'] as number[])
          break
        case 'adaptiveMaxPool':
          y = adaptiveMaxPool(x as Tensor, p['outputSize'] as number[])
          break
        default:
          y = lpPool(
            x as Tensor,
            p['normType'] as number,
            k,
            p['stride'] === undefined ? {} : { stride: p['stride'] as Size },
          )
      }
      expectMatches(y, c.output, 1e-12, 1e-13, c.name)
    })
  }
})
