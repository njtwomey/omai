import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

describe('information/reed-solomon against reedsolo', () => {
  for (const c of loadFixture('information/reed_solomon').cases) {
    it(c.name, () => {
      const p = c.params as { nsym: number; m: number; primitive: number; fcr: number }
      const code = I.reedSolomon(p.nsym, { m: p.m, primitive: p.primitive, fcr: p.fcr })
      const inputs = c.inputs.map((x) => toFlat(decode(x) as Tensor))
      const want = toFlat(decode(c.output) as Tensor)
      if (c.fn === 'rsEncode') expect(I.rsEncode(code, inputs[0] ?? [])).toEqual(want)
      else {
        const result = I.rsDecode(code, inputs[0] ?? [], { erasures: inputs[1] ?? [] })
        expect(result.codeword).toEqual(want)
        expect(result.message).toEqual(want.slice(0, want.length - p.nsym))
      }
    })
  }
})
