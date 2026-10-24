import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

describe('information/convolutional against scikit-commpy', () => {
  for (const c of loadFixture('information/convolutional').cases) {
    it(c.name, () => {
      const code = I.convolutionalCode(c.params['generators'] as number[])
      const message = toFlat(decode(c.inputs[0]) as Tensor)
      const coded = toFlat(decode(c.output) as Tensor)
      expect(I.convolutionalEncode(code, message)).toEqual(coded)
      expect(I.viterbiDecode(code, coded).bits).toEqual(message)
    })
  }
})
