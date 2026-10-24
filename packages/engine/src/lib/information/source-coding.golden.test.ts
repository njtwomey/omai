import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import { expectedLength, huffman } from './index.ts'

describe('information/source-coding against bitarray', () => {
  for (const c of loadFixture('information/source_coding').cases) {
    it(c.name, () => {
      const freq = decode(c.inputs[0]) as Tensor
      expectMatches(expectedLength(huffman(freq), freq), c.output, c.rtol, c.atol, c.name)
    })
  }
})
