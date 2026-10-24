import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

type Params = Readonly<Record<string, unknown>>
const bytes = (t: Tensor): Uint8Array => Uint8Array.from(toFlat(t))

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => number>> = {
  normalisedCompressionDistance: (x) => I.normalisedCompressionDistance(bytes(x[0] as Tensor), bytes(x[1] as Tensor)),
  lempelZivComplexity: (x, p) => I.lempelZivComplexity(x[0] as Tensor, { normalise: p['normalise'] === true }),
  blockEntropy: (x, p) => I.blockEntropy(x[0] as Tensor, p['length'] as number),
  markovEntropyRate: (x) => I.markovEntropyRate(x[0] as Tensor),
  approximateEntropy: (x, p) => I.approximateEntropy(x[0] as Tensor, p),
  sampleEntropy: (x, p) => I.sampleEntropy(x[0] as Tensor, p),
}

describe('information/complexity against zlib, antropy, scipy and numpy', () => {
  for (const c of loadFixture('information/complexity').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      const inputs = c.inputs.map((x) => decode(x) as Tensor)
      expectMatches(call(inputs, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
