import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import { paretoKHat, paretoSmoothedWeights } from './importance.ts'

type Params = Readonly<Record<string, unknown>>

const CALLS: Readonly<Record<string, (x: Tensor, p: Params) => Value | number>> = {
  psisWeights: (x, p) => paretoSmoothedWeights(x, { rEff: p['rEff'] as number }).logWeights,
  psisK: (x, p) => paretoSmoothedWeights(x, { rEff: p['rEff'] as number }).paretoK,
  paretoKHat: (x, p) =>
    paretoKHat(x, {
      ...(p['tail'] === undefined ? {} : { tail: p['tail'] as 'both' | 'right' | 'left' }),
      ...(p['rEff'] === undefined ? {} : { rEff: p['rEff'] as number }),
    }),
}

describe('sampling/importance against ArviZ', () => {
  for (const c of loadFixture('sampling/importance').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
