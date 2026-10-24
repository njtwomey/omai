import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, type Value } from '../../tensor/index.ts'
import { eti, hdi, hdiModes } from './intervals.ts'

type Params = Readonly<Record<string, unknown>>

const CALLS: Readonly<Record<string, (x: Tensor, p: Params) => Value>> = {
  hdi: (x, p) => tensor(hdi(x, p['prob'] as number, { circular: p['circular'] === true })),
  eti: (x, p) => tensor(eti(x, p['prob'] as number)),
  hdiModes: (x, p) =>
    tensor(hdiModes(x, p['prob'] as number, { maxModes: p['maxModes'] as number }).map((m) => [...m.interval])),
}

describe('sampling/diagnostics/intervals against ArviZ', () => {
  for (const c of loadFixture('sampling/diagnostics/intervals').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
