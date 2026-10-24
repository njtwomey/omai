import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, type Value } from '../../tensor/index.ts'
import { bfmi, summary } from './reports.ts'

type Params = Readonly<Record<string, unknown>>

const CALLS: Readonly<Record<string, (x: Tensor, p: Params) => Value>> = {
  summary: (x, p) => {
    const [r] = summary(x, {
      prob: p['prob'] as number,
      interval: p['interval'] as 'eti' | 'hdi',
      tailProb: p['tailProb'] as number,
    })
    if (r === undefined) throw new Error('no row')
    return tensor([r.mean, r.sd, r.low, r.high, r.essBulk, r.essTail, r.rhat, r.mcseMean, r.mcseSd])
  },
  bfmi: (x) => bfmi(x),
}

describe('sampling/diagnostics/reports against ArviZ', () => {
  for (const c of loadFixture('sampling/diagnostics/reports').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
