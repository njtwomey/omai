import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor } from '../../tensor/index.ts'
import { type EssMethod, ess, type McseMethod, mcse, type RhatMethod, rhat } from './convergence.ts'

type Params = Readonly<Record<string, unknown>>

const CALLS: Readonly<Record<string, (x: Tensor, p: Params) => number>> = {
  rhat: (x, p) => rhat(x, { method: p['method'] as RhatMethod }),
  ess: (x, p) =>
    ess(x, {
      ...(p['method'] === undefined ? {} : { method: p['method'] as EssMethod }),
      ...(p['prob'] === undefined ? {} : { prob: p['prob'] as number | [number, number] }),
      ...(p['relative'] === undefined ? {} : { relative: p['relative'] as boolean }),
    }),
  mcse: (x, p) =>
    mcse(x, {
      method: p['method'] as McseMethod,
      ...(p['prob'] === undefined ? {} : { prob: p['prob'] as number }),
    }),
}

describe('sampling/diagnostics/convergence against ArviZ', () => {
  for (const c of loadFixture('sampling/diagnostics/convergence').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
