import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { autocorrelation, autocovariance } from './autocorrelation.ts'

const CALLS: Readonly<Record<string, (x: Tensor) => Value>> = { autocovariance, autocorrelation }

describe('sampling/diagnostics/autocorrelation against ArviZ', () => {
  for (const c of loadFixture('sampling/diagnostics/autocorrelation').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor), c.output, c.rtol, c.atol, c.name)
    })
  }
})
