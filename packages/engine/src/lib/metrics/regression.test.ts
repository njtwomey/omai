import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import * as M from './index.ts'

describe('regression metrics against sklearn', () => {
  for (const c of loadFixture('metrics/regression').cases) {
    it(c.name, () => {
      const metric = (M as unknown as Record<string, M.Metric<M.MetricOptions & Record<string, unknown>>>)[c.fn]
      if (metric === undefined) throw new Error(`no metric ${c.fn}`)
      const [t, p, w] = c.inputs.map((x) => decode(x) as Tensor)
      const result = metric(t as Tensor, p as Tensor, { ...c.params, ...(w === undefined ? {} : { sampleWeight: w }) })
      expectMatches(result.estimate.value, c.output, 1e-12, 1e-14, c.name)
    })
  }
})
