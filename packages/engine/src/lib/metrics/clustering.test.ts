import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

const v = (r: { readonly estimate: { readonly value: number } }): number => r.estimate.value

describe('clustering scores', () => {
  it('returns homogeneity, completeness and V-measure together as separately', () => {
    const t = [0, 0, 1, 1, 2, 2, 2]
    const p = [0, 0, 1, 2, 2, 2, 1]
    const all = M.homogeneityCompletenessVMeasure(t, p, { beta: 2 })
    expect(all.homogeneity).toBeCloseTo(v(M.homogeneityScore(t, p)), 15)
    expect(all.completeness).toBeCloseTo(v(M.completenessScore(t, p)), 15)
    expect(all.vMeasure).toBeCloseTo(v(M.vMeasureScore(t, p, { beta: 2 })), 15)
  })
})
