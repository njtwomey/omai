import { describe, expect, it } from 'vitest'
import { featureAttributions, importances } from './attributions.ts'
import { isExposition, recordsOf } from './record.ts'

const base = {
  method: 'exact',
  features: ['a', 'b'],
  output: 'prediction',
  baseValue: 1,
  values: [
    [0.5, -0.25],
    [1, 0],
  ],
  data: [
    [1, 2],
    [3, 4],
  ],
  predictions: [1.25, 2],
}

describe('feature attributions', () => {
  it('build a tagged record that recordsOf finds', () => {
    const r = featureAttributions(base)
    expect(r.kind).toBe('featureAttributions')
    expect(isExposition(r)).toBe(true)
    expect(recordsOf([r], 'featureAttributions')).toHaveLength(1)
  })

  it('reject rows, data and predictions that do not match', () => {
    expect(() => featureAttributions({ ...base, values: [[1, 2, 3]] })).toThrow(/values\[0\]/)
    expect(() => featureAttributions({ ...base, data: [[1, 2]] })).toThrow(/data has 1/)
    expect(() => featureAttributions({ ...base, data: [[1, 2], [3]] })).toThrow(/data\[1\]/)
    expect(() => featureAttributions({ ...base, predictions: [1] })).toThrow(/predictions/)
    expect(() => featureAttributions({ ...base, indices: [0] })).toThrow(/indices/)
    expect(() => featureAttributions({ ...base, localFit: [1, 2, 3] })).toThrow(/localFit/)
    expect(() => featureAttributions({ ...base, baseValues: [1] })).toThrow(/baseValues/)
    expect(() => featureAttributions({ ...base, partners: [1] })).toThrow(/partners/)
    expect(() => featureAttributions({ ...base, partners: [1, 2] })).toThrow(/partner 2/)
    expect(() => featureAttributions({ ...base, standardErrors: [[0, 0], [0]] })).toThrow(/standardErrors\[1\]/)
  })

  it('check interaction matrices and convergence', () => {
    const m = [
      [0.5, 0],
      [0, -0.25],
    ]
    expect(featureAttributions({ ...base, interactions: [m, m] }).interactions).toHaveLength(2)
    expect(() => featureAttributions({ ...base, interactions: [m] })).toThrow(/interactions has 1/)
    expect(() => featureAttributions({ ...base, interactions: [m, [[0, 0]]] })).toThrow(/interactions\[1\] has 1/)
    expect(() => featureAttributions({ ...base, interactions: [m, [[0, 0], [0]]] })).toThrow(/interactions\[1\]\[1\]/)
    const convergence = {
      row: 1,
      samples: [10, 20],
      estimates: [
        [1, 0],
        [0.9, 0.1],
      ],
    }
    expect(featureAttributions({ ...base, convergence }).convergence?.samples).toEqual([10, 20])
    expect(() => featureAttributions({ ...base, convergence: { ...convergence, row: 2 } })).toThrow(/row 2/)
    expect(() => featureAttributions({ ...base, convergence: { ...convergence, samples: [1] } })).toThrow(
      /convergence\.estimates/,
    )
    expect(() => featureAttributions({ ...base, convergence: { ...convergence, standardErrors: [[0, 0]] } })).toThrow(
      /convergence\.standardErrors/,
    )
  })
})

describe('importances', () => {
  it('check one mean, spread and set of repeats per feature', () => {
    const r = importances({
      method: 'permutation',
      features: ['a', 'b'],
      mean: [0.2, 0],
      spread: [0.1, 0],
      repeats: [
        [0.1, 0.3],
        [0, 0],
      ],
      metric: 'accuracy',
      direction: 'higher',
    })
    expect(isExposition(r)).toBe(true)
    expect(() => importances({ ...r, mean: [1] })).toThrow(/mean/)
    expect(() => importances({ ...r, spread: [1] })).toThrow(/spread/)
    expect(() => importances({ ...r, repeats: [[1]] })).toThrow(/repeats has 1/)
    expect(() => importances({ ...r, repeats: [[1, 2], [1]] })).toThrow(/repeats\[1\]/)
  })
})
