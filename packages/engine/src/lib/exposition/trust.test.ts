import { describe, expect, it } from 'vitest'
import { calibrationRecord, conformalRecord, setSizes } from './index.ts'

describe('trust records', () => {
  it('check calibration curves', () => {
    const curve = { name: 'm', meanPredicted: [0.2, 0.8], observed: [0.1, 0.9], counts: [5, 5], ece: 0.1, mce: 0.1 }
    expect(calibrationRecord({ nBins: 2, strategy: 'uniform', curves: [curve] }).kind).toBe('calibration')
    expect(() => calibrationRecord({ nBins: 2, strategy: 'uniform', curves: [{ ...curve, counts: [5] }] })).toThrow(
      /m\.counts/,
    )
    expect(() =>
      calibrationRecord({ nBins: 2, strategy: 'uniform', curves: [{ ...curve, map: { x: [0, 1], y: [0] } }] }),
    ).toThrow(/map\.y/)
  })

  it('check conformal records and count set sizes', () => {
    const base = { task: 'regression' as const, method: 'split', alpha: 0.1, coverage: 0.9, x: [0, 1] }
    expect(() => conformalRecord({ ...base, lower: [0] })).toThrow(/lower/)
    expect(() => conformalRecord({ ...base, alpha: 1 })).toThrow(/alpha/)
    expect(() => conformalRecord({ ...base, stream: { t: [0, 1], coverage: [1] } })).toThrow(/stream\.coverage/)
    const sets = conformalRecord({
      task: 'classification',
      method: 'lac',
      alpha: 0.2,
      coverage: 2 / 3,
      classes: ['a', 'b', 'c'],
      sets: [[0], [0, 1], [], [2]],
      truth: [0, 2, 1, 2],
    })
    const s = setSizes(sets)
    expect(s.counts).toEqual([1, 2, 1, 0])
    expect(s.coverage[1]).toBe(1)
    expect(s.coverage[2]).toBe(0)
    expect(s.coverage[3]).toBeNaN()
  })
})
