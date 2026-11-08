import { calibrationRecord, conformalRecord } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { stillOf, viewFor } from './registry.ts'
import { conformalFrames, conformalView, reliabilityView } from './trust.ts'

type Opt = { grid: unknown[]; series: { name?: string; data?: unknown[] }[] }

describe('trust views', () => {
  it('draws reliability curves with their errors and counts', () => {
    const r = calibrationRecord({
      nBins: 2,
      strategy: 'uniform',
      curves: [
        { name: 'raw', meanPredicted: [0.2, 0.8], observed: [0.1, 0.9], counts: [5, 5], ece: 0.1, mce: 0.1 },
        { name: 'a', group: 'g', meanPredicted: [0.3], observed: [0.3], counts: [3], ece: 0, mce: 0 },
      ],
    })
    const o = reliabilityView(r) as unknown as Opt
    expect(o.grid).toHaveLength(2)
    expect(o.series.some((s) => s.name === 'raw (ECE 0.1)')).toBe(true)
    expect(viewFor('calibration').component).toBe('CalibrationView')
  })

  it('draws intervals, scores and a stream, and plays the stream', () => {
    const r = conformalRecord({
      task: 'regression',
      method: 'aci',
      alpha: 0.1,
      coverage: 0.75,
      x: [0, 1, 2, 3],
      y: [0, 1, 5, 3],
      prediction: [0, 1, 2, 3],
      lower: [-1, 0, 1, 2],
      upper: [1, 2, 3, 4],
      scores: [0.5, 1, 2],
      quantile: 1,
      stream: { t: [0, 1, 2, 3], coverage: [1, 1, 2 / 3, 0.75], baseline: [1, 1, 1, 1] },
    })
    expect((conformalView(r) as unknown as Opt).grid).toHaveLength(3)
    const outside = (conformalView(r) as unknown as Opt).series.find((s) => s.name === '__outside')
    expect(outside?.data).toHaveLength(1)
    const frames = conformalFrames(r)
    expect(frames.count).toBe(4)
    expect((stillOf(r) as unknown as Opt).grid).toHaveLength(3)
  })

  it('draws set sizes for classification', () => {
    const r = conformalRecord({
      task: 'classification',
      method: 'lac',
      alpha: 0.1,
      coverage: 1,
      classes: ['a', 'b'],
      sets: [[0], [0, 1]],
      truth: [0, 1],
    })
    const o = conformalView(r) as unknown as Opt
    expect(o.series[0]?.name).toBe('rows')
  })
})
