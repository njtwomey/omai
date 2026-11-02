import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { Log, metricSeries, session } from '../trace/index.ts'
import { type FieldMethodConfig, fieldProblem, fitFieldMethod } from './index.ts'

const small = (more: Partial<FieldMethodConfig>): Partial<FieldMethodConfig> => ({
  resolution: 6,
  width: 8,
  depth: 1,
  probeResolution: 5,
  ...more,
})

describe('fitFieldMethod', () => {
  it('runs as a session: scalar paths on the tape, the decoded field and the whole-data error probed at any step', () => {
    const s = session(fitFieldMethod, {
      config: small({ encoding: { kind: 'frequency', octaves: 3, includeInput: true } }),
      logs: { 'field/loss': Log(), 'field/psnr': Log() },
      mode: 'eager',
      steps: 10,
    })
    expect(metricSeries(s.run(), 'field/loss').values.length).toBe(10)
    expect(metricSeries(s.run(), 'field/psnr').values.length).toBe(10)
    const grid = s.probe(5, 'field/render') as { kind: string; values: number[]; channels: number }
    expect(grid.kind).toBe('fieldGrid')
    expect(grid.values.length).toBe(25 * grid.channels)
    const early = s.probe(0, 'field/fullMse') as number
    const late = s.probe(10, 'field/fullMse') as number
    expect(late).toBeLessThan(early)
  })

  it('builds the problem from the slots: a distance field has one output and the eikonal term when weighted', () => {
    const base: FieldMethodConfig = {
      signal: 'torus',
      resolution: 6,
      dim: 3,
      encoding: { kind: 'none' },
      backbone: { kind: 'siren', omega0: 30 },
      width: 8,
      depth: 1,
      optimiser: { kind: 'adam' },
      learningRate: 0.001,
      batchSize: 0,
      eikonalWeight: 0.1,
      probeResolution: 4,
    }
    const p = fieldProblem(base, key(0))
    expect(p.data.x.shape).toEqual([36, 3])
    expect(p.model.outputDim).toBe(1)
    expect(p.options.eikonal?.weight).toBe(0.1)
    expect(fieldProblem({ ...base, eikonalWeight: 0 }, key(0)).options.eikonal).toBeUndefined()
    expect(fieldProblem({ ...base, signal: 'image' }, key(0)).model.outputDim).toBe(3)
  })

  it('forks from a past step and continues the same run', () => {
    const a = session(fitFieldMethod, { config: small({}), mode: 'eager', steps: 8, seed: 3 })
    const fork = a.fork(4)
    fork.advance()
    expect((fork.state as { loss: number }).loss).toBe((a.state as { loss: number }).loss)
  })
})
