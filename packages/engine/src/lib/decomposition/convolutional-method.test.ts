import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { Log, metricSeries, session } from '../trace/index.ts'
import {
  type ConvolutionalDictionaryConfig,
  convolutionalDictionaryMethod,
  convolutionalDictionaryProblem,
} from './convolutional-method.ts'

const small: Partial<ConvolutionalDictionaryConfig> = {
  n: 200,
  pulses: 2,
  pulseLength: 10,
  occurrences: 4,
  filters: 2,
  length: 14,
  codeSteps: 15,
  filterSteps: 30,
}

describe('convolutionalDictionaryMethod', () => {
  it('runs as a session: scalars on the tape, the filters, maps and recovery probed at any step', () => {
    const s = session(convolutionalDictionaryMethod, {
      config: { ...small, initialisation: 'random' },
      logs: { 'cdl/objective': Log(), 'cdl/nonzeros': Log() },
      mode: 'eager',
      steps: 25,
      seed: 1,
    })
    expect(metricSeries(s.run(), 'cdl/objective').values.length).toBeGreaterThan(5)
    const objective = metricSeries(s.run(), 'cdl/objective').values
    expect(objective.at(-1) as number).toBeLessThan(objective[0] as number)
    const filters = s.probe(3, 'cdl/filters') as { kind: string; series: unknown[] }
    expect(filters.kind).toBe('curves')
    expect(filters.series.length).toBe(2)
    expect((s.probe(2, 'cdl/planted') as { series: unknown[] }).series.length).toBe(2)
    // The learnt filters match the planted pulses at the end, up to shift and sign, and better than at the start.
    const early = s.probe(0, 'cdl/recovery') as number
    const late = s.probe(s.state.t, 'cdl/recovery') as number
    expect(late).toBeGreaterThan(0.98)
    expect(late).toBeGreaterThan(early)
  })

  it('builds the problem from the configuration: the same key makes the same signal, another another', () => {
    const full: ConvolutionalDictionaryConfig = {
      n: 200,
      pulses: 2,
      pulseLength: 10,
      occurrences: 4,
      noise: 0.02,
      filters: 2,
      length: 14,
      lambda: 0.1,
      codeSteps: 15,
      filterSteps: 30,
      initialisation: 'energy',
    }
    const a = convolutionalDictionaryProblem(full, key(0))
    const b = convolutionalDictionaryProblem(full, key(0))
    const c = convolutionalDictionaryProblem(full, key(1))
    expect(Array.from(a.signal)).toEqual(Array.from(b.signal))
    expect(Array.from(a.signal)).not.toEqual(Array.from(c.signal))
    expect(a.signal.length).toBe(200)
    expect(a.planted.length).toBe(2)
    expect(a.planted[0]?.length).toBe(10)
    expect(a.plantedCodes[0]?.length).toBe(191)
    expect(a.options).toMatchObject({ filters: 2, length: 14, lambda: 0.1, initialisation: 'energy' })
  })

  it('forks from a past step and continues the same run', () => {
    const a = session(convolutionalDictionaryMethod, { config: small, mode: 'eager', steps: 6, seed: 3 })
    const fork = a.fork(3)
    fork.advance()
    fork.advance()
    fork.advance()
    expect(fork.state.objective).toBe(a.state.objective)
  })
})
