import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { dishonestCasino } from '../../lib/datasets/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { metricSeries, tensorSeries } from '../../lib/trace/index.ts'
import { hmm } from './hmm.ts'

describe('hmm', () => {
  const { data } = generate(dishonestCasino, { key: key(1), knobs: { length: 400 } })

  it('learns the casino as a session with the posteriors as a probe, and fits, decodes and scores', () => {
    const s = hmm.session(data, { key: key(2), mode: 'eager', steps: 40 })
    const run = s.run()
    const ll = metricSeries(run, 'solver/logLikelihood').values
    expect(ll.length).toBeGreaterThan(10)
    expect(ll.every((v, i) => i === 0 || v >= (ll[i - 1] ?? 0) - 1e-9)).toBe(true)
    expect(tensorSeries(run, 'solver/transition').at(-1)?.shape).toEqual([2, 2])
    const reading = s.probe(10, 'model/posterior') as {
      kind: string
      posteriors: number[][]
      truth?: number[]
      states: string[]
    }
    expect(reading.kind).toBe('stateSequence')
    expect(reading.posteriors.length).toBe(400)
    expect(reading.states).toEqual(['fair', 'loaded'])
    expect(reading.truth?.length).toBe(400)

    const fitted = hmm.fit(data, { key: key(2) })
    expect(fitted.params.kind).toBe('categorical')
    expect(fitted.exposition.some((r) => r.kind === 'stateSequence')).toBe(true)
    expect(toFlat(fitted.predict(data.x)).length).toBe(400)
    expect(fitted.score?.(data)).toBeLessThan(0)
    // Real-valued data fall back to Gaussian emissions.
    const real = hmm.fit({ x: [[0.1], [0.2], [5.1], [4.9], [0.3], [5.2], [5.0], [0.1]] }, { key: key(0) })
    expect(real.params.kind).toBe('gaussian')
  })
})
