import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { hmmSample } from '../../lib/pgm/index.ts'
import { key } from '../../lib/random/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import { casinoStudy, dishonestCasino, gaussianHMM } from './hmm.ts'

describe('hidden Markov models', () => {
  it('the dishonest casino: decoding with the true model finds the loaded die', () => {
    const s = casinoStudy({ rolls: 400, learn: false, key: key(1) })
    expect(s.accuracy).toBeGreaterThan(0.65)
    const [grid] = recordsOf(s.exposition, 'posteriorGrid')
    expect(grid?.mean.length).toBe(400)
    expect(grid?.observed?.targets.length).toBe(400)
  })

  it('Baum-Welch learns the casino from the rolls, never lowering the likelihood', () => {
    const s = casinoStudy({ rolls: 1000, key: key(2) })
    expect(s.params.emissionprob?.[1]?.[5]).toBeGreaterThan(0.3)
    expect(s.accuracy).toBeGreaterThan(0.7)
    const [steps] = recordsOf(s.exposition, 'mixtureSteps')
    const obj = steps?.objective ?? []
    for (let t = 1; t < obj.length; t++) expect(obj[t] ?? 0).toBeGreaterThanOrEqual((obj[t - 1] ?? 0) - 1e-8)
    expect(steps?.means[0]?.[0]?.length).toBe(6)
    expect(recordsOf(s.exposition, 'decisionSurface')[0]?.values?.length).toBe(2)
    expect(recordsOf(s.exposition, 'coefficientTable')[0]?.names.length).toBe(6)
  })

  it('a Gaussian HMM separates two regimes and decodes them', () => {
    const truth = {
      ...dishonestCasino(),
      emission: { kind: 'gaussian' as const, means: [[0], [4]], covariances: [[[1]], [[1]]] },
    }
    const sample = hmmSample(truth, key(3), 300)
    const x = fromStorage(Float64Array.from(sample.observations.flat() as number[]), [300, 1], 'float64')
    const f = gaussianHMM({ nComponents: 2 }).fit({ x }, { key: key(4) })
    const path = Array.from(toFlat(f.predict(x)))
    const agree = path.filter((p, t) => p === sample.states[t]).length / 300
    expect(Math.max(agree, 1 - agree)).toBeGreaterThan(0.95)
    expect(f.score({ x })).toBeCloseTo(f.params.logLikelihood, 0)
    expect(f.predictProba(x).shape).toEqual([300, 2])
  })
})
