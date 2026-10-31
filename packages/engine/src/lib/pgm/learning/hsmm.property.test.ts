import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { iterate } from '../../trace/index.ts'
import { type HsmmParameters, hsmmAlgorithm, hsmmPosterior, hsmmSample } from './hsmm.ts'

/** Random categorical semi-Markov parameters (3 states, 3 symbols, durations up to 4) from a seed. */
function randomParameters(seed: number): HsmmParameters {
  const u = Array.from(toFlat(uniform(key(seed), [3 + 6 + 9 + 12])))
  const row = (from: number, n: number, zero = -1): number[] => {
    const r = Array.from({ length: n }, (_, i) => (i === zero ? 0 : 0.05 + (u[from + i] ?? 0)))
    const z = r.reduce((a, v) => a + v, 0)
    return r.map((v) => v / z)
  }
  return {
    kind: 'categorical',
    start: row(0, 3),
    transition: [0, 1, 2].map((i) => row(3 + 2 * i, 3, i)),
    emission: [0, 1, 2].map((i) => row(9 + 3 * i, 3)),
    duration: [0, 1, 2].map((i) => row(18 + 4 * i, 4)),
  }
}

const seed = fc.integer({ min: 0, max: 2 ** 31 })

describe('hidden semi-Markov properties', () => {
  it('the occupancy is a distribution at every step and the expected segment count is the moves plus one', () => {
    fc.assert(
      fc.property(seed, seed, (s, t) => {
        const p = randomParameters(s)
        const x = hsmmSample(p, 12, key(t)).observations as number[]
        const post = hsmmPosterior(p, x)
        for (const r of post.occupancy) expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9)
        expect(post.duration.flat().reduce((a, v) => a + v, 0)).toBeCloseTo(
          post.transition.flat().reduce((a, v) => a + v, 0) + 1,
          9,
        )
        expect(post.start.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9)
      }),
      { numRuns: 25 },
    )
  })

  it('EM never lowers the log likelihood, from random parameters, on data from other random parameters', () => {
    fc.assert(
      fc.property(seed, seed, (a, b) => {
        const truth = randomParameters(a)
        const sequences = [0, 1, 2].map((i) => hsmmSample(truth, 25, key(b + i)).observations as number[])
        const alg = hsmmAlgorithm(sequences, randomParameters(b), { tol: -1 })
        const lls: number[] = []
        for (const { state } of iterate(alg, { key: key(0), maxSteps: 6 }))
          if (state.logLikelihood !== null) lls.push(state.logLikelihood)
        for (let t = 1; t < lls.length; t++) expect(lls[t] ?? 0).toBeGreaterThanOrEqual((lls[t - 1] ?? 0) - 1e-8)
      }),
      { numRuns: 15 },
    )
  })

  it('has a sample of the requested length', () => {
    expect(hsmmSample(randomParameters(1), 9, key(1)).states.length).toBe(9)
  })
})
