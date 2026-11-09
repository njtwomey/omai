import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { gaussianNaiveBayes } from '../naive-bayes.ts'
import { fixedThresholdClassifier, tunedThresholdClassifierCV } from './threshold.ts'

const dataset = fc.integer({ min: 0, max: 400 }).map((seed) => {
  // A deterministic two-class problem from a seed: no randomness in the test body.
  const n = 40
  const x = Array.from({ length: n }, (_, i) => [
    Math.sin(seed + i * 1.7) * 2 + (i % 4 === 0 ? 1.5 : 0),
    Math.cos(seed * 0.3 + i),
  ])
  const y = Array.from({ length: n }, (_, i) => (i % 4 === 0 ? 1 : 0))
  return { x, y }
})

describe('threshold tuning properties', () => {
  it('raising a fixed threshold never adds positive predictions', () => {
    fc.assert(
      fc.property(
        dataset,
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (d, a, b) => {
          const [lo, hi] = a <= b ? [a, b] : [b, a]
          const count = (t: number): number =>
            toFlat(
              fixedThresholdClassifier({ estimator: gaussianNaiveBayes(), threshold: t }).fit(d).predict(d.x),
            ).filter((v) => v === 1).length
          expect(count(hi)).toBeLessThanOrEqual(count(lo))
        },
      ),
      { numRuns: 25 },
    )
  })

  it('the tuned threshold attains the best point of its curve, which no other grid point beats', () => {
    fc.assert(
      fc.property(dataset, (d) => {
        const f = tunedThresholdClassifierCV({ estimator: gaussianNaiveBayes(), cv: 4, thresholds: 30 }).fit(d, {
          key: key(0),
        })
        const curve = f.params.curve
        expect(curve?.scores.length).toBe(30)
        expect(f.params.bestScore).toBe(Math.max(...(curve?.scores ?? [])))
        const at = curve?.thresholds.indexOf(f.params.threshold) ?? -1
        expect(curve?.scores[at]).toBe(f.params.bestScore)
        for (const s of curve?.scores ?? []) {
          expect(s).toBeGreaterThanOrEqual(0)
          expect(s).toBeLessThanOrEqual(1 + 1e-12)
        }
      }),
      { numRuns: 15 },
    )
  })
})
