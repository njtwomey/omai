import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { logisticRegression } from '../../testing/learn.ts'
import { key, normal, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import {
  backwardCorrectedLoss,
  coTeaching,
  coTeachingAlgorithm,
  forwardCorrectedLoss,
  smallLossSelection,
} from './noisy-labels.ts'

describe('small-loss selection', () => {
  it('keeps the smallest losses, a ceiling of the fraction, ties by index', () => {
    expect(smallLossSelection([0.5, 2, 0.1, 0.9], 0.5)).toEqual([2, 0])
    expect(smallLossSelection([1, 1, 1], 0.4)).toEqual([0, 1])
    expect(smallLossSelection([3, 1], 0.01)).toEqual([1])
    expect(smallLossSelection([3, 1], 1)).toEqual([1, 0])
    expect(() => smallLossSelection([1], 0)).toThrow(/keep/)
  })

  it('selects a prefix of the sorted losses', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 10, noNaN: true }), { minLength: 1, maxLength: 40 }),
        fc.double({ min: 0.05, max: 1, noNaN: true }),
        (l, keep) => {
          const sel = smallLossSelection(l, keep)
          expect(sel.length).toBe(Math.max(1, Math.ceil(keep * l.length - 1e-12)))
          const chosen = new Set(sel)
          const worstKept = Math.max(...sel.map((i) => l[i] ?? 0))
          l.forEach((v, i) => {
            if (!chosen.has(i)) expect(v).toBeGreaterThanOrEqual(worstKept)
          })
        },
      ),
    )
  })
})

describe('loss correction', () => {
  const t = [
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
  ]

  it('forward correction is the cross-entropy of the noisy-label probability; identity noise changes nothing', () => {
    const p = [[0.6, 0.3, 0.1]]
    // P(noisy = 1) = 0.6 * 0.2 + 0.3 * 0.8 + 0.1 * 0.2.
    expect(toFlat(forwardCorrectedLoss(p, [1], t))[0]).toBeCloseTo(-Math.log(0.12 + 0.24 + 0.02), 12)
    const id = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    expect(toFlat(forwardCorrectedLoss(p, [2], id))[0]).toBeCloseTo(-Math.log(0.1), 12)
    expect(toFlat(backwardCorrectedLoss(p, [2], id))[0]).toBeCloseTo(-Math.log(0.1), 12)
  })

  it('backward correction is unbiased: its expectation over the noisy label is the clean loss', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.double({ min: 0.05, max: 1, noNaN: true }),
          fc.double({ min: 0.05, max: 1, noNaN: true }),
          fc.double({ min: 0.05, max: 1, noNaN: true }),
        ),
        fc.integer({ min: 0, max: 2 }),
        ([a, b, c], truth) => {
          const z = a + b + c
          const p = [[a / z, b / z, c / z]]
          const clean = -Math.log((p[0] ?? [])[truth] ?? 1)
          let expected = 0
          for (let noisy = 0; noisy < 3; noisy++)
            expected += (t[truth]?.[noisy] ?? 0) * (toFlat(backwardCorrectedLoss(p, [noisy], t))[0] ?? 0)
          expect(expected).toBeCloseTo(clean, 9)
        },
      ),
    )
  })

  it('the forward loss, averaged over the noise, is smallest at the true class', () => {
    const noisyExpected = (p: number[], truth: number): number =>
      [0, 1, 2].reduce(
        (a, noisy) => a + (t[truth]?.[noisy] ?? 0) * (toFlat(forwardCorrectedLoss([p], [noisy], t))[0] ?? 0),
        0,
      )
    const sure = (c: number): number[] => [0, 1, 2].map((i) => (i === c ? 0.98 : 0.01))
    for (let truth = 0; truth < 3; truth++)
      for (let other = 0; other < 3; other++)
        if (other !== truth) expect(noisyExpected(sure(truth), truth)).toBeLessThan(noisyExpected(sure(other), truth))
  })
})

describe('co-teaching', () => {
  /** Two Gaussian classes in the plane with a fraction of labels flipped; the clean labels are kept as truth. */
  function noisy(seed: number, n: number, rate: number): { x: number[][]; y: number[]; clean: number[] } {
    const u = Float64Array.from(toFlat(uniform(key(seed), [2 * n])))
    const g = Float64Array.from(toFlat(normal(key(seed + 1), [2 * n])))
    const clean = Array.from({ length: n }, (_, i) => ((u[i] ?? 0) < 0.5 ? 0 : 1))
    const x = clean.map((c, i) => [
      (c === 1 ? 1.2 : -1.2) + (g[2 * i] ?? 0),
      (c === 1 ? 1.2 : -1.2) + (g[2 * i + 1] ?? 0),
    ])
    const y = clean.map((c, i) => ((u[n + i] ?? 0) < rate ? 1 - c : c))
    return { x, y, clean }
  }

  it('picks mostly clean rows, and the pair classifies the clean data well', () => {
    const d = noisy(91, 600, 0.3)
    const f = coTeaching(logisticRegression({ C: 1 }), { forgetRate: 0.3, rampRounds: 4, maxRounds: 10 }).fit({
      x: d.x,
      y: d.y,
    })
    const purity = f.cleanRows.filter((i) => d.y[i] === d.clean[i]).length / f.cleanRows.length
    expect(f.cleanRows.length).toBeGreaterThan(0.5 * 600)
    // Seventy percent of the rows are clean; the rows both models keep are clearly cleaner.
    expect(purity).toBeGreaterThan(0.85)
    expect(f.score({ x: d.x, y: d.clean })).toBeGreaterThan(0.85)
  })

  it('ramps the kept fraction down to one minus the forget rate, and the models agree more as it does', () => {
    const d = noisy(93, 300, 0.2)
    const rec = recorder()
    runAlgorithm(
      coTeachingAlgorithm(
        { x: d.x, y: d.y },
        { base: logisticRegression({ C: 1 }), forgetRate: 0.2, rampRounds: 4, maxRounds: 6 },
      ),
      {
        maxSteps: 10,
        tracer: rec,
        key: key(0),
      },
    )
    const keep = metricSeries(rec.run(), 'keep').values
    expect(keep[0]).toBe(1)
    expect(keep.at(-1)).toBeCloseTo(0.8, 12)
    for (let i = 1; i < keep.length; i++) expect(keep[i] ?? 0).toBeLessThanOrEqual((keep[i - 1] ?? 0) + 1e-12)
    expect(metricSeries(rec.run(), 'agreement').values.at(-1)).toBeGreaterThan(0.8)
  })

  it('round-trips its parameters', () => {
    const d = noisy(95, 200, 0.2)
    const est = coTeaching(logisticRegression({ C: 1 }), { maxRounds: 4 })
    const f = est.fit({ x: d.x, y: d.y })
    const again = est.revive(JSON.parse(JSON.stringify(f.params)))
    expect(toFlat(again.predict(d.x))).toEqual(toFlat(f.predict(d.x)))
  })
})
