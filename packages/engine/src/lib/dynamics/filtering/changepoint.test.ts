import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { exactBocpd } from '../../datasets/index.ts'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { binarySegmentation, bocpd, constantHazard, pelt } from './changepoint.ts'

const y = [0.1, -0.2, 0.05, 0.3, 4.1, 3.8, 4.4, 3.9, 4.2, 0.2, -0.1, 0.4]

describe('changepoint detection', () => {
  it('BOCPD agrees with the exact BOCPD of the datasets generator', () => {
    const a = bocpd(y, { kind: 'normal-known-variance', sigma: 0.4, priorMean: 0, priorStd: 3 }, 0.1)
    const b = exactBocpd(y, { kind: 'mean', sigma: 0.4, priorMean: 0, priorStd: 3 }, 0.1)
    expect(a.mapRunLength).toEqual(b.mapRunLength)
    expect(Math.abs(a.logEvidence - b.logEvidence)).toBeLessThan(1e-10)
    const v = bocpd(y, { kind: 'normal-known-mean', mean: 2, alpha: 3, beta: 2 }, 0.05)
    const w = exactBocpd(y, { kind: 'variance', mean: 2, alpha: 3, beta: 2 }, 0.05)
    expect(Math.abs(v.logEvidence - w.logEvidence)).toBeLessThan(1e-10)
    const counts = [1, 0, 2, 1, 7, 9, 8, 6, 1, 0, 2]
    const p = bocpd(counts, { kind: 'poisson-gamma', shape: 2, rate: 1 }, constantHazard(20))
    const e = exactBocpd(counts, { kind: 'rate', shape: 2, rate: 1 }, 1 / 20)
    expect(Math.abs(p.logEvidence - e.logEvidence)).toBeLessThan(1e-10)
  })

  it('finds the changes with unknown mean and variance, and in binary data', () => {
    const r = bocpd(y, { kind: 'normal-gamma', mu0: 0, kappa0: 0.1, alpha0: 1, beta0: 0.1 }, 0.05)
    expect(r.changeProbability[4]).toBeGreaterThan(0.5)
    expect(r.changeProbability[9]).toBeGreaterThan(0.5)
    expect(r.runLength.every((rl) => Math.abs(rl.reduce((a, v) => a + v, 0) - 1) < 1e-9)).toBe(true)
    const bits = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    const b = bocpd(bits, { kind: 'bernoulli-beta', alpha: 0.5, beta: 0.5 }, 0.05)
    expect(b.mapRunLength.at(-1)).toBeLessThan(10)
    expect(() => bocpd(y, { kind: 'bernoulli-beta', alpha: 1, beta: 1 }, 1)).toThrow()
  })

  it('PELT finds the optimal partition (property: brute force on short series)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 4, maxLength: 9 }),
        fc.double({ min: 0.5, max: 5, noNaN: true }),
        (s, pen) => {
          const n = s.length
          const sse = (a: number, b: number): number => {
            const seg = s.slice(a, b)
            const m = seg.reduce((x, v) => x + v, 0) / seg.length
            return seg.reduce((x, v) => x + (v - m) ** 2, 0)
          }
          // All partitions into segments of length >= 2.
          let best = Number.POSITIVE_INFINITY
          const walk = (start: number, acc: number): void => {
            if (start === n) {
              best = Math.min(best, acc)
              return
            }
            for (let e = start + 2; e <= n; e++) if (n - e === 0 || n - e >= 2) walk(e, acc + sse(start, e) + pen)
          }
          walk(0, 0)
          const bk = pelt(s, pen)
          let cost = 0
          let a = 0
          for (const b of bk) {
            cost += sse(a, b) + pen
            a = b
          }
          return Math.abs(cost - best) < 1e-8 * Math.max(1, best)
        },
      ),
      { numRuns: 40 },
    )
  })

  it('binary segmentation and PELT find the breaks of a piecewise-constant series', () => {
    const noise = toFlat(normal(key(7), [90]))
    const series = noise.map((v, i) => 0.3 * v + (i < 30 ? 0 : i < 60 ? 3 : -1))
    expect(pelt(series, 5)).toEqual([30, 60, 90])
    expect(binarySegmentation(series, { changepoints: 2 })).toEqual([30, 60, 90])
    expect(() => binarySegmentation(series, {})).toThrow()
  })
})
