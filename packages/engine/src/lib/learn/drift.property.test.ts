import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { changepointSeries } from '../datasets/index.ts'
import { key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { adwin, ddm, eddm, kswin, populationStabilityIndex } from './drift.ts'

/** Error bits: Bernoulli(p0) until `change`, Bernoulli(p1) after. */
function errorBits(seed: number, n: number, change: number, p0: number, p1: number): number[] {
  const u = toFlat(uniform(key(seed), [n]))
  return Array.from({ length: n }, (_, t) => ((u[t] ?? 1) < (t < change ? p0 : p1) ? 1 : 0))
}

/** A series with two planted mean changes (segments of at least 150 steps) whose jumps are all at least `minJump`. */
function plantedMeans(seed: number, minJump: number): { series: number[]; changepoints: number[] } | undefined {
  const g = generate(changepointSeries, {
    key: key(seed),
    knobs: { n: 420, kind: 'mean', hazard: 0.5, minSegment: 150, noise: 1, shift: 8 },
  })
  const segments = (g.truth.extra as { segments: { parameters: { mean: number } }[] }).segments
  const means = segments.map((s) => s.parameters.mean)
  for (let i = 1; i < means.length; i++) if (Math.abs((means[i] ?? 0) - (means[i - 1] ?? 0)) < minJump) return undefined
  return { series: Array.from(toFlat(g.data.values)), changepoints: [...(g.truth.changepoints ?? [])] }
}

/** The first alarm at or after `from`, and the number of alarms strictly before it. */
function alarms(drifts: readonly number[], from: number, until: number): { early: number; first: number | undefined } {
  return {
    early: drifts.filter((t) => t < from).length,
    first: drifts.find((t) => t >= from && t < until),
  }
}

describe('drift detectors on planted changepoints', () => {
  it('ADWIN and KSWIN stay quiet before a mean change and fire within a bounded delay after it', () => {
    let tested = 0
    for (let seed = 0; seed < 80 && tested < 12; seed++) {
      const p = plantedMeans(seed, 5)
      if (p === undefined || p.changepoints.length < 2) continue
      tested++
      const [c1 = 0, c2 = 0] = p.changepoints
      const a = alarms(adwin(p.series, { delta: 0.002 }).drifts, c1, c2)
      expect(a.early, `adwin seed ${seed}`).toBe(0)
      expect(a.first, `adwin seed ${seed}`).toBeDefined()
      expect((a.first ?? 0) - c1, `adwin delay, seed ${seed}`).toBeLessThanOrEqual(25)

      const k = alarms(kswin(p.series, { alpha: 1e-4, windowSize: 100, statSize: 30, key: key(seed) }).drifts, c1, c2)
      expect(k.early, `kswin seed ${seed}`).toBe(0)
      expect(k.first, `kswin seed ${seed}`).toBeDefined()
      expect((k.first ?? 0) - c1, `kswin delay, seed ${seed}`).toBeLessThanOrEqual(45)
    }
    expect(tested).toBe(12)
  })

  it('ADWIN stays quiet on a stationary stream for any seed', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 10_000 }), (seed) => {
        const g = generate(changepointSeries, {
          key: key(seed),
          knobs: { n: 300, kind: 'mean', hazard: 0.0001, noise: 1 },
        })
        if ((g.truth.changepoints ?? []).length > 0) return true
        return adwin(toFlat(g.data.values), { delta: 0.002 }).drifts.length === 0
      }),
      { numRuns: 25, seed: 7 },
    )
  })

  it('DDM and EDDM fire after a planted rise in the error rate, within a delay bound, rarely before it', () => {
    const change = 300
    let ddmQuiet = 0
    let eddmRateBefore = 0
    let eddmRateAfter = 0
    const seeds = 40
    const ddmDelays: number[] = []
    const eddmDelays: number[] = []
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    for (let seed = 0; seed < seeds; seed++) {
      const bits = errorBits(500 + seed, 600, change, 0.3, 0.7)
      const d = alarms(ddm(bits).drifts, change, 600)
      if (d.early === 0) ddmQuiet++
      expect(d.first, `ddm seed ${seed}`).toBeDefined()
      ddmDelays.push((d.first ?? 0) - change)

      // EDDM watches the distance between errors: a rate rise shrinks it. It wants rare errors and a long stream.
      const slow = errorBits(700 + seed, 3000, 2000, 0.03, 0.2)
      const eddmDrifts = eddm(slow, { minErrors: 30 }).drifts
      const e = alarms(eddmDrifts, 2000, 3000)
      eddmRateBefore += e.early / 2000 / seeds
      eddmRateAfter += eddmDrifts.filter((t) => t >= 2000).length / 1000 / seeds
      expect(e.first, `eddm seed ${seed}`).toBeDefined()
      eddmDelays.push((e.first ?? 0) - 2000)
    }
    // DDM has some false alarms on noisy streams (about one run in six over 300 steps at this error rate); EDDM, which
    // watches the spacing of rare errors, has more, so its alarms are judged by their rate before and after the change.
    expect(ddmQuiet / seeds).toBeGreaterThanOrEqual(0.7)
    expect(mean(ddmDelays)).toBeLessThan(90)
    expect(Math.max(...ddmDelays)).toBeLessThan(250)
    expect(mean(eddmDelays)).toBeLessThan(150)
    expect(eddmRateAfter).toBeGreaterThan(4 * eddmRateBefore)
  })

  it('DDM and ADWIN agree with straightforward implementations of their published definitions', () => {
    // Gama et al. (2004): p, s = sqrt(p (1 - p) / n); keep the minimum of p + s; drift at p_min + 3 s_min; reset on drift.
    const ddmReference = (bits: readonly number[], minN: number): number[] => {
      const out: number[] = []
      let [n, p, pMin, sMin] = [0, 0, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]
      bits.forEach((x, t) => {
        n++
        p += (x - p) / n
        const s = Math.sqrt((p * (1 - p)) / n)
        if (n < minN) return
        if (p + s < pMin + sMin) [pMin, sMin] = [p, s]
        if (p + s >= pMin + 3 * sMin) {
          out.push(t)
          ;[n, p, pMin, sMin] = [0, 0, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]
        }
      })
      return out
    }
    for (let seed = 0; seed < 10; seed++) {
      const bits = errorBits(seed, 500, 250, 0.2, 0.6)
      expect(ddm(bits).drifts).toEqual(ddmReference(bits, 30))
    }

    // Bifet and Gavalda (2007), ADWIN0: drop the oldest element while some split of the window has
    // |mu0 - mu1| >= sqrt(2 / m * var * ln(2 / delta')) + 2 / (3 m) * ln(2 / delta'), m = 1 / (1 / n0 + 1 / n1), delta' = delta / n.
    const adwinReference = (
      xs: readonly number[],
      delta: number,
      minSub: number,
    ): { drifts: number[]; sizes: number[] } => {
      let window: number[] = []
      const drifts: number[] = []
      const sizes: number[] = []
      xs.forEach((x, t) => {
        window.push(x)
        let again = true
        while (again) {
          again = false
          const n = window.length
          const mean = window.reduce((a, b) => a + b, 0) / n
          const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / n
          const log = Math.log(2 / (delta / n))
          for (let n0 = minSub; n0 <= n - minSub; n0++) {
            const m0 = window.slice(0, n0).reduce((a, b) => a + b, 0) / n0
            const m1 = window.slice(n0).reduce((a, b) => a + b, 0) / (n - n0)
            const m = 1 / (1 / n0 + 1 / (n - n0))
            if (Math.abs(m0 - m1) >= Math.sqrt((2 / m) * variance * log) + (2 / (3 * m)) * log) {
              window = window.slice(1)
              if (!drifts.includes(t)) drifts.push(t)
              again = true
              break
            }
          }
        }
        sizes.push(window.length)
      })
      return { drifts, sizes }
    }
    for (let seed = 0; seed < 4; seed++) {
      const p = plantedMeans(seed, 4)
      const series = (p?.series ?? plantedMeans(seed + 100, 0)?.series ?? []).slice(0, 260)
      // One bucket per element (no merging), so the histogram is the window itself.
      const mine = adwin(series, { delta: 0.01, minSubwindow: 5, maxBuckets: 100_000 })
      const ref = adwinReference(series, 0.01, 5)
      expect(mine.drifts).toEqual(ref.drifts)
      expect(mine.windowSizes).toEqual(ref.sizes)
    }
  })

  it('the detection delay shrinks as the jump grows', () => {
    const delay = (p1: number): number => {
      let total = 0
      for (let seed = 0; seed < 10; seed++) {
        const first = ddm(errorBits(900 + seed, 600, 300, 0.3, p1)).drifts.find((t) => t >= 300)
        total += (first ?? 600) - 300
      }
      return total / 10
    }
    expect(delay(0.9)).toBeLessThan(delay(0.5))
  })
})

describe('populationStabilityIndex properties', () => {
  it('is nonnegative, zero on the reference itself, and grows with the size of a shift', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -5, max: 5, noNaN: true }), { minLength: 30, maxLength: 80 }), (xs) => {
        const same = populationStabilityIndex(xs, xs, { nBins: 5 })
        const shifted = populationStabilityIndex(
          xs,
          xs.map((v) => v + 2),
          { nBins: 5 },
        )
        return same.psi >= 0 && same.psi < 1e-9 && shifted.psi >= same.psi
      }),
      { numRuns: 40, seed: 3 },
    )
  })
})
