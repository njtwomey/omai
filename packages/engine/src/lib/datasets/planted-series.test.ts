import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { DIGIT_FORMANTS, type Utterance } from './audio.ts'
import { coupledSeries, garchReturns, plantedSeries, spokenDigits } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)

/** z-normalised Euclidean distance between two windows of a series. */
function zdist(y: readonly number[], a: number, b: number, m: number): number {
  const norm = (s: number): number[] => {
    const w = y.slice(s, s + m)
    const mu = mean(w)
    const sd = Math.sqrt(mean(w.map((v) => (v - mu) ** 2))) || 1
    return w.map((v) => (v - mu) / sd)
  }
  const x = norm(a)
  const z = norm(b)
  return Math.sqrt(x.reduce((s, v, i) => s + (v - (z[i] ?? 0)) ** 2, 0))
}

describe('plantedSeries', () => {
  const m = 30
  const { data, truth } = generate(plantedSeries, { key: key(1), knobs: { n: 800, motifLength: m, occurrences: 2 } })
  const y = toFlat(data.values)
  const starts = (truth.extra ?? {})['motifStarts'] as number[]
  const discord = (truth.extra ?? {})['discordStart'] as number

  it('puts the motif copies and the discord in disjoint places', () => {
    const spans = [...starts, discord].sort((a, b) => a - b)
    for (let i = 1; i < spans.length; i++) expect((spans[i] ?? 0) - (spans[i - 1] ?? 0)).toBeGreaterThanOrEqual(m)
    expect(truth.changepoints).toEqual([400])
  })

  it('has the closest pair of windows at the planted motif and a discord among the most isolated windows', () => {
    const count = y.length - m + 1
    let bestPair = [0, 0]
    let bestDist = Number.POSITIVE_INFINITY
    const nearest = new Array<number>(count).fill(Number.POSITIVE_INFINITY)
    for (let a = 0; a < count; a++)
      for (let b = a + m; b < count; b++) {
        const d = zdist(y, a, b, m)
        if (d < bestDist) {
          bestDist = d
          bestPair = [a, b]
        }
        nearest[a] = Math.min(nearest[a] ?? 0, d)
        nearest[b] = Math.min(nearest[b] ?? 0, d)
      }
    const near = (s: number, t: number): boolean => Math.abs(s - t) <= 3
    expect(near(bestPair[0] ?? 0, starts[0] ?? -99) && near(bestPair[1] ?? 0, starts[1] ?? -99)).toBe(true)
    // The windows over the discord are among the most isolated 5 per cent (a random walk has other odd windows too).
    const sorted = [...nearest].sort((a, b) => a - b)
    const cut = sorted[Math.floor(0.95 * count)] ?? 0
    expect(Math.max(...nearest.slice(discord - m + 1, discord + m))).toBeGreaterThan(cut)
  })

  it('has a larger noise level after the change point', () => {
    const diffs = y.slice(1).map((v, i) => v - (y[i] ?? 0))
    const sd = (xs: number[]): number => Math.sqrt(mean(xs.map((v) => v * v)))
    const spans = [...starts, discord].flatMap((s) => [s - 1, s + m + 1])
    void spans
    expect(
      sd(
        diffs
          .slice(500, 790)
          .filter((_, i) => !starts.concat(discord).some((s) => i + 500 >= s - 1 && i + 500 <= s + m)),
      ),
    ).toBeGreaterThan(
      sd(diffs.slice(0, 390).filter((_, i) => !starts.concat(discord).some((s) => i >= s - 1 && i <= s + m))),
    )
  })
})

describe('garchReturns', () => {
  it('follows its variance recursion exactly and has volatility clustering and fat tails', () => {
    const { data, truth } = generate(garchReturns, {
      key: key(2),
      knobs: { n: 6000, omega: 0.05, alpha: 0.15, beta: 0.8 },
    })
    const r = toFlat(data.values)
    const v = toFlat((truth.latents ?? {})['variance'] as Tensor)
    for (let t = 1; t < r.length; t++)
      expect(v[t]).toBeCloseTo(0.05 + 0.15 * (r[t - 1] ?? 0) ** 2 + 0.8 * (v[t - 1] ?? 0), 12)
    const kurt = mean(r.map((x) => x ** 4)) / mean(r.map((x) => x ** 2)) ** 2
    expect(kurt).toBeGreaterThan(3.3)
    // Squared returns are autocorrelated.
    const sq = r.map((x) => x * x)
    const mu = mean(sq)
    const ac = mean(sq.slice(1).map((x, i) => (x - mu) * ((sq[i] ?? 0) - mu))) / mean(sq.map((x) => (x - mu) ** 2))
    expect(ac).toBeGreaterThan(0.1)
    expect(Math.abs(mean(sq) - ((truth.extra ?? {})['unconditionalVariance'] as number)) / 1).toBeLessThan(0.15)
  })

  it('standardised returns are unit variance, heavier tailed with Student innovations, and a non-stationary pair is refused', () => {
    const g = generate(garchReturns, {
      key: key(3),
      knobs: { n: 20000, alpha: 0, beta: 0, omega: 1, innovations: 'student' },
    })
    const r = toFlat(g.data.values)
    expect(Math.abs(mean(r.map((x) => x * x)) - 1)).toBeLessThan(0.1)
    expect(mean(r.map((x) => x ** 4))).toBeGreaterThan(4)
    expect(() => generate(garchReturns, { key: key(0), knobs: { alpha: 0.6, beta: 0.5 } })).toThrow(/stationary/)
  })
})

describe('coupledSeries', () => {
  it('a VAR has the planted couplings in its lagged regression and none elsewhere', () => {
    const { data, truth } = generate(coupledSeries, {
      key: key(4),
      knobs: { n: 30000, series: 3, density: 0.5, coupling: 0.5, noise: 1 },
    })
    const adjacency = (truth.extra ?? {})['adjacency'] as number[][]
    const d = 3
    const x = toFlat(data.values)
    const maxParents = Math.max(1, ...adjacency.map((row) => row.reduce((a, b) => a + b, 0)))
    // Least squares of x_i(t) on x(t-1) by the normal equations.
    const rows = x.length / d - 1
    const gram = Array.from({ length: d }, () => new Array<number>(d).fill(0))
    for (let t = 0; t < rows; t++)
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++)
          (gram[a] as number[])[b] = ((gram[a] as number[])[b] ?? 0) + (x[t * d + a] ?? 0) * (x[t * d + b] ?? 0)
    const solve3 = (rhs: number[]): number[] => {
      // Cramer's rule for the 3 by 3 system.
      const det = (m: number[][]): number =>
        (m[0]?.[0] ?? 0) * ((m[1]?.[1] ?? 0) * (m[2]?.[2] ?? 0) - (m[1]?.[2] ?? 0) * (m[2]?.[1] ?? 0)) -
        (m[0]?.[1] ?? 0) * ((m[1]?.[0] ?? 0) * (m[2]?.[2] ?? 0) - (m[1]?.[2] ?? 0) * (m[2]?.[0] ?? 0)) +
        (m[0]?.[2] ?? 0) * ((m[1]?.[0] ?? 0) * (m[2]?.[1] ?? 0) - (m[1]?.[1] ?? 0) * (m[2]?.[0] ?? 0))
      const D = det(gram)
      return [0, 1, 2].map((c) => det(gram.map((row, r) => row.map((v, cc) => (cc === c ? (rhs[r] ?? 0) : v)))) / D)
    }
    for (let i = 0; i < d; i++) {
      const rhs = [0, 1, 2].map((a) => {
        let s = 0
        for (let t = 0; t < rows; t++) s += (x[t * d + a] ?? 0) * (x[(t + 1) * d + i] ?? 0)
        return s
      })
      const beta = solve3(rhs)
      for (let j = 0; j < d; j++) {
        const want = (i === j ? 0.5 : 0) + ((adjacency[i]?.[j] ?? 0) * 0.5) / Math.sqrt(maxParents)
        expect(Math.abs((beta[j] ?? 0) - want)).toBeLessThan(0.04)
      }
    }
    expect(adjacency.every((row, i) => row[i] === 0)).toBe(true)
  })

  it('logistic maps stay in the unit interval, and driven series follow the drivers more than unlinked ones', () => {
    const { data, truth } = generate(coupledSeries, {
      key: key(5),
      knobs: { n: 3000, series: 2, dynamics: 'logistic', density: 1, coupling: 0.6, noise: 0 },
    })
    const x = toFlat(data.values)
    expect(x.every((v) => v > 0 && v < 1)).toBe(true)
    expect(((truth.extra ?? {})['adjacency'] as number[][]).flat().reduce((a, b) => a + b, 0)).toBe(2)
    const a = x.filter((_, i) => i % 2 === 0)
    const b = x.filter((_, i) => i % 2 === 1)
    const ma = mean(a)
    const mb = mean(b)
    const corr =
      mean(a.map((v, i) => (v - ma) * ((b[i] ?? 0) - mb))) /
      Math.sqrt(mean(a.map((v) => (v - ma) ** 2)) * mean(b.map((v) => (v - mb) ** 2)))
    expect(Math.abs(corr)).toBeGreaterThan(0.5)
  })
})

describe('spokenDigits', () => {
  const { data, truth } = generate(spokenDigits, { key: key(6), knobs: { utterances: 24, speakers: 3, noise: 0.02 } })
  const utterances = (truth.extra ?? {})['utterances'] as Utterance[]
  const y = toFlat(data.values)
  const segments = toFlat(data.segments as Tensor)

  it('labels the samples of each utterance with its digit and the gaps with -1', () => {
    expect(utterances).toHaveLength(24)
    for (const u of utterances) {
      expect(segments.slice(u.start, u.end).every((s) => s === u.digit)).toBe(true)
      expect(u.end).toBeLessThanOrEqual(y.length)
    }
    expect(segments[0]).toBe(-1)
    expect(segments.filter((s) => s === -1).length).toBeGreaterThan(24 * 100)
  })

  it('has more energy in utterances than in gaps, and the formant bands tell the digits apart', () => {
    const energy = (a: number, b: number): number => mean(y.slice(a, b).map((v) => v * v))
    const u0 = utterances[0] as Utterance
    expect(energy(u0.start, u0.end)).toBeGreaterThan(20 * energy(0, 100))
    const rate = 8000
    const grid = Array.from({ length: 80 }, (_, i) => 150 + 40 * i)
    /** The amplitude spectrum of a stretch on the grid, max-filtered to fill the gaps between harmonics. */
    const envelope = (a: number, b: number): number[] => {
      const amp = grid.map((f) => {
        let re = 0
        let im = 0
        for (let t = a; t < b; t++) {
          re += (y[t] ?? 0) * Math.cos((2 * Math.PI * f * t) / rate)
          im += (y[t] ?? 0) * Math.sin((2 * Math.PI * f * t) / rate)
        }
        return Math.hypot(re, im)
      })
      return amp.map((_, i) => Math.max(...amp.slice(Math.max(0, i - 3), i + 4)))
    }
    const expected = (f1: number, f2: number, scale: number): number[] =>
      grid.map((f) => Math.exp(-(((f - scale * f1) / 130) ** 2)) + 0.6 * Math.exp(-(((f - scale * f2) / 180) ** 2)))
    const cosine = (p: number[], q: number[]): number =>
      p.reduce((s, v, i) => s + v * (q[i] ?? 0), 0) /
      Math.sqrt(p.reduce((s, v) => s + v * v, 0) * q.reduce((s, v) => s + v * v, 0))
    let right = 0
    for (const u of utterances) {
      const len = u.end - u.start
      const observed = [0, 1, 2].map((q) =>
        envelope(u.start + Math.floor(((q + 0.4) * len) / 3), u.start + Math.floor(((q + 0.9) * len) / 3)),
      )
      const score = (d: number): number =>
        Math.max(
          ...[0.9, 1, 1.1].map((scale) =>
            (DIGIT_FORMANTS[d] ?? []).reduce(
              (acc, [f1, f2], q) => acc + cosine(observed[q] ?? [], expected(f1, f2, scale)),
              0,
            ),
          ),
        )
      const best = Array.from({ length: 10 }, (_, d) => d).reduce((a, d) => (score(d) > score(a) ? d : a), 0)
      if (best === u.digit) right++
    }
    expect(right / utterances.length).toBeGreaterThan(0.7)
  })

  it('is deterministic in the key and speakers keep their pitch', () => {
    const again = generate(spokenDigits, { key: key(6), knobs: { utterances: 24, speakers: 3, noise: 0.02 } })
    expect(Array.from(toFlat(again.data.values).slice(0, 2000))).toEqual(Array.from(y.slice(0, 2000)))
    const pitch = new Map<number, number>()
    for (const u of utterances) {
      expect(pitch.get(u.speaker) ?? u.pitch).toBe(u.pitch)
      pitch.set(u.speaker, u.pitch)
    }
  })
})
