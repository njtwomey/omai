import { describe, expect, it } from 'vitest'
import { gaussian, gaussianEta } from '../families.ts'
import { gaussianMv, gaussianMvEta } from '../families-vector.ts'
import { messageArg } from './args.ts'
import { laplaceRow, laplaceSite, mixtureRow, mixtureSite, studentTRow, studentTSite } from './robust.ts'

/** The mean, variance and log mass of N(m, v) f(x) on a fine uniform grid: the reference every rule is checked against. */
function brute(m: number, v: number, logf: (x: number) => number): { mean: number; variance: number; logMass: number } {
  const sd = Math.sqrt(v)
  const n = 400000
  const lo = m - 12 * sd
  const h = (24 * sd) / n
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + i * h
    const w = Math.exp(-0.5 * ((x - m) / sd) ** 2 + logf(x)) / (sd * Math.sqrt(2 * Math.PI))
    z += w
    m1 += w * x
    m2 += w * x * x
  }
  const mean = m1 / z
  return { mean, variance: m2 / z - mean * mean, logMass: Math.log(z * h) }
}
/** The site message to the variable a cavity N(m, v) gets from a rule, as the moments of the tilted Gaussian it implies. */
function tiltedOf(site: { message: number[] }, m: number, v: number): { mean: number; variance: number } {
  const j = 1 / v + (site.message[1] ?? 0)
  return { mean: (m / v + (site.message[0] ?? 0)) / j, variance: 1 / j }
}

describe('the robust sites', () => {
  it('the mixture site matches the moments of the Gaussian cavity times the mixture, exactly', () => {
    const params = { weights: [0.3, 0.7], means: [1, -1], sds: [2, 0.3] }
    for (const [m, v] of [
      [0, 1],
      [2, 0.5],
      [-0.7, 9],
    ] as const) {
      const out = mixtureSite.ep?.({
        mode: 'ep',
        to: 0,
        args: [messageArg(gaussian, gaussianEta(m, v))],
        params,
        power: 1,
      })
      const t = tiltedOf(out as { message: number[] }, m, v)
      const ref = brute(m, v, (x) =>
        Math.log((0.3 * Math.exp(-0.5 * ((x - 1) / 2) ** 2)) / 2 + (0.7 * Math.exp(-0.5 * ((x + 1) / 0.3) ** 2)) / 0.3),
      )
      expect(t.mean).toBeCloseTo(ref.mean, 5)
      expect(t.variance).toBeCloseTo(ref.variance, 5)
    }
  })

  it('the Student-t and Laplace sites match a fine grid, including against a cavity much wider than the likelihood', () => {
    const lgam = (z: number): number => {
      // log gamma by the Lanczos series (an independent implementation for the reference)
      const g = 7
      const c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
      ]
      const x = z - 1
      let a = c[0] as number
      const t = x + g + 0.5
      for (let i = 1; i < 9; i++) a += (c[i] as number) / (x + i)
      return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
    }
    const y = 1.3
    const s = 0.6
    const df = 3
    const logt = (x: number): number =>
      lgam((df + 1) / 2) -
      lgam(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      Math.log(s) -
      ((df + 1) / 2) * Math.log(1 + ((y - x) / s) ** 2 / df)
    const logl = (x: number): number => -Math.log(2 * s) - Math.abs(y - x) / s
    for (const [m, v] of [
      [0, 1],
      [3, 0.2],
      [0, 100],
      [-1, 4],
    ] as const) {
      const args = [messageArg(gaussian, gaussianEta(m, v))]
      const a = tiltedOf(
        studentTSite.ep?.({ mode: 'ep', to: 0, args, params: { y, scale: s, df }, power: 1 }) as never,
        m,
        v,
      )
      const ra = brute(m, v, logt)
      expect(a.mean).toBeCloseTo(ra.mean, 3)
      expect(a.variance / ra.variance).toBeCloseTo(1, 3)
      const b = tiltedOf(
        laplaceSite.ep?.({ mode: 'ep', to: 0, args, params: { y, scale: s }, power: 1 }) as never,
        m,
        v,
      )
      const rb = brute(m, v, logl)
      expect(b.mean).toBeCloseTo(rb.mean, 3)
      expect(b.variance / rb.variance).toBeCloseTo(1, 3)
    }
    // The evidence is the log mass of the cavity (its normaliser) plus the log of the integral.
    const args = [messageArg(gaussian, gaussianEta(0, 1))]
    const ev = studentTSite.logEvidence?.({ mode: 'ep', to: 0, args, params: { y, scale: s, df }, power: 1 }) as number
    expect(ev).toBeCloseTo(gaussian.logPartition(gaussianEta(0, 1)) + brute(0, 1, logt).logMass, 4)
  })

  it('the rows are the same factors on a weight vector, as a rank-one message', () => {
    const w = gaussianMv(2)
    const cavity = gaussianMvEta([0.5, -0.2], [2, 0.3, 0.3, 1])
    const a = [1, 0]
    for (const [row, params] of [
      [mixtureRow, { weights: [0.5, 0.5], means: [0, 0], sds: [3, 0.1] }],
      [studentTRow, { y: 1, scale: 0.5, df: 4 }],
      [laplaceRow, { y: 1, scale: 0.5 }],
    ] as const) {
      const out = row.ep?.({ mode: 'ep', to: 0, args: [messageArg(w, cavity)], params: { ...params, a }, power: 1 })
      expect(out?.message.length).toBe(6)
      // With a = e_1 the message touches only the first weight: the second row and column are zero.
      expect(out?.message[1]).toBeCloseTo(0, 12)
      expect(out?.message[4]).toBeCloseTo(0, 12)
      expect(out?.message[2]).not.toBe(0)
    }
  })
})
