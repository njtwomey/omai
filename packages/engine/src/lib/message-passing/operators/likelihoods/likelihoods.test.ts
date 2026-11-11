import { describe, expect, it } from 'vitest'
import { messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import { likelihoodRowOperator, likelihoodSiteOperator, likelihoods } from './likelihoods.ts'

/** The mean, variance and log mass of N(m, v) f(x) on a fine uniform grid: the reference every rule is checked against. */
function brute(m: number, v: number, logf: (x: number) => number): { mean: number; variance: number; logMass: number } {
  const sd = Math.sqrt(v)
  const n = 150000
  const lo = m - 12 * sd
  const h = (24 * sd) / n
  let top = Number.NEGATIVE_INFINITY
  const lw: number[] = []
  for (let i = 0; i <= n; i++) {
    const x = lo + i * h
    const l = -0.5 * ((x - m) / sd) ** 2 + logf(x)
    lw.push(l)
    if (l > top) top = l
  }
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + i * h
    const w = Math.exp((lw[i] as number) - top)
    z += w
    m1 += w * x
    m2 += w * x * x
  }
  const mean = m1 / z
  return { mean, variance: m2 / z - mean * mean, logMass: top + Math.log((z * h) / (sd * Math.sqrt(2 * Math.PI))) }
}

/** The moments of the tilted Gaussian a site message implies against its cavity N(m, v). */
function tiltedOf(site: readonly number[], m: number, v: number): { mean: number; variance: number } {
  const j = 1 / v + (site[1] ?? 0)
  return { mean: (m / v + (site[0] ?? 0)) / j, variance: 1 / j }
}

const cases: readonly { name: string; y: number; params: Record<string, number | readonly number[]> }[] = [
  { name: 'poisson-log', y: 4, params: { exposure: 2.5 } },
  { name: 'poisson-log', y: 0, params: { exposure: 1 } },
  { name: 'poisson-log-truncated', y: 1, params: { exposure: 1 } },
  { name: 'poisson-log-truncated', y: 5, params: { exposure: 2 } },
  { name: 'negative-binomial-log', y: 7, params: { dispersion: 0.4, exposure: 1.5 } },
  { name: 'negative-binomial-log', y: 0, params: { dispersion: 1.2, exposure: 1 } },
  { name: 'weibull-log', y: 2.2, params: { shape: 1.7 } },
  { name: 'weibull-log', y: 1.1, params: { shape: 2.4, censored: 1 } },
  { name: 'log-normal', y: 3.5, params: { sigma: 0.6 } },
  { name: 'log-normal', y: 3.5, params: { sigma: 0.6, censored: 1 } },
  { name: 'binomial-logit', y: 3, params: { n: 10 } },
  { name: 'binomial-logit', y: 0, params: { n: 6 } },
  { name: 'binomial-probit', y: 8, params: { n: 9 } },
  { name: 'ordinal-probit', y: 1, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'ordinal-probit', y: 3, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'ordinal-probit', y: 0, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'ordinal-logit', y: 1, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'ordinal-logit', y: 3, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'ordinal-logit', y: 0, params: { cuts: [-0.5, 0.7, 1.9] } },
  { name: 'asymmetric-laplace', y: 1.3, params: { scale: 0.5, tau: 0.8 } },
  { name: 'asymmetric-laplace', y: -0.4, params: { scale: 0.3, tau: 0.15 } },
  { name: 'student-t', y: 1.3, params: { scale: 0.6, df: 3 } },
  { name: 'laplace', y: 1.3, params: { scale: 0.6 } },
]
// A cavity narrower than, comparable to, much wider than the likelihood, and one far from where it lives.
const cavities: readonly (readonly [number, number])[] = [
  [0, 1],
  [1.2, 0.05],
  [0, 100],
  [-2, 4],
  [3, 0.3],
]

describe('the registry of numeric likelihoods', () => {
  for (const c of cases)
    it(`${c.name} (y = ${c.y}, ${JSON.stringify(c.params)}): tilted moments and evidence match a fine grid`, () => {
      const table = likelihoods()
      const logf = (table[c.name] as NonNullable<(typeof table)[string]>).density(c.y, c.params as never)
      for (const [m, v] of cavities) {
        const args = [messageArg(gaussian, gaussianEta(m, v))]
        const out = likelihoodSiteOperator.ep?.({
          mode: 'ep',
          to: 0,
          args,
          params: { likelihood: c.name, y: c.y, ...c.params },
          power: 1,
        })
        const t = tiltedOf((out as { message: number[] }).message, m, v)
        const ref = brute(m, v, logf)
        const sd = Math.sqrt(ref.variance)
        expect(Math.abs(t.mean - ref.mean) / sd, `mean at N(${m}, ${v})`).toBeLessThan(2e-3)
        expect(t.variance / ref.variance, `variance at N(${m}, ${v})`).toBeCloseTo(1, 2)
        const ev = likelihoodSiteOperator.logEvidence?.({
          mode: 'ep',
          to: 0,
          args,
          params: { likelihood: c.name, y: c.y, ...c.params },
          power: 1,
        }) as number
        expect(ev, `evidence at N(${m}, ${v})`).toBeCloseTo(gaussian.logPartition(gaussianEta(m, v)) + ref.logMass, 3)
      }
    })

  it('the densities normalise: Poisson and negative binomial over the counts, binomial over the successes, ordinal over the categories', () => {
    const table = likelihoods()
    const u = 0.37
    const total = (name: string, ys: readonly number[], params: Record<string, number | readonly number[]>): number =>
      ys.reduce(
        (a, y) => a + Math.exp((table[name] as NonNullable<(typeof table)[string]>).density(y, params as never)(u)),
        0,
      )
    const counts = Array.from({ length: 400 }, (_, i) => i)
    expect(total('poisson-log', counts, { exposure: 1.7 })).toBeCloseTo(1, 10)
    expect(total('poisson-log-truncated', counts.slice(1), { exposure: 1.7 })).toBeCloseTo(1, 10)
    expect(total('negative-binomial-log', counts, { dispersion: 0.6, exposure: 1.7 })).toBeCloseTo(1, 8)
    expect(total('binomial-logit', [0, 1, 2, 3, 4, 5, 6, 7], { n: 7 })).toBeCloseTo(1, 12)
    expect(total('binomial-probit', [0, 1, 2, 3, 4, 5, 6, 7], { n: 7 })).toBeCloseTo(1, 12)
    expect(total('ordinal-probit', [0, 1, 2, 3], { cuts: [-0.5, 0.7, 1.9] })).toBeCloseTo(1, 12)
    expect(total('ordinal-logit', [0, 1, 2, 3], { cuts: [-0.5, 0.7, 1.9] })).toBeCloseTo(1, 12)
  })

  it('the Weibull and log-normal densities are the continuous densities of the time, and the censored form is their survival', () => {
    const table = likelihoods()
    const u = 0.4
    const w = table['weibull-log']
    const ln = table['log-normal']
    // The integral of the density over the time is one; the survival at y is the integral above it.
    const integrate = (
      name: 'weibull-log' | 'log-normal',
      params: Record<string, number>,
      a: number,
      b: number,
    ): number => {
      const l = table[name]
      const n = 200000
      const h = (b - a) / n
      let s = 0
      for (let i = 0; i < n; i++)
        s += Math.exp((l as NonNullable<typeof l>).density(a + (i + 0.5) * h, params as never)(u)) * h
      return s
    }
    expect(w).toBeDefined()
    expect(ln).toBeDefined()
    expect(integrate('weibull-log', { shape: 1.6 }, 1e-9, 60)).toBeCloseTo(1, 4)
    expect(integrate('log-normal', { sigma: 0.7 }, 1e-9, 400)).toBeCloseTo(1, 3)
    const yc = 1.9
    const surv = Math.exp((w as NonNullable<typeof w>).density(yc, { shape: 1.6, censored: 1 })(u))
    expect(surv).toBeCloseTo(integrate('weibull-log', { shape: 1.6 }, yc, 80), 4)
    const survLn = Math.exp((ln as NonNullable<typeof ln>).density(yc, { sigma: 0.7, censored: 1 })(u))
    expect(survLn).toBeCloseTo(integrate('log-normal', { sigma: 0.7 }, yc, 600), 3)
  })

  it('a power scales the log density, and the message is divided by it (fractional EP)', () => {
    const args = [messageArg(gaussian, gaussianEta(0, 1))]
    const params = { likelihood: 'poisson-log', y: 3, exposure: 1 }
    const half = likelihoodSiteOperator.ep?.({ mode: 'ep', to: 0, args, params, power: 0.5 })
    const t = tiltedOf(
      ((half as { message: number[] }).message as number[]).map((v) => v * 0.5),
      0,
      1,
    )
    const ref = brute(0, 1, (x) => 0.5 * (likelihoods()['poisson-log']?.density(3, params as never)(x) ?? 0))
    expect(t.mean).toBeCloseTo(ref.mean, 3)
  })

  it('the row is the site on the predictor a . w, as a rank-one message on the weights', () => {
    const w = gaussianMv(2)
    const cavity = gaussianMvEta([0.5, -0.2], [2, 0.3, 0.3, 1])
    const a = [1, 0]
    const params = { likelihood: 'negative-binomial-log', y: 3, dispersion: 0.5, exposure: 1, a }
    const out = likelihoodRowOperator.ep?.({ mode: 'ep', to: 0, args: [messageArg(w, cavity)], params, power: 1 })
    expect(out?.message.length).toBe(6)
    expect(out?.message[1]).toBeCloseTo(0, 12)
    expect(out?.message[2]).not.toBe(0)
  })

  it('an invalid outcome or parameter is reported by check, not by a NaN', () => {
    const bad = likelihoodSiteOperator.check?.({ likelihood: 'poisson-log', y: -1, exposure: 1 }, [])
    expect(bad).toMatch(/non-negative integer/)
    expect(likelihoodSiteOperator.check?.({ likelihood: 'nope', y: 1 }, [])).toMatch(/unknown likelihood/)
    expect(likelihoodSiteOperator.check?.({ likelihood: 'poisson-log', y: 2, exposure: 1 }, [])).toBeUndefined()
  })
})
