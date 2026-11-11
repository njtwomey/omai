// The EP programs, as a suite: each simulates from a seed with known truth, runs expectation propagation on the model written as
// user code, and must recover the truth (the truth inside the 90% intervals at the stated rate, the error in posterior sds
// under the stated bound) over a few seeds. A program where EP is poor says so in its check and is held here to what it
// honestly achieves, with the reason.
import { describe, expect, it } from 'vitest'
import { agreement, passes, recovery } from '../shared/ep/recovery.ts'
import { infer as inferChange, simulate as simulateChange } from './ep-program-changepoint/program.ts'
import { gibbsReference } from './ep-program-changepoint/reference.ts'
import { infer as inferCox, simulate as simulateCox } from './ep-program-cox/program.ts'
import { nutsReference } from './ep-program-cox/reference.ts'
import { infer as inferTrack, simulate as simulateTrack } from './ep-program-tracker/program.ts'
import { gibbsReference as gibbsTrack } from './ep-program-tracker/reference.ts'

const SEEDS = [1, 2, 3, 4]

describe('ep-program-cox: a log-Gaussian Cox process', () => {
  it('recovers the intensity: the true f inside EP intervals and errors within posterior sds, on every seed', () => {
    for (const seed of SEEDS) {
      const sim = simulateCox(seed, { amplitude: 0.8, lengthScale: 1.5 })
      const { rows, run } = inferCox(sim.counts, sim.prior)
      expect(run.status, `seed ${seed}`).toBe('converged')
      const r = recovery(rows, sim.f)
      expect(passes(r), `seed ${seed}: ${JSON.stringify(r)}`).toBe(true)
    }
  }, 60_000)
  it('agrees with NUTS on the same model (whitened): means within half a posterior sd, spreads within 15%', () => {
    const sim = simulateCox(1, { amplitude: 0.8, lengthScale: 1.5 })
    const { rows } = inferCox(sim.counts, sim.prior)
    const a = agreement(rows, nutsReference(sim.counts, sim.prior, { chains: 2, warmup: 150, draws: 400 }))
    expect(a.meanZ).toBeLessThan(0.5)
    expect(a.sdRatioRange[0]).toBeGreaterThan(0.85)
    expect(a.sdRatioRange[1]).toBeLessThan(1.15)
  }, 120_000)
})

/** Mean over steps of the largest gap between EP's and Gibbs's probability of a regime, and the smallest ratio of EP's rate sd to Gibbs's. */
function regimeGap(seed: number): {
  gap: number
  accuracy: number
  converged: boolean
  status: string
  sdRatio: number
} {
  const sim = simulateChange(seed)
  const ep = inferChange(sim.y, sim.stay)
  const gibbs = gibbsReference(sim.y, sim.stay)
  const gap =
    ep.regimes.reduce(
      (s, p, t) => s + Math.max(...p.map((v, k) => Math.abs(v - ((gibbs.regimes[t] as number[])[k] as number)))),
      0,
    ) / ep.regimes.length
  const accuracy = ep.regimes.filter((p, t) => p.indexOf(Math.max(...p)) === sim.z[t]).length / ep.regimes.length
  const sdRatio = Math.min(...ep.rates.map((r, k) => r.sd / (gibbs.rateSd[k] as number)))
  return { gap, accuracy, converged: ep.run.status === 'converged', status: String(ep.run.status), sdRatio }
}

describe('ep-program-changepoint: a Poisson rate that switches between three regimes, rates unknown', () => {
  it('recovers the regimes and agrees with Gibbs when every regime is visited (seeds 1, 3, 6)', () => {
    for (const seed of [1, 3, 6]) {
      const r = regimeGap(seed)
      expect(r.converged, `seed ${seed}`).toBe(true)
      expect(r.accuracy, `seed ${seed}`).toBeGreaterThanOrEqual(0.9)
      expect(r.gap, `seed ${seed}`).toBeLessThan(0.05)
    }
  }, 60_000)

  it('is poor, and says so, when a regime is not visited (seeds 2, 5, 8): EP commits to one reading where Gibbs is spread, and no variant of the engine changes that', () => {
    // Plain EP, with its default damped restart. Measured against Gibbs: the mean regime-probability gap is 0.2 to 0.3 and the
    // rate sds of the barely visited regimes are 0.05 to 0.3 of the sampler's. Power EP and alpha-divergence (on a gate of
    // Poisson branches, the one form of this model they apply to), damping and the improper-message policies do not restore
    // them (see docs/probprog/ep-plan.md section 11), so this is held as the limit of plain EP, to be flipped when that changes.
    for (const seed of [2, 5, 8]) {
      const r = regimeGap(seed)
      expect(r.gap, `seed ${seed}`).toBeGreaterThan(0.1)
      expect(r.sdRatio, `seed ${seed}: the narrowest rate sd against Gibbs`).toBeLessThan(0.4)
    }
  }, 60_000)

  it('does not report a success it did not reach: seed 8 does not converge, and says so', () => {
    expect(regimeGap(8).status).not.toBe('converged')
  }, 60_000)
})

describe('ep-program-tracker: an AR(1) path seen through Gaussian noise with wild readings', () => {
  const noise = { sd: 0.5, outliers: 0.1 }
  it('recovers the path through the outliers, agrees with Gibbs, and does much better than plain Gaussian noise', () => {
    for (const seed of SEEDS) {
      const sim = simulateTrack(seed, noise)
      const robust = inferTrack(sim.y, noise)
      expect(robust.run.status, `seed ${seed}`).toBe('converged')
      const r = recovery(robust.rows, sim.x)
      expect(passes(r), `seed ${seed}: ${JSON.stringify(r)}`).toBe(true)
      const a = agreement(robust.rows, gibbsTrack(sim.y, noise, { sweeps: 1000, burn: 100 }))
      expect(a.meanZ, `seed ${seed}`).toBeLessThan(0.3)
      expect(a.sdRatioRange[0], `seed ${seed}`).toBeGreaterThan(0.8)
      expect(a.sdRatioRange[1], `seed ${seed}`).toBeLessThan(1.25)
      // The contrast: believing every reading is overconfident and wrong.
      if (sim.wild.filter(Boolean).length >= 9)
        expect(recovery(inferTrack(sim.y, noise, false).rows, sim.x).zrms).toBeGreaterThan(1.8)
    }
  }, 60_000)
})
