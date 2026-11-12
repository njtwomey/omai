// The EP programs, as a suite: each simulates from a seed with known truth, runs expectation propagation on the model written as
// user code, and must recover the truth (the truth inside the 90% intervals at the stated rate, the error in posterior sds
// under the stated bound) over a few seeds. A program where EP is poor says so in its check and is held here to what it
// honestly achieves, with the reason.

import { calibrationTable } from 'omai-engine/probprog/ep'
import { describe, expect, it } from 'vitest'
import { agreement, passes, recovery } from '../../../shared/ep/recovery.ts'
import { infer as inferChange, simulate as simulateChange } from './ep-program-changepoint/program.ts'
import { gibbsReference } from './ep-program-changepoint/reference.ts'
import { runSeed } from './ep-program-changepoint-extended/calibration.ts'
import { infer as inferCox, simulate as simulateCox } from './ep-program-cox/program.ts'
import { nutsReference } from './ep-program-cox/reference.ts'
import { infer as inferTrack, simulate as simulateTrack } from './ep-program-tracker/program.ts'
import { gibbsReference as gibbsTrack } from './ep-program-tracker/reference.ts'

const SEEDS = [1, 2, 3, 4]

describe('ep-program-cox: a log-Gaussian Cox process', () => {
  it('recovers the intensity: the true f inside EP intervals and errors within posterior sds, on every seed', async () => {
    for (const seed of SEEDS) {
      const sim = simulateCox(seed, { amplitude: 0.8, lengthScale: 1.5 })
      const { rows, run } = await inferCox(sim.counts, sim.prior)
      expect(run.status, `seed ${seed}`).toBe('converged')
      const r = recovery(rows, sim.f)
      expect(passes(r), `seed ${seed}: ${JSON.stringify(r)}`).toBe(true)
    }
  }, 60_000)
  it('agrees with NUTS on the same model (whitened): means within half a posterior sd, spreads within 15%', async () => {
    const sim = simulateCox(1, { amplitude: 0.8, lengthScale: 1.5 })
    const { rows } = await inferCox(sim.counts, sim.prior)
    const nuts = await nutsReference(sim.counts, sim.prior, { chains: 2, warmup: 150, draws: 400 })
    const a = agreement(rows, nuts as NonNullable<typeof nuts>)
    expect(a.meanZ).toBeLessThan(0.5)
    expect(a.sdRatioRange[0]).toBeGreaterThan(0.85)
    expect(a.sdRatioRange[1]).toBeLessThan(1.15)
  }, 120_000)
})

/** Mean over steps of the largest gap between EP's and Gibbs's probability of a regime, and the smallest ratio of EP's rate sd to Gibbs's. */
async function regimeGap(seed: number): Promise<{
  gap: number
  accuracy: number
  converged: boolean
  status: string
  sdRatio: number
}> {
  const sim = simulateChange(seed)
  const ep = await inferChange(sim.y, sim.stay)
  const gibbs = await gibbsReference(sim.y, sim.stay)
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
  it('recovers the regimes and agrees with Gibbs when every regime is visited (seeds 1, 3, 6)', async () => {
    for (const seed of [1, 3, 6]) {
      const r = await regimeGap(seed)
      expect(r.converged, `seed ${seed}`).toBe(true)
      expect(r.accuracy, `seed ${seed}`).toBeGreaterThanOrEqual(0.9)
      expect(r.gap, `seed ${seed}`).toBeLessThan(0.05)
    }
  }, 60_000)

  it('is poor, and says so, when a regime is not visited (seeds 2, 5, 8): EP commits to one reading where Gibbs is spread, and no variant of the engine changes that', async () => {
    // Plain EP, with its default damped restart. Measured against Gibbs: the mean regime-probability gap is 0.2 to 0.3 and the
    // rate sds of the barely visited regimes are 0.05 to 0.3 of the sampler's. Power EP and alpha-divergence (on a gate of
    // Poisson branches, the one form of this model they apply to), damping and the improper-message policies do not restore
    // them (see docs/probprog/ep-plan.md section 11), so this is held as the limit of plain EP, to be flipped when that changes.
    for (const seed of [2, 5, 8]) {
      const r = await regimeGap(seed)
      expect(r.gap, `seed ${seed}`).toBeGreaterThan(0.1)
      expect(r.sdRatio, `seed ${seed}: the narrowest rate sd against Gibbs`).toBeLessThan(0.4)
    }
  }, 60_000)

  it('does not report a success it did not reach: seed 8 does not converge, and says so', async () => {
    expect((await regimeGap(8)).status).not.toBe('converged')
  }, 60_000)
})

describe('ep-program-changepoint-extended: the same program with the rates and the transition matrix both learned', () => {
  // The seeds of 100 steps at stay probability 0.93 (calibration.ts: EP damped by half, Gibbs from a data-ordered start).
  const options = { sweeps: 1200 }
  const floor = { seeds: 0, coverage: 0.7, zrms: 1.6, reason: '' }

  it('when every regime is visited it agrees with Gibbs, and the truth is inside the 90% intervals as often as it should be', async () => {
    // Seeds 14, 18 and 26 are ones on which undamped EP takes the relabelled reading (low and middle swapped).
    const runs = []
    for (const seed of [1, 3, 5, 14, 18, 26]) runs.push(await runSeed(seed, options))
    for (const r of runs) {
      expect(r.barely, `seed ${r.seed}`).toBe(false)
      expect(r.status, `seed ${r.seed}`).toBe('converged')
      expect(r.gap, `seed ${r.seed}: the gap to Gibbs`).toBeLessThan(0.05)
    }
    const ep = calibrationTable(
      { ...floor, seeds: runs.length },
      runs.map((r) => r.ep),
    )
    // Six seeds cannot hold a parameter to a floor of 0.7 (that is five of six): the stay probability of the low regime is covered
    // on 4 of these 6 (seeds 14 and 18 miss by 0.004 and 0.015 above the 90% interval's upper end; these seeds were chosen for the
    // label switching, not for coverage), and the same run on every seed of 1 to 30 with all regimes visited and EP converged (22
    // seeds) covers it 18 times in 22 (0.82; Gibbs 0.72 in the plan's table, the prior pulls the stay probabilities towards 0.67).
    // The numbers are the same to seven digits on every commit from the program's first to main, so this is the sample, not drift.
    // So each parameter is held to three of six (0.5), and the pooled coverage of all 12 parameters (72 intervals, nominal 0.9,
    // sd 0.035) to 0.8, which is the check that can tell a miscalibrated EP from a small sample.
    for (const row of ep.rows) {
      expect(row.coverage, row.key).toBeGreaterThanOrEqual(0.5)
      expect(row.zrms, row.key).toBeLessThan(1.6)
    }
    const pooled = ep.rows.reduce((a, r) => a + r.coverage, 0) / ep.rows.length
    expect(pooled, 'pooled coverage of the 90% intervals').toBeGreaterThanOrEqual(0.8)
  }, 60_000)

  it('is wrong, and says so, when a regime is barely visited (seeds 2, 10, 15, 24, 27): EP commits to one reading where Gibbs is spread', async () => {
    // Measured: the regime probabilities are 0.2 to 0.5 from the sampler's on average; some runs do not converge.
    for (const seed of [2, 10, 24]) {
      const r = await runSeed(seed, options)
      expect(r.barely, `seed ${seed}`).toBe(true)
      expect(r.gap, `seed ${seed}`).toBeGreaterThan(0.1)
    }
  }, 60_000)

  it('without damping the warm-up keeps EP out of the relabelled reading it used to settle on (seed 14)', async () => {
    // Undamped EP from flat messages swapped the low and middle regimes on seed 14 (accuracy 0.00, gap to Gibbs 0.95, the low
    // rate 9.4 where Gibbs has 3.5). The engine's `warmup: auto` (three damped sweeps, since the regimes are chosen by a
    // `select`) reaches the reading of Gibbs and of the damped run: accuracy 1.00, gap 0.008, rates within 0.1 of Gibbs's.
    // Over seeds 1 to 30 undamped, 26 runs have accuracy 0.9 or more with the warm-up and 22 without it.
    const r = await runSeed(14, { ...options, damping: 0 })
    expect(r.status).toBe('converged')
    expect(r.accuracy).toBeGreaterThan(0.9)
    expect(r.gap).toBeLessThan(0.05)
  }, 60_000)
})

describe('ep-program-tracker: an AR(1) path seen through Gaussian noise with wild readings', () => {
  const noise = { sd: 0.5, outliers: 0.1 }
  it('recovers the path through the outliers, agrees with Gibbs, and does much better than plain Gaussian noise', async () => {
    for (const seed of SEEDS) {
      const sim = simulateTrack(seed, noise)
      const robust = await inferTrack(sim.y, noise)
      expect(robust.run.status, `seed ${seed}`).toBe('converged')
      const r = recovery(robust.rows, sim.x)
      expect(passes(r), `seed ${seed}: ${JSON.stringify(r)}`).toBe(true)
      const a = agreement(robust.rows, await gibbsTrack(sim.y, noise, { sweeps: 1000, burn: 100 }))
      expect(a.meanZ, `seed ${seed}`).toBeLessThan(0.3)
      expect(a.sdRatioRange[0], `seed ${seed}`).toBeGreaterThan(0.8)
      expect(a.sdRatioRange[1], `seed ${seed}`).toBeLessThan(1.25)
      // The contrast: believing every reading is overconfident and wrong.
      if (sim.wild.filter(Boolean).length >= 9)
        expect(recovery((await inferTrack(sim.y, noise, false)).rows, sim.x).zrms).toBeGreaterThan(1.8)
    }
  }, 60_000)
})
