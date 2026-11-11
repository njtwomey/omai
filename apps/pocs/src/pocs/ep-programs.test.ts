// The EP programs, as a suite: each simulates from a seed with known truth, runs expectation propagation on the model written as
// user code, and must recover the truth (the truth inside the 90% intervals at the stated rate, the error in posterior sds
// under the stated bound) over a few seeds. A program where EP is poor says so in its check and is held here to what it
// honestly achieves, with the reason.
import { describe, expect, it } from 'vitest'
import { agreement, passes, recovery } from '../shared/ep/recovery.ts'
import { infer as inferCox, simulate as simulateCox } from './ep-program-cox/program.ts'
import { nutsReference } from './ep-program-cox/reference.ts'

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
