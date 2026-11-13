// The EP programs of the second batch (sparse regression, zero-inflated counts, tournament skills): each is simulated from a
// known truth over a few seeds, run by EP and held to its recovery check (truth recovered within the stated tolerances, EP
// against the exact reference where there is one). A failing row names what EP got wrong.
import { epSpecMethod } from 'omai-engine/probprog/inference'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import type { ProgramCheck } from '../../../shared/ep/program-check.ts'
import { passes } from '../../../shared/ep/program-check.ts'
import * as skills from './ep-program-skills/compute.ts'
import { checkSparse, defaultSetup, exactSparse, runSparse } from './ep-program-sparse/compute.ts'
import { simulate as simulateSparse, sparseModel } from './ep-program-sparse/program.ts'
import * as zip from './ep-program-zero-inflated/compute.ts'

interface Program {
  readonly name: string
  readonly seeds: readonly number[]
  readonly check: (seed: number) => Promise<ProgramCheck>
}

const programs: readonly Program[] = [
  {
    name: 'sparse regression with outliers and unknown noise',
    seeds: [1, 2, 3],
    check: async (seed) => {
      const run = await runSparse({ ...defaultSetup, seed })
      return checkSparse(run, exactSparse({ features: run.setup.features, ...run.data }))
    },
  },
  {
    name: 'counts with excess zeros',
    seeds: [1, 2, 3],
    check: async (seed) => {
      const run = await zip.runZip({ ...zip.defaultSetup, seed })
      return zip.checkZip(run, zip.exactZip(run.data))
    },
  },
  {
    name: 'who is better: a tournament with unknown skills',
    seeds: [1, 2, 3],
    check: async (seed) => {
      const run = await skills.runSkills({ ...skills.defaultSetup, seed })
      return skills.checkSkills(run, (await skills.referenceSkills(run.data)) as skills.SkillsReference)
    },
  },
]

describe('the tournament skills, pooled over seeds', () => {
  it('EP\u2019s 90% intervals cover at least 80% of the true skills', async () => {
    let inside = 0
    let total = 0
    for (const seed of [1, 2, 3]) {
      const run = await skills.runSkills({ ...skills.defaultSetup, seed })
      inside += skills.coverageOf(run.data.skills, run.found) * run.data.skills.length
      total += run.data.skills.length
    }
    expect(inside / total).toBeGreaterThanOrEqual(0.8)
  }, 120_000)
})

describe.each(programs)('$name', (program) => {
  it.each(program.seeds)(
    'recovers the truth (seed %i)',
    async (seed) => {
      const check = await program.check(seed)
      const failed = check.rows
        .filter((r) => r.gating !== false && !r.pass)
        .map((r) => `${r.label}: ${r.value} (${r.tolerance})`)
      expect(failed).toEqual([])
      expect(passes(check)).toBe(true)
    },
    120_000,
  )
})

describe('the sparse regression with outliers: a run that does not move because it cannot update is not a success', () => {
  // Noise sd 1.25, seed 17: undamped EP collapses at the first sweeps (the gamma noise precision goes to 6e-6), some row factors
  // then have an improper cavity for the precision and are skipped for ever, every inclusion probability stays at the prior 0.2,
  // and the beliefs stop moving. It used to be reported `converged`. (Noise sd 1, seed 3, where this was first seen, no longer
  // collapses: the shared window and zoomed grid of the gamma tilt keep the precision from going to zero, see below.)
  const fit = (noise: number, seed: number, restarts: number) => {
    const data = simulateSparse({ ...defaultSetup, noise, seed })
    const method = epSpecMethod(sparseModel('gaussian', noise).bind({ X: data.X, y: data.y }))
    const s = session(method, {
      config: { tol: 1e-8, damping: 0, restarts },
      seed: 1,
      steps: 300,
      mode: 'eager',
      logs: 'all',
    })
    const posterior = method.posterior?.(s.state, s.config)
    if (posterior === undefined) throw new Error('EP gave no posterior')
    const pip = Array.from(toFlat(posterior.mean('z')))
    const active = data.truth.map((v) => v !== 0)
    return { s, posterior, pip, active, tau: Number(toFlat(posterior.mean('tau'))[0]) }
  }

  it('without a restart it is stopped as stalled, with the reason, and the degenerate fixed point is visible', () => {
    // Whether undamped EP collapses on these data is a knife edge: it did on the macOS (arm64) build of V8 and did not on the Linux
    // (x64) one of CI, where exp, log and sin differ from it in the last bit for about one input in ten and the run settles on a proper
    // fixed point. So the test holds the rule in both cases: a run that collapsed is stopped as stalled, never converged, and a run
    // that did not is converged with a precision that is not zero. (The verdict of a stall is tested on its own in
    // engine/verdict.test.ts.)
    const r = fit(1.25, 17, 0)
    if (r.tau < 1e-3) {
      expect(r.s.status).toBe('stopped')
      expect(r.posterior.status).toBe('stopped')
      expect(String(r.posterior.diagnostics['reason'])).toMatch(/^stalled: \d+ message updates could not be computed/)
      expect(Math.max(...r.pip) - Math.min(...r.pip)).toBeLessThan(0.01)
    } else {
      expect(r.s.status).toBe('converged')
      expect(r.posterior.status).toBe('converged')
    }
  })

  it('with the default damped restart it recovers: the four active coefficients are found and the noise precision is sane', () => {
    // Seed 20 stalls without a restart too (13 updates dropped, the inclusions spread), and the restart reaches a fixed point.
    const r = fit(1.25, 20, 1)
    expect(r.s.status).toBe('converged')
    expect(r.s.state.restarts).toBe(1)
    expect(r.tau).toBeGreaterThan(0.04)
    for (const [j, on] of r.active.entries())
      if (on) expect(r.pip[j], `active coefficient ${j}`).toBeGreaterThan(0.8)
      else expect(r.pip[j], `inactive coefficient ${j}`).toBeLessThan(0.25)
  }, 60_000)

  it('noise sd 1, seed 3 no longer collapses: undamped EP converges at once, with no restart, to the right coefficients', () => {
    // Before the gamma tilt used one window and zoomed grid (agent-a9fac1-numerics) this run stopped as stalled, the noise precision
    // at 2e-5 and every inclusion at 0.2, and needed the restart. Now: 23 sweeps, precision 0.099 (the outliers inflate the noise
    // of a Gaussian likelihood), the four active inclusions 1.00, 0.95, 1.00 and 1.00, the eight others at most 0.05.
    const r = fit(1, 3, 0)
    expect(r.s.status).toBe('converged')
    expect(r.s.state.restarts).toBe(0)
    expect(r.tau).toBeGreaterThan(0.05)
    for (const [j, on] of r.active.entries())
      expect(on ? r.pip[j] : 1 - (r.pip[j] as number), `coefficient ${j}`).toBeGreaterThan(0.8)
  }, 60_000)
})
