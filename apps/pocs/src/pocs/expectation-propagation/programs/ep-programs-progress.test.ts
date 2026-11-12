// Every EP program page runs on the shared run mechanism: its fit reports sweeps (done over the cap, the change against the
// tolerance) through the hooks, the progress never goes backwards, and a stop ends the fit where it stands with the sweeps
// done kept and a posterior that can still be read. The references (Gibbs, NUTS) report and stop the same way.
import { describe, expect, it } from 'vitest'
import { epHooks } from '../../../shared/ep/sweeps.ts'
import { type HostBase, type RunProgress, type RunState, startRun } from '../../../shared/run.ts'
import { infer as inferChange, simulate as simulateChange } from './ep-program-changepoint/program.ts'
import { gibbsReference as gibbsChange } from './ep-program-changepoint/reference.ts'
import { infer as inferExt, simulate as simulateExt } from './ep-program-changepoint-extended/program.ts'
import { runCircular } from './ep-program-circular/compute.ts'
import { infer as inferCox, simulate as simulateCox } from './ep-program-cox/program.ts'
import { nutsReference } from './ep-program-cox/reference.ts'
import { referenceSkills, runSkills, defaultSetup as skillsSetup } from './ep-program-skills/compute.ts'
import { runSparse, defaultSetup as sparseSetup } from './ep-program-sparse/compute.ts'
import { infer as inferTrack, simulate as simulateTrack } from './ep-program-tracker/program.ts'
import { runZip, defaultSetup as zipSetup } from './ep-program-zero-inflated/compute.ts'

const tick = { yieldToPage: () => Promise.resolve() }

/** Runs a task on the run mechanism; `stopAt` stops it when its progress has reached that many steps. */
async function drive<R>(
  task: (host: HostBase) => Promise<R | undefined>,
  stopAt = Number.POSITIVE_INFINITY,
): Promise<{ end: RunState<R>; seen: RunProgress[] }> {
  const seen: RunProgress[] = []
  let handle: ReturnType<typeof startRun> | undefined
  const end = await new Promise<RunState<R>>((resolve) => {
    handle = startRun<R>(
      (host) => task(host),
      (s) => {
        const p = s.progress
        if (p !== undefined && seen.at(-1) !== p) {
          seen.push(p)
          if (p.done >= stopAt) handle?.stop()
        }
        if (s.phase === 'done' || s.phase === 'converged' || s.phase === 'stopped' || s.phase === 'failed') resolve(s)
      },
      tick,
    )
  })
  return { end, seen }
}

/** The sweeps of one stage never go backwards and stay within the cap. */
function expectMonotone(seen: readonly RunProgress[], stage?: string): void {
  const own = seen.filter((p) => stage === undefined || p.stage === stage)
  expect(own.length).toBeGreaterThan(1)
  for (const [i, p] of own.entries()) {
    expect(p.done).toBeLessThanOrEqual(p.total)
    if (i > 0) expect(p.done).toBeGreaterThanOrEqual((own[i - 1] as RunProgress).done)
  }
}

describe('the EP fit reports its sweeps and can be stopped', () => {
  const cases: readonly {
    readonly name: string
    readonly task: () => (host: HostBase) => Promise<{ run: { status: unknown; t: number } }>
  }[] = [
    {
      name: 'cox',
      task: () => async (host) => {
        const sim = simulateCox(1, { amplitude: 0.8, lengthScale: 1.5 })
        return inferCox(sim.counts, sim.prior, epHooks(host, 'EP'))
      },
    },
    {
      name: 'changepoint',
      task: () => async (host) => {
        const sim = simulateChange(1, 0.95, 40)
        return inferChange(sim.y, 0.95, epHooks(host, 'EP'))
      },
    },
    {
      name: 'changepoint-extended',
      task: () => async (host) => {
        const sim = simulateExt(1, 0.93, 40)
        return inferExt(sim.y, 0.5, epHooks(host, 'EP'))
      },
    },
    {
      name: 'tracker',
      task: () => async (host) => {
        const sim = simulateTrack(1, { sd: 0.5, outliers: 0.1 }, 40)
        return inferTrack(sim.y, { sd: 0.5, outliers: 0.1 }, true, epHooks(host, 'EP'))
      },
    },
    {
      name: 'zero-inflated',
      task: () => async (host) => (await runZip({ ...zipSetup, n: 40 } as typeof zipSetup, epHooks(host, 'EP'))).fit,
    },
    {
      name: 'skills',
      task: () => async (host) => (await runSkills(skillsSetup, epHooks(host, 'EP'))).fit,
    },
  ]

  it.each(cases)(
    '$name: sweeps reported against the cap, and the run ends done or converged',
    async ({ task }) => {
      const { end, seen } = await drive(task())
      expectMonotone(seen)
      expect(seen[0]?.stage).toBe('EP')
      expect(seen.some((p) => p.detail?.includes('tolerance'))).toBe(true)
      expect(['done', 'converged']).toContain(end.phase)
      expect(end.progress?.done).toBe(end.progress?.total)
      expect(end.value?.run.t).toBeGreaterThan(0)
    },
    60_000,
  )

  it.each(cases)(
    '$name: a stop after the first sweeps ends the fit early and keeps what was done',
    async ({ task }) => {
      const { end, seen } = await drive(task(), 2)
      expect(end.phase).toBe('stopped')
      expect(end.value).toBeDefined()
      const last = seen.at(-1) as RunProgress
      expect(end.value?.run.t).toBeLessThan(last.total)
      expect(end.value?.run.t).toBeGreaterThanOrEqual(2)
    },
    60_000,
  )
})

describe('the references report and stop', () => {
  it('the Gibbs reference reports its sweeps; a stop keeps the draws so far (it goes on to 100 draws, so it can be read)', async () => {
    const sim = simulateChange(1, 0.95, 40)
    const full = await drive((host) => gibbsChange(sim.y, 0.95, { sweeps: 600, burn: 100 }, epHooks(host, 'Gibbs')))
    expectMonotone(full.seen, 'Gibbs')
    expect(full.end.value?.sweeps).toBe(500)
    const stopped = await drive(
      (host) => gibbsChange(sim.y, 0.95, { sweeps: 600, burn: 100 }, epHooks(host, 'Gibbs')),
      5,
    )
    expect(stopped.end.phase).toBe('stopped')
    expect(stopped.end.value?.sweeps).toBeGreaterThanOrEqual(100)
    expect(stopped.end.value?.sweeps).toBeLessThan(500)
    expect(stopped.end.value?.rateMean.every(Number.isFinite)).toBe(true)
  }, 60_000)

  it('the NUTS reference of the Cox process reports iterations over warmup + draws, and a stop in warmup gives nothing', async () => {
    const sim = simulateCox(1, { amplitude: 0.8, lengthScale: 1.5 })
    const run = (host: HostBase) =>
      nutsReference(sim.counts, sim.prior, { chains: 1, warmup: 20, draws: 30 }, epHooks(host, 'NUTS', 'iteration'))
    const full = await drive(run)
    expectMonotone(full.seen, 'NUTS')
    expect(full.end.value?.mean.length).toBe(40)
    const stopped = await drive(run, 5)
    expect(stopped.end.phase).toBe('stopped')
    expect(stopped.end.value).toBeUndefined()
  }, 60_000)

  it('the NUTS reference of the tournament reports iterations too', async () => {
    const fit = await runSkills(skillsSetup)
    const { seen, end } = await drive((host) => referenceSkills(fit.data, epHooks(host, 'NUTS', 'iteration')), 3)
    expectMonotone(seen, 'NUTS')
    expect(end.phase).toBe('stopped')
  }, 60_000)
})

describe('the other fits on the shared run', () => {
  it('the sparse regression reports a stage per model', async () => {
    const { end, seen } = await drive((host) =>
      runSparse({ ...sparseSetup, rows: 30, features: 6, active: 2 }, (stage) => epHooks(host, stage)),
    )
    const stages = [...new Set(seen.map((p) => p.stage))]
    expect(stages).toEqual(['EP, Gaussian likelihood', 'EP, Student-t likelihood', 'EP, closed-form mixture prior'])
    expect(['done', 'converged']).toContain(end.phase)
  }, 60_000)

  it('the circular fit reports sweeps and a stop keeps the sweeps done, with the record of the run for the player', async () => {
    const setup = { seed: 1, truth: 'one', components: 1, draws: 60, variant: 'data' } as const
    const { end, seen } = await drive((host) => runCircular(setup, undefined, epHooks(host, 'EP')), 3)
    expectMonotone(seen)
    expect(end.phase).toBe('stopped')
    expect(end.value?.fit.run.t).toBeGreaterThanOrEqual(3)
    expect(end.value?.record).toBeDefined()
    expect(Number.isFinite(end.value?.found.sd.mean)).toBe(true)
  }, 60_000)
})
