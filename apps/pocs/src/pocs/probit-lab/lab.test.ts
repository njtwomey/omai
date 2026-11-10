// The probit lab's compute: every method on offer runs on the probit model through the same code a worker runs, within a
// small budget, and the results hold what the page draws.
import { describe, expect, it } from 'vitest'
import { runLab } from './lab.ts'
import { METHODS } from './methods.ts'
import { type PoolState, poolSize, runPool } from './pool.ts'
import { DATASETS, makeProblem } from './problem.ts'
import { chainsAt, frameAt, gridAt, marginalsAt, visibleRows } from './shown.ts'
import type { LabRequest, MethodResult } from './types.ts'

const host = (stopAfterMs?: number): Parameters<typeof runLab>[1] => {
  const t0 = performance.now()
  return {
    now: () => performance.now(),
    yieldToPage: () => new Promise((resolve) => setTimeout(resolve, 0)),
    stopped: () => stopAfterMs !== undefined && performance.now() - t0 > stopAfterMs,
    status: () => undefined,
  }
}

const problem = makeProblem({ dataset: 'moons', rows: 60, noise: 0.2, seed: 1, degree: 2, priorScale: 1 })
const request = (method: string, budgetMs = 700): LabRequest => ({ problem, method, budgetMs, seed: 1, gridSize: 8 })

describe('the problem', () => {
  it('makes every dataset on offer, binary, split 70/30, with a rectangle around the points', () => {
    for (const d of DATASETS) {
      const p = makeProblem({ dataset: d.id, rows: 50, noise: d.noise[2], seed: 2, degree: 2, priorScale: 1 })
      expect(p.train.y.length + p.test.y.length).toBe(50)
      expect(p.test.y).toHaveLength(15)
      expect(new Set([...p.train.y, ...p.test.y])).toEqual(new Set([0, 1]))
      expect(p.extent.x[0]).toBeLessThan(Math.min(...p.train.x.map((r) => r[0] ?? 0)))
    }
  })

  it('is the same for the same seed', () => {
    expect(makeProblem(problem.settings)).toEqual(problem)
  })
})

describe('the methods on offer', () => {
  const results = new Map<string, MethodResult>()
  for (const m of METHODS)
    it(`${m.label} runs on the probit model and returns frames, a pool, a grid and scores`, async () => {
      const r = await runLab(request(m.id, m.organisation === 'chain' ? 1500 : 500), host())
      results.set(m.id, r)
      expect(r.id).toBe(m.id)
      expect(r.names).toHaveLength(6)
      expect(r.frames.length).toBeGreaterThan(0)
      expect(r.grids).toHaveLength(r.frames.length * 64)
      expect(r.pool.values.length).toBe(r.pool.at.length * 6)
      expect(r.steps).toBeGreaterThan(0)
      if (m.organisation === 'iterate') {
        expect(r.pool.at.length).toBeGreaterThan(0)
        expect(r.test.accuracy).toBeGreaterThan(0.5)
        expect(r.coefficients[0]?.sd).toBeGreaterThanOrEqual(0)
      }
      if (m.curve !== undefined && r.steps > 3) expect(r.curve?.values.length).toBeGreaterThan(0)
    })

  it('the draws of a sampler appear over time and stay; those of an approximation are shown one snapshot at a time', () => {
    const nuts = results.get('hmc') ?? results.get('mala')
    if (nuts !== undefined && nuts.pool.at.length > 0) {
      const early = visibleRows(nuts, 0).length
      const late = visibleRows(nuts, Number.POSITIVE_INFINITY).length
      expect(late).toBe(nuts.pool.at.length)
      expect(early).toBeLessThanOrEqual(late)
    }
    const svi = results.get('svi-mf')
    if (svi !== undefined) {
      expect(visibleRows(svi, svi.elapsedMs).length).toBe(60)
      expect(frameAt(svi, 0)).toBe(0)
      expect(gridAt(svi, svi.elapsedMs)).toHaveLength(64)
    }
  })

  it('reads the marginals and the chain diagnostics of the results at a time', () => {
    const all = [...results.values()]
    const c = marginalsAt(all, [0, 1], Number.POSITIVE_INFINITY)
    expect(c?.quantities.map((q) => q.name)).toEqual(['w[0]', 'w[1]'])
    const sampler = all.find((r) => r.organisation === 'chain' && r.pool.at.length > 20)
    if (sampler !== undefined) expect(chainsAt(sampler, [0, 1], Number.POSITIVE_INFINITY)?.quantities).toHaveLength(2)
  })
})

describe('stopping and the pool', () => {
  it('a run that is stopped returns what it has', async () => {
    const r = await runLab(request('svi-mf', 20_000), host(300))
    expect(r.ending).toBe('stopped')
    expect(r.elapsedMs).toBeLessThan(5000)
    expect(r.frames.length).toBeGreaterThan(0)
  })

  it('the pool runs the requests where there are no Workers, one after the other, and reports each', async () => {
    const seen: PoolState[] = []
    await new Promise<void>((resolve) => {
      runPool(
        [request('laplace', 300), request('ep', 300)],
        { laplace: 'Laplace', ep: 'EP' },
        {
          onChange: (s) => {
            seen.push(s)
            if (!s.running) resolve()
          },
        },
      )
    })
    const last = seen.at(-1) as PoolState
    expect(last.lines.map((l) => l.phase)).toEqual(['done', 'done'])
    expect(Object.keys(last.results).sort()).toEqual(['ep', 'laplace'])
    expect(last.workers).toBe(0)
    expect(seen[0]?.lines.map((l) => l.phase)).toEqual(['queued', 'queued'])
  })

  it('uses at most min(methods, cores - 1, 4) workers', () => {
    expect(poolSize(10, 16)).toBe(4)
    expect(poolSize(2, 16)).toBe(2)
    expect(poolSize(10, 3)).toBe(2)
    expect(poolSize(10, 1)).toBe(1)
  })
})
