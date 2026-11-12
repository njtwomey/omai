// The mixture page's compute: every method runs on the library's mixture model through the code a worker runs, within a small
// budget, and the results hold what the page draws.
import { describe, expect, it } from 'vitest'
import { type PoolState, runPool } from '../../../shared/workerPool.ts'
import { jobId, METHODS, type MixtureResult, namesOf, runMixture, waitingTimes } from './compute.ts'
import { chainsAt, coordinatesOf, fitTrace, pairsOf, predictiveAt, visibleRows } from './shown.ts'

const host = (): Parameters<typeof runMixture>[1] => ({
  now: () => performance.now(),
  yieldToPage: () => new Promise((resolve) => setTimeout(resolve, 0)),
  stopped: () => false,
  status: () => undefined,
})
const xs = Array.from({ length: 30 }, (_, i) => 40 + 2 * i)

describe('the data', () => {
  it('is the 272 Old Faithful waiting times', () => {
    const { minutes, mean } = waitingTimes()
    expect(minutes).toHaveLength(272)
    expect(mean).toBeCloseTo(70.9, 1)
    expect(namesOf(3)).toHaveLength(7)
    expect(coordinatesOf('both', 3)).toEqual([0, 1, 2, 3, 4, 5])
  })
})

describe('the methods on offer', () => {
  const results = new Map<string, MixtureResult>()
  for (const m of METHODS)
    it(`${m.label} runs on the unordered and the ordered mixture and returns draws with times`, async () => {
      for (const ordered of [false, true]) {
        const r = await runMixture(
          { method: jobId(m.kind, ordered), kind: m.kind, ordered, components: 2, budgetMs: 2500, seed: 1 },
          host(),
        )
        results.set(jobId(m.kind, ordered), r)
        expect(r.names).toHaveLength(5)
        expect(r.steps).toBeGreaterThan(0)
        // The ordered fit has the means in order, draw by draw, and weights that sum to one.
        const n = r.pool.at.length
        if (n === 0) continue
        for (let i = 0; i < n; i++) {
          const row = r.pool.values.subarray(i * 5, (i + 1) * 5)
          expect((row[2] ?? 0) + (row[3] ?? 0)).toBeCloseTo(1, 6)
          if (ordered) expect(row[0]).toBeLessThanOrEqual((row[1] ?? 0) + 1e-9)
        }
        if (m.organisation === 'chain') expect(r.chains).toBeGreaterThanOrEqual(2)
        else expect(r.frames.length).toBeGreaterThan(1)
      }
    }, 20000)

  it('the draws of a sampler appear over time and stay; a fit is shown one snapshot at a time', () => {
    const gibbs = results.get('gibbs-free') as MixtureResult
    expect(visibleRows(gibbs, 0).length).toBeLessThan(visibleRows(gibbs, Number.POSITIVE_INFINITY).length)
    expect(visibleRows(gibbs, Number.POSITIVE_INFINITY)).toHaveLength(gibbs.pool.at.length)
    const svi = results.get('svi-free') as MixtureResult
    expect(visibleRows(svi, Number.POSITIVE_INFINITY).length).toBe(60)
  })

  it('reads the predictive density, the pair record, the diagnostics and the trace at a playhead', () => {
    const gibbs = results.get('gibbs-free') as MixtureResult
    const p = predictiveAt(gibbs, xs, Number.POSITIVE_INFINITY)
    expect(p?.mean).toHaveLength(30)
    expect(p?.components).toHaveLength(2)
    expect(p?.lower.every((v, i) => v <= (p?.upper[i] ?? 0))).toBe(true)
    expect(predictiveAt(gibbs, xs, -1)).toBeUndefined()
    expect(pairsOf([gibbs, results.get('gibbs-ordered') as MixtureResult], 'means')?.methods).toHaveLength(2)
    const c = chainsAt(gibbs, 'means', Number.POSITIVE_INFINITY)
    expect(c?.quantities.map((q) => q.name)).toEqual(['mean 1', 'mean 2', 'spread'])
    expect(chainsAt(results.get('svi-free') as MixtureResult, 'means', 1e9)).toBeUndefined()
    expect(fitTrace(results.get('svi-free') as MixtureResult, 0, 1e9).x.length).toBeGreaterThan(1)
  })
})

describe('the pool', () => {
  it('runs the requests on the main thread when there are no Workers and reports each result', async () => {
    const states: PoolState<MixtureResult>[] = []
    const requests = [true, false].map((ordered) => ({
      method: jobId('gibbs', ordered),
      kind: 'gibbs',
      ordered,
      components: 2,
      budgetMs: 800,
      seed: 2,
    }))
    await new Promise<void>((resolve) => {
      runPool(
        requests,
        {},
        {
          run: runMixture,
          onChange: (s) => {
            states.push(s)
            if (!s.running) resolve()
          },
        },
      )
    })
    expect(Object.keys(states.at(-1)?.results ?? {}).sort()).toEqual(['gibbs-free', 'gibbs-ordered'])
  }, 20000)
})
