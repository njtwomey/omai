// The coin page's compute: every method on offer is fitted along the ladder of a small problem through the code a worker
// runs, and the frames hold what the page draws.
import { describe, expect, it } from 'vitest'
import { type PoolState, runPool } from '../../shared/workerPool.ts'
import {
  type CoinRequest,
  type CoinResult,
  exactAt,
  ladderOf,
  METHODS,
  makeProblem,
  PRESETS,
  runCoin,
} from './compute.ts'

const host = (): Parameters<typeof runCoin>[1] => ({
  now: () => performance.now(),
  yieldToPage: () => new Promise((resolve) => setTimeout(resolve, 0)),
  stopped: () => false,
  status: () => undefined,
})

const beta = makeProblem({ family: 'beta', truth: 0.3, n: 14, preset: 'flat', seed: 1 })
const gamma = makeProblem({ family: 'gamma', truth: 3.2, n: 14, preset: 'flat', seed: 1 })

describe('the problem', () => {
  it('has a ladder from 0 to every observation, ascending and distinct', () => {
    for (const n of [1, 5, 14, 100, 200]) {
      const l = ladderOf(n)
      expect(l[0]).toBe(0)
      expect(l.at(-1)).toBe(n)
      expect(new Set(l).size).toBe(l.length)
      expect(l).toEqual([...l].sort((a, b) => a - b))
      expect(l.length).toBeLessThanOrEqual(21)
    }
    expect(ladderOf(200).length).toBeGreaterThanOrEqual(15)
  })

  it('draws coins and counts from the seed, the same every time', () => {
    expect(beta.observations).toHaveLength(14)
    expect(beta.observations.every((v) => v === 0 || v === 1)).toBe(true)
    expect(gamma.observations.every((v) => Number.isInteger(v) && v >= 0)).toBe(true)
    expect(makeProblem({ family: 'beta', truth: 0.3, n: 14, preset: 'flat', seed: 1 })).toEqual(beta)
  })

  it('has the exact posterior the conjugate formula gives, and the prior at n = 0', () => {
    const k = beta.observations.reduce((a, b) => a + b, 0)
    expect(exactAt(beta, 14).oracles('p')[0]?.mean).toBeCloseTo((1 + k) / (2 + 14), 12)
    expect(exactAt(beta, 0).oracles('p')[0]?.mean).toBeCloseTo(0.5, 12)
    expect(gamma.domain[1]).toBeGreaterThan(gamma.truth)
    for (const family of ['beta', 'gamma'] as const)
      for (const id of Object.keys(PRESETS[family]) as (keyof (typeof PRESETS)['beta'])[])
        expect(makeProblem({ family, truth: family === 'beta' ? 0.5 : 4, n: 5, preset: id, seed: 2 }).prior).toEqual(
          PRESETS[family][id].prior,
        )
  })
})

describe('the methods on offer', () => {
  for (const [name, problem] of [
    ['coin', beta],
    ['counter', gamma],
  ] as const)
    for (const m of METHODS)
      it(`${m.label} is fitted along the ladder of the ${name}`, async () => {
        const small = { ...problem, ladder: [0, 1, 4, 14] }
        const r = await runCoin({ method: m.id, problem: small }, host())
        expect(r.message).toBeUndefined()
        expect(r.frames.map((f) => f.n)).toEqual([0, 1, 4, 14])
        for (const f of r.frames) {
          expect(f.density).toHaveLength(problem.grid.length)
          expect(Array.from(f.density).every((v) => Number.isFinite(v) && v >= 0)).toBe(true)
          expect(f.lower).toBeLessThan(f.mean)
          expect(f.upper).toBeGreaterThan(f.mean)
          expect(f.sd).toBeGreaterThan(0)
        }
        const last = r.frames.at(-1)
        const truthMean = exactAt(problem, 14).oracles(name === 'coin' ? 'p' : 'rate')[0]?.mean ?? Number.NaN
        if (m.id === 'exact') expect(r.frames.every((f) => f.kl === 0 && f.w1 === 0)).toBe(true)
        else expect(last?.w1).toBeLessThan(0.5)
        expect(Math.abs((last?.mean ?? 0) - truthMean)).toBeLessThan(0.15 * truthMean)
      }, 30_000)
})

describe('the pool', () => {
  it('runs the requests through the shared pool on this thread, one result per method', async () => {
    const problem = { ...beta, ladder: [0, 3, 14] }
    const ids = ['exact', 'laplace', 'ep']
    const state = await new Promise<PoolState<CoinResult>>((resolve) => {
      runPool<CoinRequest, CoinResult>(
        ids.map((method) => ({ method, problem })),
        Object.fromEntries(METHODS.map((m) => [m.id, m.label])),
        { run: runCoin, onChange: (s) => !s.running && resolve(s) },
      )
    })
    expect(Object.keys(state.results).sort()).toEqual([...ids].sort())
    expect(state.lines.every((l) => l.phase === 'done')).toBe(true)
  })
})
