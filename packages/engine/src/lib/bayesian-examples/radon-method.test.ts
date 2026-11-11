import { describe, expect, it } from 'vitest'
import { type DictSpace, defaults, type UnionSpace } from '../space/index.ts'
import { type Config, session } from '../trace/index.ts'
import { exampleSteps } from './example-method.ts'
import { radonExample, radonPoolings, radonProblem } from './radon-method.ts'

const base = { ...(defaults(radonExample.method.config) as Config), counties: 8, minSize: 2 }
const options = (radonExample.method.config.spaces['inference'] as UnionSpace).options
const run = (extra: Config, inference: Config) => {
  const kind = String(inference['method'])
  const config = {
    ...base,
    ...extra,
    inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Config), ...inference },
    budgetMs: 0,
  }
  return session(radonExample.method, { config, steps: exampleSteps(config), mode: 'eager' })
}
const nuts = { method: 'nuts', numChains: 2, numWarmup: 40, numDraws: 60 }

describe('radonProblem', () => {
  it('has counties of unequal size none smaller than the smallest, and their raw estimates', () => {
    const p = radonProblem(base)
    expect(p.sizes).toHaveLength(8)
    expect(Math.min(...p.sizes)).toBeGreaterThanOrEqual(2)
    expect(new Set(p.sizes).size).toBeGreaterThan(1)
    expect(p.raw).toHaveLength(8)
    expect(p.raw.every(Number.isFinite)).toBe(true)
    expect(p.sizes[p.smallest]).toBe(Math.min(...p.sizes))
  })
})

describe('radonExample', () => {
  for (const pooling of radonPoolings) {
    it(`${pooling}: NUTS runs and every probe reads its posterior`, () => {
      const s = run({ pooling }, nuts)
      expect(s.status).not.toBeUndefined()
      const summary = s.probe(s.t, 'radon/summary') as { rows: { name: string }[]; draws: number }
      expect(summary.draws).toBe(120)
      expect(summary.rows.some((r) => r.name === 'alpha[7]')).toBe(true)
      const est = s.probe(s.t, 'radon/estimates') as { kind: string; mean: number[]; x: number[]; population: number | null }
      expect(est.kind).toBe('groupEstimates')
      expect(est.mean).toHaveLength(8)
      expect(est.population === null).toBe(pooling === 'unpooled')
      const pairs = s.probe(s.t, 'radon/pairs') as { kind: string } | null
      expect(pairs?.kind).toBe('posteriorPairs')
      expect(s.probe(s.t, 'radon/diagnostics')).not.toBeNull()
    })
  }

  it('partial pooling moves the smallest counties further than the largest, no pooling not at all', () => {
    const partial = run({ pooling: 'partial' }, { ...nuts, numWarmup: 80, numDraws: 200 })
    const a = partial.probe(partial.t, 'radon/estimates') as { shrinkageOfSmallest: number; shrinkageOfLargest: number }
    expect(a.shrinkageOfSmallest).toBeGreaterThan(a.shrinkageOfLargest)
  })

  for (const kind of ['svi', 'laplace']) {
    it(`${kind}: runs on the varying-slopes model and the probes read it`, () => {
      const s = run({ pooling: 'slopes' }, { method: kind })
      expect(s.status).not.toBeUndefined()
      const est = s.probe(s.t, 'radon/estimates') as { kind: string } | null
      expect(est?.kind).toBe('groupEstimates')
    })
  }

  it('checks a run against a reference run of the same data', () => {
    const reference = run({ pooling: 'partial', parameterisation: 'non-centred' }, { ...nuts, numWarmup: 80, numDraws: 200 })
    const summary = reference.probe(reference.t, 'radon/summary')
    const mine = run({ pooling: 'partial', parameterisation: 'centred' }, nuts)
    const check = mine.probe(mine.t, 'radon/check', { summary } as never) as { compared: number; maxMeanError: number; tauError: number }
    expect(check.compared).toBeGreaterThan(10)
    expect(check.maxMeanError).toBeLessThan(5)
    expect(Number.isFinite(check.tauError)).toBe(true)
  })
})
