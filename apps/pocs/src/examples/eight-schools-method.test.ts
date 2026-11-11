import { type DictSpace, defaults, type UnionSpace } from 'omai-engine/space'
import { type Config, session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { schoolsExample, schoolsModels, schoolsProblem, schoolsReferenceModel } from './eight-schools-method.ts'
import { exampleSteps } from './example-method.ts'

const base = defaults(schoolsExample.method.config) as Config
const options = (schoolsExample.method.config.spaces['inference'] as UnionSpace).options
const run = (extra: Config, inference: Config) => {
  const kind = String(inference['method'])
  const config = {
    ...base,
    ...extra,
    inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Config), ...inference },
    budgetMs: 0,
  }
  return session(schoolsExample.method, { config, steps: exampleSteps(config), mode: 'eager' })
}
const nuts = { method: 'nuts', numChains: 2, numWarmup: 40, numDraws: 60 }

describe('schoolsProblem', () => {
  it('is the classic eight schools by default and carries the truth of generated groups', () => {
    expect(schoolsProblem(base).y).toHaveLength(8)
    expect(schoolsProblem(base).truth).toBeUndefined()
    const g = schoolsProblem({ ...base, dataset: { source: 'generated', groups: 14, tau: 2, se: 6, seSpread: 0.5 } })
    expect(g.y).toHaveLength(14)
    expect(g.truth?.tau).toBe(2)
    expect(new Set(g.sigma).size).toBeGreaterThan(1)
  })

  it('names the non-centred model as the reference of both partial-pooling forms', () => {
    expect(schoolsReferenceModel('centred')).toBe('non-centred')
    expect(schoolsReferenceModel('no-pooling')).toBe('no-pooling')
  })
})

describe('schoolsExample', () => {
  for (const model of schoolsModels) {
    it(`${model}: NUTS runs and every probe reads its posterior`, () => {
      const s = run({ model }, nuts)
      expect(s.status).not.toBeUndefined()
      const summary = s.probe(s.t, 'schools/summary') as { rows: { name: string }[]; draws: number }
      expect(summary.draws).toBe(120)
      expect(summary.rows.some((r) => r.name === 'theta[7]')).toBe(true)
      const est = s.probe(s.t, 'schools/estimates') as { kind: string; mean: number[]; population: number | null }
      expect(est.kind).toBe('groupEstimates')
      expect(est.mean).toHaveLength(8)
      expect(est.population === null).toBe(model === 'no-pooling')
      const pairs = s.probe(s.t, 'schools/pairs') as { kind: string; coordinates: { name: string }[] } | null
      expect(pairs?.kind).toBe(model === 'complete-pooling' ? undefined : 'posteriorPairs')
      expect(s.probe(s.t, 'schools/diagnostics')).not.toBeNull()
    })
  }

  it('complete pooling puts every school on mu and no pooling leaves them near their raw estimates', () => {
    const p = run({ model: 'complete-pooling' }, nuts)
    const pooled = p.probe(p.t, 'schools/estimates') as { mean: number[]; population: number }
    expect(Math.max(...pooled.mean) - Math.min(...pooled.mean)).toBeLessThan(1e-9)
    const unpooled = run({ model: 'no-pooling' }, nuts)
    const est = unpooled.probe(unpooled.t, 'schools/estimates') as {
      mean: number[]
      raw: number[]
      meanShrinkage: number
    }
    for (const [j, m] of est.mean.entries()) expect(Math.abs(m - (est.raw[j] as number))).toBeLessThan(12)
  })

  for (const kind of ['svi', 'laplace', 'pathfinder']) {
    it(`${kind}: runs on the non-centred model and the probes read it`, () => {
      const s = run({ model: 'non-centred' }, { method: kind })
      expect(s.status).not.toBeUndefined()
      const est = s.probe(s.t, 'schools/estimates') as { kind: string } | null
      expect(est?.kind).toBe('groupEstimates')
      expect(s.probe(s.t, 'schools/diagnostics')).toBeNull()
    })
  }

  it('checks a run against a reference run of the same data', () => {
    const reference = run({ model: 'non-centred' }, { ...nuts, numWarmup: 80, numDraws: 200 })
    const summary = reference.probe(reference.t, 'schools/summary')
    const mine = run({ model: 'centred' }, nuts)
    const check = mine.probe(mine.t, 'schools/check', { summary } as never) as {
      compared: number
      maxMeanError: number
      tauError: number
    }
    expect(check.compared).toBe(10)
    expect(check.maxMeanError).toBeLessThan(5)
    expect(Number.isFinite(check.tauError)).toBe(true)
  })
})
