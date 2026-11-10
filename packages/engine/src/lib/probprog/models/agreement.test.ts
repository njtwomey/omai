// The benchmark runner: the exact methods reproduce the closed-form truths, an engine of every model runs at a small budget,
// a plan expected to differ that agrees (or a plan that must agree and cannot run) is a surprise, a campaign writes its
// table, and a worker-style run by name equals the truth.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { benchmarkAgreement, runPlan } from './agreement.ts'
import { benchmarkMethods, defineBenchmark, methodVerdicts } from './benchmark.ts'
import { benchmarkCampaigns } from './campaigns.ts'
import { betaBernoulli, gammaPoisson, normalNormal } from './conjugate.ts'
import { exactPlan, nutsPlan } from './plans.ts'
import { ar1StateSpace } from './state-space.ts'
import { benchmarkModels } from './suite.ts'

describe('the exact methods reproduce the exact truths', () => {
  it.each([
    [betaBernoulli, ['conjugate']],
    [normalNormal, ['conjugate', 'gaussian']],
    [gammaPoisson, ['conjugate']],
    [ar1StateSpace, ['gaussian']],
  ] as const)('%s', (model, kinds) => {
    const a = benchmarkAgreement(model, { only: kinds, scale: 0.1 })
    expect(a.runs.map((r) => r.plan.kind)).toEqual([...kinds])
    expect(a.failures).toEqual([])
    expect(a.passed).toBe(true)
    for (const row of a.table.rows) expect(Math.abs(row.meanError), `${row.method} ${row.quantity}`).toBeLessThan(0.1)
  })
})

// The funnels, the banana and the two-mode target have no observed node: their methods are made on the bound spec, as every
// other model's, and every planned method's own check accepts it.
describe('a target with no observed node runs on its spec', () => {
  const densities = benchmarkModels().filter((m) => m.target === 'density')
  it('is the funnels, the banana and the two-mode target', () => {
    expect(densities.map((m) => m.name)).toEqual(['funnel-centred', 'funnel-non-centred', 'banana', 'bimodal'])
  })
  it.each(densities.map((m) => [m.name, m] as const))('%s: every planned method applies and runs', (_name, model) => {
    expect(typeof model.methodModel()).toBe('object')
    const verdicts = methodVerdicts(model)
    for (const plan of model.plans(0.02)) {
      expect(verdicts.find((v) => v.kind === plan.kind)?.verdict.applies, plan.kind).toBe(true)
      const run = runPlan(model, plan, { key: key(1) })
      expect(run.posterior.sites.length, plan.kind).toBeGreaterThan(0)
      expect(run.status, plan.kind).not.toBe('failed')
    }
  })
})

describe('an engine of every model runs at a small budget', () => {
  it.each(benchmarkModels().map((m) => [m.name, m] as const))('%s', (_name, model) => {
    const plans = model.plans(0.02)
    const plan = plans.find((p) => p.kind === 'nuts') ?? plans[0]
    const run = runPlan(model, plan as NonNullable<typeof plan>, { key: key(1) })
    expect(run.posterior.sites.length).toBeGreaterThan(0)
    expect(['converged', 'completed']).toContain(run.status)
    expect(run.steps).toBeGreaterThan(0)
  })
})

describe('the expectations of a plan', () => {
  const custom = (plans: ReturnType<typeof exactPlan>[]) =>
    defineBenchmark({
      name: 'probe',
      title: 'Probe',
      description: 'The normal-normal model with other plans.',
      tags: ['probe'],
      references: [],
      definition: normalNormal.definition as NonNullable<typeof normalNormal.definition>,
      generate: normalNormal.generate,
      truth: (data) => normalNormal.truth(data),
      plans: () => plans,
    })

  it('a method expected to differ that agrees is a surprise, and fails the run', () => {
    const a = benchmarkAgreement(custom([exactPlan('conjugate', { expect: 'differ', note: 'it should differ' })]))
    expect(a.surprises).toHaveLength(1)
    expect(a.surprises[0]).toContain('agrees with it')
    expect(a.passed).toBe(false)
  })

  it('a method that must agree and cannot run (its own check says why) is a surprise with the reason', () => {
    const a = benchmarkAgreement(custom([exactPlan('enumeration')]))
    expect(a.skipped[0]?.kind).toBe('enumeration')
    expect(a.skipped[0]?.reason).toContain('continuous latent nodes')
    expect(a.surprises[0]).toContain('did not run')
    expect(a.passed).toBe(false)
  })

  it('a method that is only reported fails nothing, even far from the truth, and tolerances can be tightened', () => {
    const wrong = custom([nutsPlan(0.05, { expect: 'report', note: 'tiny budget' })])
    expect(benchmarkAgreement(wrong).passed).toBe(true)
    const tight = custom([nutsPlan(0.05, { tolerances: { ks: 1e-6 }, note: 'tiny budget, impossible tolerance' })])
    const a = benchmarkAgreement(tight)
    expect(a.failures.length).toBeGreaterThan(0)
    expect(a.failures[0]?.failed).toContain('ks')
  })

  it('quantities a posterior does not hold are noted, not compared', () => {
    const a = benchmarkAgreement(
      defineBenchmark({
        name: 'probe-noted',
        title: 'Probe',
        description: 'A truth with a quantity no posterior has.',
        tags: ['probe'],
        references: [],
        definition: normalNormal.definition as NonNullable<typeof normalNormal.definition>,
        generate: normalNormal.generate,
        truth: (data) => {
          const t = normalNormal.truth(data)
          return {
            ...t,
            marginals: { ...t.marginals, ghost: t.marginals['mu'] as NonNullable<(typeof t.marginals)['mu']> },
          }
        },
        plans: () => [exactPlan('conjugate')],
      }),
    )
    expect(a.notes[0]).toContain('no draws of ghost')
    expect(a.table.rows.map((r) => r.quantity)).toEqual(['mu'])
  })
})

describe('the campaigns', () => {
  it('has a campaign per model, named benchmark/<model>, and writes a table with the truth, the verdicts and the rows', () => {
    const models = benchmarkModels()
    const campaigns = benchmarkCampaigns()
    expect(campaigns.map((c) => c.name)).toEqual(models.map((m) => `benchmark/${m.name}`))
    // The exact methods only, so that the budget does not decide it.
    const exactOnly = defineBenchmark({
      name: 'normal-normal-exact',
      title: 'Normal-normal, exact methods',
      description: 'The normal-normal model with its exact methods planned.',
      tags: ['probe'],
      references: [],
      definition: normalNormal.definition as NonNullable<typeof normalNormal.definition>,
      generate: normalNormal.generate,
      truth: (data) => normalNormal.truth(data),
      plans: () => [exactPlan('conjugate'), exactPlan('gaussian')],
    })
    const c = benchmarkCampaigns([exactOnly])[0]
    const out = c?.run({ key: key(0), scale: 0.3 })
    expect(out?.passed).toBe(true)
    expect(out?.markdown).toContain('Truth (exact)')
    expect(out?.markdown).toContain('| method | quantity | mean |')
    expect(out?.markdown).toContain('Not applicable:')
    expect(out?.verdict).toBe('2 method(s) agree with the exact truth; 0 reported')
  })
})

describe('a worker runs a benchmark by name', () => {
  it('conjugate/beta-bernoulli through the registry equals the truth, and a program model is on offer under the same naming', () => {
    const methods = benchmarkMethods([betaBernoulli, ar1StateSpace], ['conjugate', 'gaussian'])
    expect(methods.map((m) => m.name)).toEqual([
      'conjugate/beta-bernoulli',
      'gaussian/beta-bernoulli',
      'conjugate/ar1-state-space',
      'gaussian/ar1-state-space',
    ])
    const m = methods[0] as NonNullable<(typeof methods)[0]>
    const s = session(m, { steps: 3, mode: 'eager' })
    const mean = item(m.posterior?.(s.state, s.config)?.mean('p') as never)
    expect(mean).toBeCloseTo(betaBernoulli.truth().marginals['p']?.mean ?? Number.NaN, 9)
  })
})
