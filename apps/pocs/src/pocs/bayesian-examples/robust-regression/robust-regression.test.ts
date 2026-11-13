import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { robustRegressionSteps } from '../../../examples/index.ts'
import { robustRegressionMethod } from './method.ts'

// The page's default time budget is 2 s, and a run that has used it is `stopped`, not `converged`. NUTS takes about that long here
// (it took 2.07 s on the CI runner, and was stopped), so the test lifts the budget: what it checks is the run and its probes.
const config = { ...(defaults(robustRegressionMethod.config) as Config), rows: 25, budgetMs: 60_000 }

describe('the robust-regression page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(robustRegressionMethod, { config, steps: robustRegressionSteps(config) }, [
      { label: '', config: {} },
    ])
    for (let i = 0; i < 200 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const fit = s?.probe(s.t, 'robust/fit', { seen: 15 }) as { seen: number; coefficients: unknown[]; errors: unknown }
    expect(fit.seen).toBe(15)
    expect(fit.coefficients.length).toBeGreaterThanOrEqual(3)
    expect(fit.errors).toBeDefined()
    const df = s?.probe(s.t, 'robust/df') as { draws: number[] }
    expect(df.draws.length).toBeGreaterThan(5)
  })
})
