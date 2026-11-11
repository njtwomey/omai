import { linearRegressionSteps } from 'omai-engine/bayesian-examples'
import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { linearRegressionMethod } from './method.ts'

const config = { ...(defaults(linearRegressionMethod.config) as Config), rows: 20 }

describe('the linear-regression page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(linearRegressionMethod, { config, steps: linearRegressionSteps(config) }, [
      { label: '', config: {} },
    ])
    for (let i = 0; i < 50 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const fit = s?.probe(s.t, 'linreg/fit', { seen: 10 }) as { seen: number; coefficients: unknown[] }
    expect(fit.seen).toBe(10)
    expect(fit.coefficients.length).toBe(2)
    const pairs = s?.probe(s.t, 'linreg/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
    const check = s?.probe(s.t, 'linreg/check') as { against: string; maxMeanError: number }
    expect(check.against).toBe('exact')
    expect(check.maxMeanError).toBeLessThan(1)
  })

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 15 }
    const p = inlineProducer(linearRegressionMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(15)
    expect(p.sessions[0]?.status).toBe('stopped')
  })
})
