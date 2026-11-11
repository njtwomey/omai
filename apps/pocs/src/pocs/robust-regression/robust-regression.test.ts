import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { robustRegressionSteps } from '../../examples/index.ts'
import { robustRegressionMethod } from './method.ts'

const config = { ...(defaults(robustRegressionMethod.config) as Config), rows: 25 }

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
