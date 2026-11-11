import { mixtureSteps } from 'omai-engine/bayesian-examples'
import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { mixtureMethod } from './method.ts'

const config = { ...(defaults(mixtureMethod.config) as Config), rows: 40, budgetMs: 0 }

describe('the gaussian-mixture page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(mixtureMethod, { config, steps: mixtureSteps(config) }, [{ label: '', config: {} }])
    for (let i = 0; i < 100 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('completed')
    const fit = s?.probe(s.t, 'mixture/fit') as { rows: { name: string }[]; mean: number[] }
    expect(fit.rows.map((r) => r.name)).toEqual(['mean 1', 'mean 2', 'weight 1', 'weight 2', 'spread'])
    expect(fit.mean).toHaveLength(90)
    const pairs = s?.probe(s.t, 'mixture/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
    const diagnostics = s?.probe(s.t, 'mixture/diagnostics') as { kind: string }
    expect(diagnostics.kind).toBe('chainDiagnostics')
  })

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 15 }
    const p = inlineProducer(mixtureMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(15)
    expect(p.sessions[0]?.status).toBe('stopped')
  })
})
