import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { gpClassificationSteps } from '../../../examples/index.ts'
import { gpClassificationMethod } from './method.ts'

const config = { ...(defaults(gpClassificationMethod.config) as Config), rows: 20 }

describe('the gp-classification page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(gpClassificationMethod, { config, steps: gpClassificationSteps(config) }, [
      { label: '', config: {} },
    ])
    for (let i = 0; i < 50 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const fit = s?.probe(s.t, 'gpc/fit') as { dim: number; probability: { mean: number[] } }
    expect(fit.dim).toBe(1)
    expect(fit.probability.mean.length).toBe(100)
    const pairs = s?.probe(s.t, 'gpc/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
    const draws = s?.probe(s.t, 'gpc/draws') as { names: string[] }
    expect(draws.names.length).toBe(20)
  })

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 15 }
    const p = inlineProducer(gpClassificationMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(15)
    expect(p.sessions[0]?.status).toBe('stopped')
  })
})
