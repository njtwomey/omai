import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { exampleSteps } from '../../../examples/index.ts'
import { schoolsExample } from './method.ts'

const config = { ...(defaults(schoolsExample.method.config) as Config), budgetMs: 0 }

describe('the eight schools page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(schoolsExample.method, { config, steps: exampleSteps(config) }, [
      { label: '', config: {} },
    ])
    for (let i = 0; i < 100 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const est = s?.probe(s.t, 'schools/estimates') as { kind: string; mean: number[] }
    expect(est.kind).toBe('groupEstimates')
    expect(est.mean).toHaveLength(8)
    const pairs = s?.probe(s.t, 'schools/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
  })
})
