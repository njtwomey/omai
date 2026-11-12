import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { exampleSteps } from '../../../examples/index.ts'
import { radonExample } from './method.ts'

const config = { ...(defaults(radonExample.method.config) as Config), counties: 6, budgetMs: 0 }

describe('the radon page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(radonExample.method, { config, steps: exampleSteps(config) }, [{ label: '', config: {} }])
    for (let i = 0; i < 100 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const est = s?.probe(s.t, 'radon/estimates') as { kind: string; mean: number[] }
    expect(est.kind).toBe('groupEstimates')
    expect(est.mean).toHaveLength(6)
    const summary = s?.probe(s.t, 'radon/summary') as { divergences: number }
    expect(typeof summary.divergences).toBe('number')
  })
})
