import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { classifierObjective, classifierSteps } from '../../../examples/index.ts'
import { classifierMethod } from './method.ts'

// The page's default time budget is 3 s, and a run that has used it is `stopped`, not `converged`. On a slow CI runner the default run
// can take that long, so the test lifts the budget: what it checks is the run and its probes.
const config = { ...(defaults(classifierMethod.config) as Config), rows: 60, budgetMs: 60_000 }

describe('the classify page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(classifierMethod, { config, steps: classifierSteps(config) }, [{ label: '', config: {} }])
    for (let i = 0; i < 50 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const surface = s?.probe(s.t, 'classify/surface', { size: 9 }) as { kind: string }
    expect(surface.kind).toBe('decisionSurface')
    const scores = s?.probe(s.t, 'classify/scores') as { testAccuracy: number; maxRhat: number }
    expect(scores.testAccuracy).toBeGreaterThan(0.5)
    expect(scores.maxRhat).toBeLessThan(1.5)
  })

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 12 }
    const p = inlineProducer(classifierMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(12)
    expect(p.sessions[0]?.status).toBe('stopped')
  })

  it('names the curve of a variational method and none for a sampler', () => {
    expect(classifierObjective({ inference: { method: 'svi' } })?.path).toBe('svi/elbo')
    expect(classifierObjective(config)).toBeUndefined()
  })
})
