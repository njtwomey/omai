import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { irtSteps } from '../../../examples/index.ts'
import { irtMethod } from './method.ts'

const config = { ...(defaults(irtMethod.config) as Config), students: 15, questions: 6, budgetMs: 0 }

describe('the item-response page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(irtMethod, { config, steps: irtSteps(config) }, [{ label: '', config: {} }])
    for (let i = 0; i < 100 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(['completed', 'converged']).toContain(s?.status)
    const r = s?.probe(s.t, 'irt/recovery') as { abilities: { truth: number[]; coverage: number } }
    expect(r.abilities.truth).toHaveLength(15)
    expect(r.abilities.coverage).toBeGreaterThan(0)
    const cal = s?.probe(s.t, 'irt/calibration') as { counts: { students: number } }
    expect(cal.counts.students).toBe(15)
    const pairs = s?.probe(s.t, 'irt/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
  }, 60000)

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 15 }
    const p = inlineProducer(irtMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(15)
    expect(p.sessions[0]?.status).toBe('stopped')
  })
})
