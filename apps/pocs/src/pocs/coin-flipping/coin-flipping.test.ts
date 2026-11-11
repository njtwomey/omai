import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { inlineProducer } from 'omai-render/producer'
import { describe, expect, it } from 'vitest'
import { coinSteps } from '../../examples/index.ts'
import { coinMethod } from './method.ts'

const config = { ...(defaults(coinMethod.config) as Config), flips: 12 }

describe('the coin-flipping page', () => {
  it('runs the default method on the main thread (the fallback) and its probes read the posterior', () => {
    const p = inlineProducer(coinMethod, { config, steps: coinSteps(config) }, [{ label: '', config: {} }])
    for (let i = 0; i < 50 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    const s = p.sessions[0]
    expect(s?.status).toBe('converged')
    const updating = s?.probe(s.t, 'coin/updating', { seen: 4 }) as { seen: number; comparison: { kind: string } }
    expect(updating.seen).toBe(4)
    expect(updating.comparison.kind).toBe('posteriorComparison')
    const check = s?.probe(s.t, 'coin/check') as { meanError: number; numDraws: number }
    expect(Math.abs(check.meanError)).toBeLessThan(1)
    expect(check.numDraws).toBeGreaterThan(50)
  })

  it('stops at the step budget the method carries', () => {
    const budgeted = { ...config, budgetSteps: 15 }
    const p = inlineProducer(coinMethod, { config: budgeted, steps: 1000 }, [{ label: '', config: {} }])
    for (let i = 0; i < 20 && p.sessions[0]?.status === undefined; i++) p.advance({ steps: 50 })
    expect(p.sessions[0]?.t).toBe(15)
    expect(p.sessions[0]?.status).toBe('stopped')
  })
})
