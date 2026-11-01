import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { maximiseEvidence, maximiseEvidenceAlgorithm } from './bayesian.ts'
import { evidenceUpdates } from './bayesian-options.ts'
import { designOf } from './index.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const],
  detail: 'full' as const,
}
const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

// A line with noise: y = 1 + 2 t + a little scatter, features (1, t).
const t = Array.from({ length: 30 }, (_, i) => i / 10)
const noise = t.map((_, i) => 0.15 * Math.sin(7 * i + 1))
const design = designOf({ x: t.map((v) => [1, v]), y: t.map((v, i) => 1 + 2 * v + (noise[i] ?? 0)) }, 'example')

describe('maximiseEvidenceAlgorithm', () => {
  it('raises the log evidence every iteration and ends where maximiseEvidence does', () => {
    const rec = recorder()
    const run = runAlgorithm(maximiseEvidenceAlgorithm(design, { alpha: 1, beta: 1 }), {
      maxSteps: 200,
      tracer: rec,
      policy,
    })
    expect(run.status).toBe('converged')
    const evidence = metricsOf(rec, 'evidence/logEvidence')
    expect(evidence.length).toBe(run.steps)
    expect(evidence.every((v, i) => i === 0 || v >= (evidence[i - 1] ?? 0) - 1e-9)).toBe(true)
    const fit = maximiseEvidence(design, { alpha: 1, beta: 1 })
    expect(fit.iterations).toBe(run.steps)
    expect(fit.alpha).toBeCloseTo(run.state.alpha, 12)
    expect(fit.beta).toBeCloseTo(run.state.beta, 12)
    expect(metricsOf(rec, 'evidence/gamma').at(-1)).toBeGreaterThan(0)
    expect(metricsOf(rec, 'evidence/gamma').at(-1)).toBeLessThanOrEqual(2 + 1e-9)
  })

  it('resumes from a JSON checkpoint, including the not-yet-evaluated evidence', () => {
    const alg = maximiseEvidenceAlgorithm(design, { alpha: 1, beta: 1 })
    const whole = runAlgorithm(alg, { maxSteps: 200 })
    const fresh = viaJson(alg.serialise?.(alg.init({ words: [0, 0], kind: 'key' } as never)) ?? null)
    const rest0 = runAlgorithm(alg, { maxSteps: 200, start: { state: fresh, step: 0, serialised: true } })
    expect(rest0.steps).toBe(whole.steps)
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 198, start: { state: stored, step: 2, serialised: true } })
    expect(rest.steps).toBe(whole.steps)
    expect(rest.state.alpha).toBeCloseTo(whole.state.alpha, 12)
    expect(rest.state.iterations).toBe(whole.state.iterations)
  })

  it('EM and MacKay climb to the same maximum, EM more slowly', () => {
    const mackay = maximiseEvidence(design, { alpha: 1, beta: 1 }, { maxIter: 5000, tol: 1e-6 })
    const em = maximiseEvidence(design, { alpha: 1, beta: 1 }, { maxIter: 5000, tol: 1e-6, update: { kind: 'em' } })
    expect(em.converged).toBe(true)
    expect(em.iterations).toBeGreaterThanOrEqual(mackay.iterations)
    expect(em.alpha).toBeCloseTo(mackay.alpha, 1)
    expect(Math.log(em.beta)).toBeCloseTo(Math.log(mackay.beta), 1)
    expect(evidenceUpdates.names).toEqual(['mackay', 'em'])
  })

  it('stops at the iteration budget without converging', () => {
    const fit = maximiseEvidence(design, { alpha: 1, beta: 1 }, { maxIter: 1, tol: 0 })
    expect(fit.iterations).toBe(1)
    expect(fit.converged).toBe(false)
    expect(() => maximiseEvidenceAlgorithm(design, { alpha: 0, beta: 1 })).toThrow(/positive/)
  })
})
