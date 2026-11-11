import { defaults } from 'omai-engine/space'
import { type Config, session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import {
  volatilityExample,
  volatilityMethod,
  volatilityProblem,
  volatilitySteps,
} from './stochastic-volatility-method.ts'

const base = { ...(defaults(volatilityMethod.config) as Config), length: 40 }

function run(over: Config, inference: Config): ReturnType<typeof session> {
  const offer = volatilityExample.offer({ ...base, ...over })
  const config = {
    ...base,
    ...over,
    inference: { ...(defaults(offer.space.options[inference['method'] as string] as never) as Config), ...inference },
  }
  return session(volatilityMethod, { config, steps: volatilitySteps(config), mode: 'eager' }) as never
}

describe('stochastic volatility', () => {
  it('simulates returns with a known truth, or takes the library GARCH series with its true variance', () => {
    const sim = volatilityProblem(base)
    expect(sim.y.length).toBe(40)
    expect(sim.truth?.phi).toBe(0.95)
    const garch = volatilityProblem({ ...base, dataset: 'garch' })
    expect(garch.truth).toBeNull()
    expect(garch.truthLogVariance.length).toBe(40)
  })

  it('maps expectation propagation through the AR(1) scan when mu, phi and sigma are known, and says why not otherwise', () => {
    const known = volatilityExample.offer({ ...base, parameters: 'known' })
    expect(known.applicable).toEqual(expect.arrayContaining(['ep', 'particle-filter', 'nuts']))
    const inferred = volatilityExample.offer(base)
    expect(inferred.applicable).not.toContain('ep')
    // With phi and sigma inferred the initial state's scale is a computed random value, which has no message rule (a product with a random slope).
    expect(inferred.unavailable['ep']).toMatch(/precision|scale/)
  })

  it('EP on the known-parameter model is close to the exact point-mass smoother, closer than Laplace', () => {
    const over = { parameters: 'known', parameterisation: 'centred' }
    const ep = run(over, { method: 'ep' })
    const check = ep.probe(ep.t, 'sv/check') as { maxMeanError: number; medianSdRatio: number; against: string }
    expect(check.against).toBe('smoother')
    expect(check.maxMeanError).toBeLessThan(0.3)
    expect(check.medianSdRatio).toBeGreaterThan(0.9)
    expect(check.medianSdRatio).toBeLessThan(1.1)
    const laplace = run(over, { method: 'laplace' })
    const lc = laplace.probe(laplace.t, 'sv/check') as { maxMeanError: number }
    expect(check.maxMeanError).toBeLessThan(lc.maxMeanError)
  }, 60000)

  it('reads a fit, scores and pairs from a variational run with the numbers inferred', () => {
    const s = run({}, { method: 'svi' })
    const fit = s.probe(s.t, 'sv/fit') as { t: number[]; lo: number[]; hi: number[] }
    expect(fit.t.length).toBe(40)
    expect(fit.lo.every((v, i) => v < (fit.hi[i] ?? 0))).toBe(true)
    const scores = s.probe(s.t, 'sv/scores') as { coverage: number; numbers: { phi: { mean: number } } | null }
    expect(scores.coverage).toBeGreaterThan(0)
    expect(scores.numbers?.phi.mean).toBeDefined()
    const pairs = s.probe(s.t, 'sv/pairs') as { kind: string }
    expect(pairs.kind).toBe('posteriorPairs')
  }, 60000)
})
