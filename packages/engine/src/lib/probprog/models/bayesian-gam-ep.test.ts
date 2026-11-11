// Expectation propagation on every form of the Bayesian GAM: the offer for all twelve (prior, hyperprior) combinations, and the
// accuracy of the two forms written non-centred (`z / sqrt(precision)`): the penalty prior with log-normal hyperpriors against the
// exact reference, and ARD against a sampler.
import { describe, expect, it } from 'vitest'
import { session } from '../../trace/index.ts'
import { toFactorGraph } from '../model/index.ts'
import {
  bayesianGam,
  bayesianGamAgreement,
  type bayesianGamFit,
  bayesianGamMethod,
  bayesianGamMethods,
  bayesianGamReference,
} from './bayesian-gam.ts'

const x = Array.from({ length: 60 }, (_, i) => i / 59)
const noise = (i: number): number => 0.25 * Math.sin(i * 12.9898) * Math.cos(i * 78.233)
const y = x.map((v, i) => Math.sin(2 * Math.PI * v) + noise(i))

const fitOf = (kind: string, model: ReturnType<typeof bayesianGam>, config = {}) => {
  const s = session(bayesianGamMethod(kind, model, { fit: { gridSize: 20 } }), {
    steps: 3000,
    mode: 'eager',
    config: config as never,
  })
  return s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
}

describe('expectation propagation on the Bayesian GAM', () => {
  const priors = ['penalty', 'spikeSlab', 'ard'] as const
  const hypers = ['logNormal', 'gammaNoise', 'fixed', 'gammaSmooth'] as const
  const combos = priors.flatMap((prior) => hypers.map((hyper) => [prior, hyper] as const))

  it.each(combos)('is offered for the %s prior with %s hyperpriors', (prior, hyper) => {
    const offer = bayesianGamMethods(prior, hyper)
    expect(offer.applicable, `${prior} ${hyper}: ${offer.unavailable['ep']}`).toContain('ep')
  })

  it('opens on EP for the quick forms (a gamma or known scale), HMC for the rest', () => {
    const first = (prior: 'penalty' | 'ard', hyper: 'gammaSmooth' | 'logNormal'): unknown =>
      (bayesianGamMethods(prior, hyper).space as unknown as { default?: unknown }).default
    expect(first('penalty', 'gammaSmooth')).toBe('ep')
    expect(first('penalty', 'logNormal')).toBe('hmc')
    expect(first('ard', 'gammaSmooth')).toBe('hmc')
  })

  it('compiles the non-centred scale to one apply-product factor per smoothed coefficient, with no dead apply factor left', () => {
    const g = bayesianGam(x, y, { k: 10 })
    const c = toFactorGraph(g.definition.bind(g.data))
    expect(c.issues).toEqual([])
    const operators = (c.graph?.factors ?? []).map((f) => f.operator)
    expect(operators.filter((o) => o === 'apply-product')).toHaveLength(g.penalised)
    expect(operators).not.toContain('apply')
    const steps = (c.graph?.factors ?? []).find((f) => f.operator === 'apply-product')?.params?.['steps']
    // exp, then the eigenvalue times it (plus the fixed precision where there is one), then the square root and the reciprocal.
    const names = (steps as unknown as [string][]).map((s) => s[0])
    expect([names[0], names[1], names.at(-2), names.at(-1)]).toEqual(['exp', 'mul', 'sqrt', 'recip'])
  })

  it('agrees with the exact reference on the penalty prior with log-normal hyperpriors: the curve, the band and the smoothing', () => {
    const model = bayesianGam(x, y, { k: 10 })
    const reference = bayesianGamReference(model, { gridSize: 20, points: 28 })
    const fit = fitOf('ep', model)
    const a = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(a.meanError).toBeLessThan(0.02)
    // The posterior is read as independent marginals of the coefficients, which widens the band (as for the per-coefficient route).
    expect(a.bandRatio).toBeGreaterThan(0.95)
    expect(a.bandRatio).toBeLessThan(1.3)
    expect(Math.abs(a.logLambdaShift)).toBeLessThan(0.5)
    const lam = fit.hyper.logLambda as readonly [number, number]
    expect(lam[1]).toBeGreaterThan(0.8 * reference.hyper.logLambda[1])
    expect(lam[1]).toBeLessThan(1.25 * reference.hyper.logLambda[1])
    expect(fit.hyper.logSigma[0]).toBeCloseTo(reference.hyper.logSigma[0], 1)
  })

  it('agrees with a sampler on ARD: the curve, the band and which bases are in use', () => {
    const model = bayesianGam(x, y, { k: 10, prior: 'ard' })
    const ep = fitOf('ep', model)
    const nuts = fitOf('nuts', model, { numChains: 1, numWarmup: 100, numDraws: 200, maxTreeDepth: 5 })
    const sd = model.standardisation.sd
    const worst = Math.max(...ep.mean.map((m, i) => Math.abs(m - (nuts.mean[i] ?? 0)) / sd))
    expect(worst).toBeLessThan(0.06)
    const width = (f: typeof ep): number => f.upper.reduce((a, u, i) => a + u - (f.lower[i] ?? 0), 0)
    expect(width(ep) / width(nuts)).toBeGreaterThan(0.8)
    expect(width(ep) / width(nuts)).toBeLessThan(1.3)
    for (const [j, v] of (ep.inclusion ?? []).entries())
      expect(Math.abs(v - (nuts.inclusion?.[j] ?? 0))).toBeLessThan(0.25)
  })
})
