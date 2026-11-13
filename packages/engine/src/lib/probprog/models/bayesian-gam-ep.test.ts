// Expectation propagation on every form of the Bayesian GAM: the offer for all twelve (prior, hyperprior) combinations, and the
// accuracy of the two forms written non-centred (`z / sqrt(precision)`): the penalty prior with log-normal hyperpriors against the
// exact reference, and ARD against a sampler.
import { describe, expect, it } from 'vitest'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, toFactorGraph } from '../model/index.ts'
import {
  bayesianGam,
  bayesianGamAgreement,
  type bayesianGamFit,
  bayesianGamFormStarting,
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

  it('compiles the non-centred penalty prior to one dense factor on the coefficient vector, with no standard normal, apply or apply-product left', () => {
    const g = bayesianGam(x, y, { k: 10 })
    const c = toFactorGraph(g.definition.bind(g.data))
    expect(c.issues).toEqual([])
    const factors = c.graph?.factors ?? []
    const operators = factors.map((f) => f.operator)
    expect(operators.filter((o) => o === 'gaussian-mv-function-scale')).toHaveLength(1)
    expect(operators).not.toContain('apply-product')
    expect(operators).not.toContain('apply')
    // The coefficients are one vector variable (read by the likelihood as a vector), and z is not a variable at all.
    const names = (c.graph?.variables ?? []).map((v) => v.name)
    expect(names).toContain('gamma')
    expect(names.some((n) => n.startsWith('z'))).toBe(false)
    const steps = factors.find((f) => f.operator === 'gaussian-mv-function-scale')?.params?.['steps'] as unknown as [
      string,
    ][][]
    // One chain per coefficient: exp, then the eigenvalue times it (plus the fixed precision where there is one), the square root and
    // the reciprocal; a coefficient the penalty does not touch is a constant.
    expect(steps).toHaveLength(10)
    const smoothed = steps.filter((chain) => chain[0]?.[0] === 'exp')
    expect(smoothed).toHaveLength(g.penalised)
    expect([smoothed[0]?.[1]?.[0], smoothed[0]?.at(-2)?.[0], smoothed[0]?.at(-1)?.[0]]).toEqual([
      'mul',
      'sqrt',
      'recip',
    ])
  })

  it('keeps one apply-product factor per coefficient when the dense form is switched off or the scales have no one variable', () => {
    const g = bayesianGam(x, y, { k: 10 })
    const elements = toFactorGraph(g.definition.bind(g.data), { sharedScale: 'elements' })
    const operators = (elements.graph?.factors ?? []).map((f) => f.operator)
    expect(operators.filter((o) => o === 'apply-product')).toHaveLength(g.penalised)
    expect(operators).not.toContain('gaussian-mv-function-scale')
    // A scale with one latent for each coefficient (not one shared variable) cannot be one factor on the vector.
    const model = defineModel('own-scales', (m) => {
      const row = m.range('row')
      const coef = m.range('coef')
      const X = m.data('X', { dims: [row, coef] })
      const obs = m.data('y', { dims: [row] })
      const a = m.latent('a', d.normal(0, 1), { dims: [coef] })
      const z = m.latent('z', d.normal(0, 1), { dims: [coef] })
      const beta = m.derived('beta', z.div(a.exp().sqrt()))
      m.observe('y', d.normal(X.matmul(beta), 0.3), obs)
    })
    const own = toFactorGraph(
      model.bind({
        X: [
          [1, 0],
          [0, 1],
          [1, 1],
        ],
        y: [0.1, 0.2, 0.3],
      }),
    )
    expect(own.issues).toEqual([])
    const ownOperators = (own.graph?.factors ?? []).map((f) => f.operator)
    expect(ownOperators).not.toContain('gaussian-mv-function-scale')
    expect(ownOperators.filter((o) => o === 'apply-product')).toHaveLength(2)
  })

  it('compiles ARD to one apply-product per coefficient, assembled into one Gaussian vector (gaussian-mv-stack) the likelihood reads', () => {
    const g = bayesianGam(x, y, { k: 10, prior: 'ard' })
    const c = toFactorGraph(g.definition.bind(g.data))
    expect(c.issues).toEqual([])
    const operators = (c.graph?.factors ?? []).map((f) => f.operator)
    expect(operators.filter((o) => o === 'apply-product')).toHaveLength(10)
    expect(operators.filter((o) => o === 'gaussian-mv-stack')).toHaveLength(1)
    // No scalar product is tied to a coordinate of the vector one by one (the rows' dot products read the vector).
    const ties = (c.graph?.factors ?? []).filter(
      (f) => f.operator === 'dot-constant' && String(f.args[0]).startsWith('beta#'),
    )
    expect(ties).toEqual([])
    // The rows read the vector (a dot product each), not a sum of scalar variables (a loop of messages).
    expect(operators).not.toContain('linear')
    expect(operators).not.toContain('sum')
    const beta = (c.graph?.variables ?? []).find((v) => v.name === 'beta')
    expect(beta?.family).toEqual({ kind: 'gaussian-mv', size: 10 })
  })

  it('agrees with the exact reference on the penalty prior with log-normal hyperpriors: the curve, the band and the smoothing', () => {
    const model = bayesianGam(x, y, { k: 10 })
    // 60 points on each hyperparameter: the spread of the log noise needs more than the default 28 to be resolved.
    const reference = bayesianGamReference(model, { gridSize: 20, points: 60 })
    const fit = fitOf('ep', model)
    const a = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(a.meanError).toBeLessThan(0.01)
    // The coefficients are one Gaussian vector (the dense factor), so the band keeps their correlations: within a few percent
    // (the per-coefficient form was 17% too wide).
    expect(a.bandRatio).toBeGreaterThan(0.95)
    expect(a.bandRatio).toBeLessThan(1.06)
    expect(Math.abs(a.logLambdaShift)).toBeLessThan(0.15)
    const lam = fit.hyper.logLambda as readonly [number, number]
    expect(lam[1]).toBeGreaterThan(0.8 * reference.hyper.logLambda[1])
    expect(lam[1]).toBeLessThan(1.25 * reference.hyper.logLambda[1])
    expect(fit.hyper.logSigma[0]).toBeCloseTo(reference.hyper.logSigma[0], 1)
    expect(fit.hyper.logSigma[1]).toBeGreaterThan(0.8 * reference.hyper.logSigma[1])
    expect(fit.hyper.logSigma[1]).toBeLessThan(1.3 * reference.hyper.logSigma[1])
  })

  it('reaches its fixed point in few sweeps on the dense form', () => {
    const model = bayesianGam(x, y, { k: 10 })
    const s = session(bayesianGamMethod('ep', model, { fit: { gridSize: 20 } }), { steps: 3000, mode: 'eager' })
    expect(s.t).toBeLessThan(40)
  })

  it('agrees with a sampler on ARD: the curve, the band and which bases are in use', () => {
    const model = bayesianGam(x, y, { k: 10, prior: 'ard' })
    const ep = fitOf('ep', model)
    const nuts = fitOf('nuts', model, { numChains: 1, numWarmup: 100, numDraws: 200, maxTreeDepth: 5 })
    const sweeps = session(bayesianGamMethod('ep', model, { fit: { gridSize: 20 } }), { steps: 3000, mode: 'eager' }).t
    // The posterior of the coefficients is one Gaussian vector (the likelihood's dense message), which settles in few sweeps (64 as scalars).
    expect(sweeps).toBeLessThan(40)
    const sd = model.standardisation.sd
    const worst = Math.max(...ep.mean.map((m, i) => Math.abs(m - (nuts.mean[i] ?? 0)) / sd))
    expect(worst).toBeLessThan(0.04)
    const width = (f: typeof ep): number => f.upper.reduce((a, u, i) => a + u - (f.lower[i] ?? 0), 0)
    expect(width(ep) / width(nuts)).toBeGreaterThan(0.8)
    expect(width(ep) / width(nuts)).toBeLessThan(1.2)
    for (const [j, v] of (ep.inclusion ?? []).entries())
      expect(Math.abs(v - (nuts.inclusion?.[j] ?? 0))).toBeLessThan(0.25)
  })
})

describe('variational message passing on the non-centred Bayesian GAM', () => {
  it('runs the penalty prior with log-normal hyperpriors (the dense function-scale factor) and keeps the curve of the exact reference', () => {
    const model = bayesianGam(x, y, { k: 10 })
    expect(bayesianGamMethods('penalty').applicable).toContain('vmp')
    const reference = bayesianGamReference(model, { gridSize: 20, points: 60 })
    const fit = fitOf('vmp', model, { damping: 0.3, tol: 1e-9 })
    const a = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(a.meanError).toBeLessThan(0.01)
    // The mean field keeps the correlations of the coefficients in the likelihood's dense message: the band is the reference's within a few percent.
    expect(a.bandRatio).toBeGreaterThan(0.93)
    expect(a.bandRatio).toBeLessThan(1.06)
    expect(fit.hyper.logSigma[0]).toBeCloseTo(reference.hyper.logSigma[0], 1)
    // The mean field is narrower than the exact posterior of the log smoothing parameter, never wider.
    const lam = fit.hyper.logLambda as readonly [number, number]
    expect(lam[1]).toBeLessThan(1.05 * reference.hyper.logLambda[1])
    expect(lam[1]).toBeGreaterThan(0.7 * reference.hyper.logLambda[1])
  })

  it('runs ARD (each coefficient an apply-product, assembled into one vector): the curve of EP, a band a fifth wider, the noise of EP', () => {
    // k = 6 keeps the quadrature of the scalar products quick, and 60 sweeps are at the fixed point (the curve moves less than 1e-3 after 50).
    const model = bayesianGam(x, y, { k: 6, prior: 'ard' })
    expect(bayesianGamMethods('ard').applicable).toContain('vmp')
    expect(bayesianGamMethods('ard').unavailable['vmp']).toBeUndefined()
    const ep = fitOf('ep', model)
    const vmp = fitOf('vmp', model, { damping: 0.3, tol: 1e-9, maxSweeps: 60 })
    const sd = model.standardisation.sd
    const worst = Math.max(...vmp.mean.map((m, i) => Math.abs(m - (ep.mean[i] ?? 0)) / sd))
    // The mean field is a little off the moment match of EP (a tenth of the response sd at worst here, 0.03 against the sampler at k = 10).
    expect(worst).toBeLessThan(0.1)
    const width = (f: typeof ep): number => f.upper.reduce((a, u, i) => a + u - (f.lower[i] ?? 0), 0)
    expect(width(vmp) / width(ep)).toBeGreaterThan(1)
    expect(width(vmp) / width(ep)).toBeLessThan(1.4)
    expect(Math.abs(vmp.hyper.logSigma[0] - ep.hyper.logSigma[0])).toBeLessThan(0.1)
  })

  it("opens ARD under VMP damped (0.5, 60 sweeps), because undamped it stalls with a collapsed noise; the other forms keep the method's own defaults", () => {
    const knobs = (prior: 'penalty' | 'ard'): Record<string, { default?: unknown }> =>
      (
        bayesianGamMethods(prior).space as unknown as {
          options: Record<string, { spaces: Record<string, { default?: unknown }> }>
        }
      ).options['vmp']?.spaces ?? {}
    expect([knobs('ard')['damping']?.default, knobs('ard')['maxSweeps']?.default]).toEqual([0.5, 60])
    expect([knobs('penalty')['damping']?.default, knobs('penalty')['maxSweeps']?.default]).toEqual([0, 200])
    expect(bayesianGamFormStarting('penalty')).toEqual({})
    expect(bayesianGamFormStarting('ard')['vmp']).toEqual({ damping: 0.5, maxSweeps: 60 })
  })

  it('runs ARD under VMP with no knobs given to the curve of EP and a finite noise, where the undamped run does not reach it', {
    timeout: 120_000,
  }, () => {
    const model = bayesianGam(x, y, { k: 6, prior: 'ard' })
    const ep = fitOf('ep', model)
    // no config: the form's starting knobs are the method's defaults
    const damped = session(bayesianGamMethod('vmp', model, { fit: { gridSize: 20 } }), { steps: 3000, mode: 'eager' })
    expect(damped.t).toBeLessThanOrEqual(60)
    const fit = damped.probe(damped.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
    const sd = model.standardisation.sd
    expect(Math.max(...fit.mean.map((m, i) => Math.abs(m - (ep.mean[i] ?? 0)) / sd))).toBeLessThan(0.1)
    expect(Math.abs(fit.hyper.logSigma[0] - ep.hyper.logSigma[0])).toBeLessThan(0.1)
    // The same run with the damping taken off does not settle on that noise. Where it goes depends on the last bits of exp and log
    // (arm64 and x64 builds of V8 differ in them): the log noise ran off to -46 and the run stopped on macOS, and wandered to
    // -0.8 and to +0.4 and +5 (100 sweeps, not converged) under a last-bit change of those functions, where EP has -1.65.
    const bare = fitOf('vmp', model, { damping: 0, maxSweeps: 100 })
    expect(Math.abs(bare.hyper.logSigma[0] - ep.hyper.logSigma[0])).toBeGreaterThan(0.3)
  })
})
