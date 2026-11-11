// The Bayesian GAM with the smoothing parameter and the noise in the precision idiom, under expectation propagation, against
// the exact reference (two-dimensional quadrature over the smoothing parameter and the noise).
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
} from '../../message-passing/index.ts'
import { runAlgorithm, session } from '../../trace/index.ts'
import { factorGraphPosterior, factorGraphVerdict, toFactorGraph } from '../model/index.ts'
import {
  bayesianGam,
  bayesianGamAgreement,
  bayesianGamFit,
  bayesianGamMethod,
  bayesianGamMethods,
  bayesianGamReference,
} from './bayesian-gam.ts'

const x = Array.from({ length: 60 }, (_, i) => i / 59)
const noise = (i: number): number => 0.25 * Math.sin(i * 12.9898) * Math.cos(i * 78.233)
const y = x.map((v, i) => Math.sin(2 * Math.PI * v) + noise(i))
const model = bayesianGam(x, y, { k: 10, hyper: 'gammaSmooth' })
const reference = bayesianGamReference(model, { gridSize: 20, points: 40 })

const fitOf = (kind: string, config = {}) => {
  const s = session(bayesianGamMethod(kind, model, { fit: { gridSize: 20 } }), {
    steps: 3000,
    mode: 'eager',
    config: config as never,
  })
  return s.probe(s.t, 'gam/fit') as unknown as ReturnType<typeof bayesianGamFit>
}

describe('the gamma-smoothing Bayesian GAM', () => {
  it('compiles to one dense coefficient vector with a shared precision multiplier, and EP runs on it', () => {
    const c = toFactorGraph(model.definition.bind(model.data))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toContain('gaussian-mv-precision-scale')
    const g = compileGraph(c.graph as never, factorOperators())
    expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
    expect(bayesianGamMethods('penalty', 'gammaSmooth').applicable).toEqual(expect.arrayContaining(['ep']))
  })

  it('EP agrees with the exact reference: the curve, the band and the smoothing parameter', () => {
    const fit = fitOf('ep')
    const a = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(a.meanError).toBeLessThan(0.01)
    expect(a.bandRatio).toBeGreaterThan(0.95)
    expect(a.bandRatio).toBeLessThan(1.05)
    expect(Math.abs(a.logLambdaShift)).toBeLessThan(0.15)
    const lam = fit.hyper.logLambda as readonly [number, number]
    expect(lam[1]).toBeGreaterThan(0.8 * reference.hyper.logLambda[1])
    expect(lam[1]).toBeLessThan(1.2 * reference.hyper.logLambda[1])
    expect(fit.hyper.logSigma[0]).toBeCloseTo(reference.hyper.logSigma[0], 1)
  })

  it('the per-coefficient route (a scalar variable and a precision factor each) has the right curve and smoothing parameter, and a band within 20% (the dense route within 2%)', () => {
    const c = toFactorGraph(model.definition.bind(model.data), { sharedScale: 'elements' })
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'gaussian-gamma-precision').length).toBeGreaterThan(model.size)
    const g = compileGraph(c.graph as never, factorOperators())
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-7 }), { maxSteps: 600 })
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    const fit = bayesianGamFit(model, post, { gridSize: 20 })
    const a = bayesianGamAgreement(fit, reference, model.standardisation.sd)
    expect(a.meanError).toBeLessThan(0.01)
    expect(a.bandRatio).toBeGreaterThan(0.8)
    expect(a.bandRatio).toBeLessThan(1.25)
    expect(Math.abs(a.logLambdaShift)).toBeLessThan(0.3)
    expect(['converged', 'completed']).toContain(r.status)
  }, 30000)
})
