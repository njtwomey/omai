import { describe, expect, it } from 'vitest'
import { MODELS } from '../../../testing/probprog.ts'
import { halfCauchy, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { clip, item, mul, square } from '../../tensor/index.ts'
import { analyse } from '../analysis.ts'
import { factor, type Model, observe, plate, sample } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { potentialEnergy } from '../density.ts'
import { diagnoseMode, laplaceApproximation, laplaceAt, laplaceEngine } from './laplace.ts'

describe('Laplace approximation', () => {
  it('is exact for a Gaussian posterior, evidence included', () => {
    const model = MODELS['normalMean'] as Model
    const exact = conjugateInference(model, undefined)
    const post = laplaceApproximation(model, undefined, { key: key(0) })
    expect(post.status).toBe('converged')
    expect(item(post.mean('mu'))).toBeCloseTo(item(exact.mean('mu')), 8)
    expect(item(post.variance('mu'))).toBeCloseTo(item(exact.variance('mu')), 8)
    expect(post.logEvidence?.value).toBeCloseTo(exact.logEvidence?.value ?? 0, 8)
    expect(post.logProb?.({ mu: item(exact.mean('mu')) })).toBeCloseTo(
      -0.5 * Math.log(2 * Math.PI * item(exact.variance('mu'))),
      8,
    )
  })

  it('ends with status failed on an improper posterior', () => {
    // The factor cancels the prior's curvature: the log density is flat.
    const flat: Model = () => {
      const x = sample('x', normal(0, 1))
      factor('cancel', mul(square(x), 0.5))
    }
    const post = laplaceApproximation(flat, undefined, { key: key(1), maxIterations: 50 })
    expect(post.status).toBe('failed')
    expect(String(post.diagnostics['reason'])).toMatch(/positive definite|starting point/)
    expect(laplaceEngine().applies(analyse(MODELS['mvn'] as Model, undefined)).applies).toBe(true)
  })
})

// The centred hierarchical posterior has a density that grows without bound as the scale goes to 0, so the optimiser runs to the
// floor of the scale: that is mathematically what the mode is, and the fit has to say that it is not a point of the posterior.
describe('Laplace diagnostics of the mode', () => {
  const Y = [28, 8, -3, 7, -1, 1, 18, 12]
  const SIGMA = [15, 10, 16, 11, 9, 11, 10, 18]
  const centred: Model = () => {
    const mu = sample('mu', normal(0, 5))
    const tau = sample('tau', halfCauchy(5))
    plate('school', Y.length, () => {
      const theta = sample('theta', normal(mu, tau))
      observe('y', normal(theta, SIGMA), Y)
    })
  }

  it('finds nothing wrong with a Gaussian posterior', () => {
    const post = laplaceApproximation(MODELS['normalMean'] as Model, undefined, { key: key(0) })
    expect(post.diagnostics['degenerateMode']).toBe(false)
    expect(post.diagnostics['messages']).toEqual([])
    expect(post.diagnostics['boundary']).toEqual([])
    expect(Number(post.diagnostics['conditionNumber'])).toBeLessThan(1e3)
  })

  it('says the mode of the centred eight schools is on the boundary tau -> 0', () => {
    for (const k of [0, 1, 2]) {
      const post = laplaceApproximation(centred, undefined, { key: key(k) })
      // The fit itself is unchanged: it ends, with the normal at the point the optimiser reached.
      expect(['completed', 'converged', 'failed']).toContain(post.status)
      const text = [post.diagnostics['messages'], post.diagnostics['reason']].flat().join(' ')
      expect(post.diagnostics['degenerateMode'] ?? true, `key ${k}: ${text}`).toBe(true)
      expect(text, `key ${k}`).toMatch(/boundary|ill-conditioned|not positive definite/)
      if (post.status !== 'failed') expect(String(post.diagnostics['boundary'])).toMatch(/tau/)
    }
  })

  it('reports the same for a scale floored where it enters the likelihood: the fit completes, and says so', () => {
    const floored: Model = () => {
      const mu = sample('mu', normal(0, 5))
      const tau = sample('tau', halfCauchy(5))
      plate('school', Y.length, () => {
        const theta = sample('theta', normal(mu, clip(tau, 1e-6, 1e6)))
        observe('y', normal(theta, SIGMA), Y)
      })
    }
    const post = laplaceApproximation(floored, undefined, { key: key(0) })
    expect(post.status).toBe('completed')
    expect(post.diagnostics['degenerateMode']).toBe(true)
    expect(String(post.diagnostics['boundary'])).toMatch(/tau \(unconstrained -1[34]\./)
    expect((post.diagnostics['messages'] as string[]).join(' ')).toMatch(/boundary/)
    expect((post.diagnostics['messages'] as string[]).join(' ')).toMatch(/ill-conditioned/)
  })

  it('reads a fit directly: an extreme log scale that did not converge, a singular covariance, a non-finite mode', () => {
    const um = potentialEnergy(centred, undefined)
    const mode = [0, -30, ...Array.from({ length: 8 }, () => 0)]
    const fit = laplaceAt(um, [0.5, 0.2, ...Array.from({ length: 8 }, () => 0.5)], {
      logDensity: 0,
      gradNorm: 0,
      iterations: 0,
      converged: false,
    })
    expect(diagnoseMode(um, fit).degenerate).toBe(false)
    const far = { ...fit, mode: { ...fit.mode, data: Float64Array.from(mode) } }
    const d = diagnoseMode(um, far as typeof fit)
    expect(d.boundary.length).toBe(1)
    expect(d.boundary[0]).toMatch(/^tau \(unconstrained -30\.0\)/)
    // The same point, reached with a vanishing gradient, is a mode, however extreme.
    expect(diagnoseMode(um, { ...far, converged: true } as typeof fit).boundary).toEqual([])
    const broken = { ...fit, mode: { ...fit.mode, data: Float64Array.from(mode.map((v, i) => (i === 3 ? Number.NaN : v))) } }
    const b = diagnoseMode(um, broken as typeof fit)
    expect(b.nonFinite).toBe(true)
    expect(b.degenerate).toBe(true)
  })
})
