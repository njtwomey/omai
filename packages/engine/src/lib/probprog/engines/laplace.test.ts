import { describe, expect, it } from 'vitest'
import { MODELS } from '../../../testing/probprog.ts'
import { normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { item, mul, square } from '../../tensor/index.ts'
import { analyse } from '../analysis.ts'
import { factor, type Model, sample } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { laplaceApproximation, laplaceEngine } from './laplace.ts'

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
