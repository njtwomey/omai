import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { MODELS } from '../../../testing/probprog.ts'
import { key } from '../../random/index.ts'
import { potentialEnergy } from '../density.ts'
import type { Model } from '../primitives.ts'
import { laplaceApproximation, laplaceFit } from './laplace.ts'

describe('probprog/engines/laplace against NumPyro and scipy', () => {
  for (const c of loadFixture('probprog/engines/laplace').cases) {
    it(c.name, () => {
      const model = MODELS[c.params['model'] as string] as Model
      const um = potentialEnergy(model, undefined)
      const fit = laplaceFit(um, new Float64Array(um.dim))
      if (c.fn === 'mode') expectMatches(fit.mode, c.output, c.rtol, c.atol, c.name)
      else if (c.fn === 'covariance') expectMatches(fit.covariance, c.output, c.rtol, c.atol, c.name)
      else {
        const post = laplaceApproximation(model, undefined, { key: key(0) })
        expectMatches(post.logEvidence?.value ?? Number.NaN, c.output, c.rtol, c.atol, c.name)
      }
    })
  }
})
