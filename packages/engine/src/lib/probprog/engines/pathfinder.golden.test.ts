// Pathfinder against BlackJAX's `pathfinder.approximate` (oracle/cases/probprog/engines/pathfinder.py): the moments of the
// unconstrained draws of the best normal on the path, within a fraction of the posterior sd (two approximations of one posterior
// from different starts), and the ELBO at least BlackJAX's.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { MODELS, weightedMoments } from '../../../testing/probprog.ts'
import { runAlgorithm } from '../../trace/index.ts'
import type { Model } from '../primitives.ts'
import { pathfinderAlgorithm } from './pathfinder.ts'

describe('Pathfinder against BlackJAX', () => {
  const golden = loadFixture('probprog/engines/pathfinder')
  for (const c of golden.cases.filter((x) => x.fn === 'moments')) {
    it(`${c.params['model']}: the moments of the draws are within a fraction of the posterior sd`, () => {
      const model = MODELS[c.params['model'] as string] as Model
      const dim = c.params['dim'] as number
      const expected = Array.from((c.output as { data: number[] }).data)
      const r = runAlgorithm(
        pathfinderAlgorithm(model, undefined, { maxIterations: 60, numDraws: 2000, elboDraws: 100 }),
        { seed: 0, maxSteps: 400 },
      )
      const { mean, sd } = weightedMoments(r.state.draws as Float64Array, r.state.logWeights as Float64Array, dim)
      for (let j = 0; j < dim; j++) {
        const refSd = expected[dim + j] ?? 1
        expect(Math.abs((mean[j] ?? 0) - (expected[j] ?? 0)), `mean ${j}`).toBeLessThan(0.25 * refSd)
        expect(Math.abs((sd[j] ?? 0) / refSd - 1), `sd ${j}`).toBeLessThan(0.35)
      }
      // The ELBO of the best normal on the path: at least BlackJAX's (it estimates it from other draws, so within noise).
      const elbo = golden.cases.find((x) => x.fn === 'elbo' && x.params['model'] === c.params['model'])
      const reference = (elbo?.output as { data: number[] } | undefined)?.data[0] ?? Number.NaN
      expect(r.state.elbo).toBeGreaterThan(reference - 0.3)
    })
  }
})
