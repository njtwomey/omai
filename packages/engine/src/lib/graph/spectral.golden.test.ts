import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import {
  chebyshevFilter,
  dirichletEnergy,
  fiedlerVector,
  graphFourierTransform,
  graphWavelets,
  heatDiffusion,
  heatKernel,
  iterativeSpectralEmbedding,
  laplacian,
  normalisedLaplacian,
  spectralEmbedding,
} from './index.ts'

describe('graph signal processing against numpy, scipy and networkx', () => {
  for (const c of loadFixture('graph/spectral').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      const normalised = p['normalised'] === true
      const x = p['signal'] as number[][] | undefined
      const match = (y: Parameters<typeof expectMatches>[0]): void => expectMatches(y, c.output, c.rtol, c.atol, c.name)
      const choice = { normalised }
      switch (c.fn) {
        case 'frequencies':
          return match(graphFourierTransform(g, x ?? [], choice).frequencies)
        case 'gft':
          return match(graphFourierTransform(g, x ?? [], choice).coefficients)
        case 'heat':
          return match(heatKernel(g, 0.7, choice))
        case 'diffusion':
          return match(heatDiffusion(g, x ?? [], 0.7, choice))
        case 'chebyshev': {
          const l = normalised ? normalisedLaplacian(g) : laplacian(g)
          return match(chebyshevFilter(l, x ?? [], p['theta'] as number[], { lambdaMax: p['lambdaMax'] as number }))
        }
        case 'wavelets':
          return match(graphWavelets(g, p['scales'] as number[], { ...choice, signal: x ?? [] }))
        case 'dirichlet':
          return match(dirichletEnergy(g, x ?? [], choice))
        case 'fiedler': {
          // Eigenvectors are defined up to sign: align with the reference first.
          const want = (decode(c.output) as Tensor).data
          for (const method of ['dense', 'lanczos'] as const) {
            const v = fiedlerVector(g, { normalised, method })
            const sign = ddot(v, want) < 0 ? -1 : 1
            match(
              fromStorage(
                Float64Array.from(v, (t) => sign * t),
                [v.length],
                'float64',
              ),
            )
          }
          return
        }
        case 'embedding':
          return match(spectralEmbedding(g, 3))
        case 'iterative':
          return match(fromStorage(iterativeSpectralEmbedding(g, 3, { tol: 1e-9 }).values, [3], 'float64'))
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
