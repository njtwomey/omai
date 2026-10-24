import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { expectGpCase } from '../../testing/kernels.ts'
import type { Value } from '../tensor/index.ts'
import {
  additiveChi2Kernel,
  chi2Distance,
  chi2Kernel,
  constantKernel,
  diffusionKernel,
  histogramIntersectionKernel,
  jaccardKernel,
  rbfKernel,
  regularisedLaplacianKernel,
  setKernel,
  skewedChi2Kernel,
  spectrumKernel,
  subsequenceKernel,
  whiteNoiseKernel,
  wlSubtreeKernel,
} from './index.ts'

describe('structured kernels against scikit-learn, scipy and NetworkX', () => {
  for (const c of loadFixture('kernels/structured').cases) {
    it(c.name, () => {
      const inputs = c.inputs.map(decode) as Value[]
      const [x, y] = inputs as [Value, Value]
      const out = c.output as Record<string, unknown>
      const p = c.params
      const match = (got: Value, expected: unknown = c.output, label = c.name): void =>
        expectMatches(got, expected, c.rtol, c.atol, label)
      switch (c.fn) {
        case 'gp':
          return expectGpCase(
            c,
            p['kernel'] === 'white' ? whiteNoiseKernel({ noiseLevel: 0.3 }) : constantKernel({ value: 1.7 }),
          )
        case 'chi2':
          return match(chi2Kernel({ gamma: p['gamma'] as number }).k(x, y))
        case 'chi2Distance':
          return match(chi2Distance(x, y))
        case 'additiveChi2':
          return match(additiveChi2Kernel().k(x, y))
        case 'skewedChi2':
          return match(skewedChi2Kernel({ skewedness: p['skewedness'] as number }).k(x, y))
        case 'intersection':
          return match(histogramIntersectionKernel().k(x, y))
        case 'spectrum': {
          const strings = p['strings'] as string[]
          const k = p['k'] as number
          match(spectrumKernel(strings, undefined, { k }), out['raw'], `${c.name} raw`)
          match(spectrumKernel(strings, strings, { k }), out['raw'], `${c.name} raw (two sets)`)
          return match(spectrumKernel(strings, undefined, { k, normalise: true }), out['normalised'], `${c.name} norm`)
        }
        case 'subsequence': {
          const strings = p['strings'] as string[]
          const options = { n: p['n'] as number, lambda: p['lam'] as number }
          match(subsequenceKernel(strings, undefined, { ...options, normalise: false }), out['raw'], `${c.name} raw`)
          return match(subsequenceKernel(strings, strings, options), out['normalised'], `${c.name} normalised`)
        }
        case 'set': {
          const left = p['left'] as number
          const gamma = p['gamma'] as number
          return match(
            setKernel(rbfKernel({ lengthScale: 1 / Math.sqrt(2 * gamma) }), inputs.slice(0, left), inputs.slice(left)),
          )
        }
        case 'jaccard':
          return match(jaccardKernel(p['a'] as number[][], p['b'] as number[][]))
        case 'diffusion': {
          const g = graphFromCase(p)
          match(diffusionKernel(g, 0.6), out['combinatorial'], `${c.name} combinatorial`)
          return match(diffusionKernel(g, 1.3, { normalised: true }), out['normalised'], `${c.name} normalised`)
        }
        case 'regularisedLaplacian':
          return match(regularisedLaplacianKernel(graphFromCase(p), 0.7))
        case 'wl': {
          const graphs = (p['graphs'] as Record<string, unknown>[]).map((s) => graphFromCase(s))
          const labels = p['labels'] as string[][] | undefined
          return match(
            wlSubtreeKernel(graphs, undefined, {
              iterations: p['iterations'] as number,
              ...(labels === undefined ? {} : { nodeLabels: labels }),
            }),
          )
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
