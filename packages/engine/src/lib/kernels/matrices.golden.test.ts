import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Value } from '../tensor/index.ts'
import {
  centerKernel,
  effectiveDimension,
  hsic,
  kernelAlignment,
  kernelPca,
  mmdSquared,
  type PairwiseKernelMetric,
  pairwiseKernels,
  targetAlignment,
} from './index.ts'

describe('kernel matrices against scikit-learn and the definitions', () => {
  for (const c of loadFixture('kernels/matrices').cases) {
    it(c.name, () => {
      const inputs = c.inputs.map(decode) as Value[]
      const [a, b, d, e] = inputs as [Value, Value, Value, Value]
      const out = c.output as Record<string, unknown>
      const p = c.params
      const match = (got: Value, expected: unknown = c.output, label = c.name): void =>
        expectMatches(got, expected, c.rtol, c.atol, label)
      switch (c.fn) {
        case 'center':
          return match(centerKernel(a))
        case 'centerCross':
          return match(centerKernel(a, b))
        case 'kernelPca': {
          const pca = kernelPca(a, { nComponents: p['nComponents'] as number })
          match(pca.eigenvalues, out['eigenvalues'], `${c.name} eigenvalues`)
          match(pca.eigenvectors, out['eigenvectors'], `${c.name} eigenvectors`)
          // Projections and transforms are sign-sensitive; scikit-learn's eigenvectors share the convention of eigh.
          match(pca.projections, out['projections'], `${c.name} projections`)
          match(pca.transform(a), out['projections'], `${c.name} transform of the training kernel`)
          return match(pca.transform(b), out['transform'], `${c.name} transform`)
        }
        case 'alignment':
          match(kernelAlignment(a, b), out['plain'], `${c.name} plain`)
          match(kernelAlignment(a, b, { centred: true }), out['centred'], `${c.name} centred`)
          match(kernelAlignment(a, d), out['linear'], `${c.name} linear`)
          match(targetAlignment(a, e), out['target'], `${c.name} target`)
          return match(targetAlignment(a, e, { centred: true }), out['targetCentred'], `${c.name} target centred`)
        case 'hsic':
          match(hsic(a, b), out['biased'], `${c.name} biased`)
          match(hsic(a, b, { estimator: 'unbiased' }), out['unbiased'], `${c.name} unbiased`)
          match(hsic(a, d), out['biasedDependent'], `${c.name} biased dependent`)
          match(hsic(a, d, { estimator: 'unbiased' }), out['unbiasedDependent'], `${c.name} unbiased dependent`)
          return match(hsic(a, d, { normalise: true }), out['normalised'], `${c.name} normalised`)
        case 'mmd':
          match(mmdSquared(a, b, d), out['biased'], `${c.name} biased`)
          match(mmdSquared(a, b, d, { estimator: 'unbiased' }), out['unbiased'], `${c.name} unbiased`)
          return match(mmdSquared(a, b, d, { estimator: 'linear' }), out['linear'], `${c.name} linear`)
        case 'effectiveDimension':
          return match(effectiveDimension(a, p['lam'] as number))
        case 'pairwise': {
          const gamma = p['gamma'] as number | undefined
          return match(pairwiseKernels(a, b, p['metric'] as PairwiseKernelMetric, gamma === undefined ? {} : { gamma }))
        }
        case 'pairwiseGram':
          return match(pairwiseKernels(a, undefined, p['metric'] as PairwiseKernelMetric))
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
