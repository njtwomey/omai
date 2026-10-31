import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import {
  constantKernel,
  type Kernel,
  linearKernel,
  maternKernel,
  periodicKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  sumKernel,
  whiteNoiseKernel,
} from '../kernels/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { logMarginalLikelihood } from './likelihood.ts'
import { gpRegression } from './regression.ts'

type Spec = { readonly k: string; readonly [name: string]: unknown }

/** The kernel a spec of the oracle describes. */
function kernelOf(spec: Spec): Kernel {
  const num = (name: string): number => spec[name] as number
  const parts = (): Kernel[] => (spec['parts'] as Spec[]).map(kernelOf)
  switch (spec.k) {
    case 'rbf':
      return rbfKernel({ lengthScale: spec['l'] as number | number[] })
    case 'matern':
      return maternKernel({ lengthScale: spec['l'] as number | number[], nu: num('nu') })
    case 'rq':
      return rationalQuadraticKernel({ lengthScale: num('l'), alpha: num('alpha') })
    case 'periodic':
      return periodicKernel({ lengthScale: num('l'), period: num('p') })
    case 'const':
      return constantKernel({ value: num('c') })
    case 'white':
      return whiteNoiseKernel({ noiseLevel: num('n') })
    case 'dot':
      return linearKernel({ sigma0: num('s0') })
    case 'sum':
      return sumKernel(...parts())
    case 'prod':
      return productKernel(...parts())
    default:
      throw new Error(`unknown kernel ${spec.k}`)
  }
}

describe('gpRegression against scikit-learn', () => {
  for (const c of loadFixture('gp/regression').cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const xt = decode(c.inputs[2]) as Tensor
      const p = c.params as { kernel: Spec; alpha: number; normalizeY: boolean; optimise: boolean }
      const kernel = kernelOf(p.kernel)
      const est = gpRegression.with({ kernel, alpha: p.alpha, normalizeY: p.normalizeY, optimise: p.optimise })
      if (!p.optimise) {
        // The likelihood and its gradient at the kernel's own hyperparameters, on the targets as fitted.
        const yv = toFlat(y)
        const mean = p.normalizeY ? yv.reduce((a, b) => a + b, 0) / yv.length : 0
        const sd = p.normalizeY ? Math.sqrt(yv.reduce((a, b) => a + (b - mean) ** 2, 0) / yv.length) : 1
        const yn = tensor(yv.map((v) => (v - mean) / sd))
        const lml = logMarginalLikelihood(kernel, x, yn, { alpha: p.alpha })
        expectMatches(lml.value, out['lml'], c.rtol, c.atol, `${c.name} lml`)
        expectMatches(tensor(Array.from(lml.gradient)), out['gradient'], c.rtol, c.atol, `${c.name} gradient`)
      }
      const fitted = est.fit({ x, y }, { explain: false })
      const { mean, std } = fitted.predictWithStd(xt)
      expectMatches(mean, out['mean'], c.rtol, c.atol, `${c.name} mean`)
      expectMatches(std, out['std'], c.rtol, c.atol, `${c.name} std`)
      if (out['cov'] !== undefined)
        expectMatches(fitted.predictCovariance(xt), out['cov'], c.rtol, c.atol, `${c.name} covariance`)
      if (p.optimise) {
        expectMatches(fitted.params.logMarginalLikelihood, out['lml'], c.rtol, c.atol, `${c.name} fitted lml`)
        expectMatches(tensor(fitted.params.theta), out['theta'], c.rtol, 1e-2, `${c.name} theta`)
      } else {
        expectMatches(fitted.params.logMarginalLikelihood, out['lml'], c.rtol, c.atol, `${c.name} lml`)
      }
    })
  }
})
