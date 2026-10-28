import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { rbfKernel } from '../kernels/index.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { relevanceVectorMachine } from './relevance-vector.ts'

describe('relevanceVectorMachine against scikit-learn ARDRegression on the kernel matrix', () => {
  for (const c of loadFixture('gp/relevance_vector').cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const xt = decode(c.inputs[2]) as Tensor
      const p = c.params as { lengthScale: number; fitIntercept: boolean }
      const est = relevanceVectorMachine.with({
        kernel: rbfKernel({ lengthScale: p.lengthScale }),
        fitIntercept: p.fitIntercept,
        tol: 1e-10,
        maxIter: 3000,
      })
      const fitted = est.fit({ x, y }, { explain: false })
      expectMatches(tensor(fitted.params.relevant), out['relevant'], 0, 0, `${c.name} relevant`)
      const dense = new Array<number>((x.shape[0] ?? 0) as number).fill(0)
      for (const [a, j] of fitted.params.relevant.entries()) dense[j] = fitted.params.coef[a] ?? 0
      expectMatches(tensor(dense), out['coef'], c.rtol, c.atol, `${c.name} coef`)
      expectMatches(fitted.params.intercept, out['intercept'], c.rtol, c.atol, `${c.name} intercept`)
      expectMatches(fitted.params.noise, out['noise'], c.rtol, c.atol, `${c.name} noise`)
      expectMatches(fitted.predict(xt), out['mean'], c.rtol, c.atol, `${c.name} mean`)
    })
  }
})
