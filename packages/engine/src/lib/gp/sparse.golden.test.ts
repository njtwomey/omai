import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { rbfKernel } from '../kernels/index.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import { type SparseApproximation, sparseBound, sparseGp } from './sparse.ts'

describe('sparseGp against dense numpy forms of the approximations', () => {
  for (const c of loadFixture('gp/sparse').cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const [x, y, xt, z] = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params as { approximation: SparseApproximation; lengthScale: number; noise: number }
      const kernel = rbfKernel({ lengthScale: p.lengthScale })
      expectMatches(
        sparseBound(p.approximation, kernel, x as Tensor, y as Tensor, z as Tensor, p.noise),
        out['bound'],
        c.rtol,
        c.atol,
        `${c.name} bound`,
      )
      const fitted = sparseGp
        .with({
          kernel,
          approximation: p.approximation,
          inducing: toArray(z as Tensor) as number[][],
          noise: p.noise,
          optimise: false,
        })
        .fit({ x: x as Tensor, y: y as Tensor }, { explain: false })
      const { mean, std } = fitted.predictWithStd(xt as Tensor)
      expectMatches(mean, out['mean'], c.rtol, c.atol, `${c.name} mean`)
      expectMatches(
        tensor(Array.from(toFlat(std), (v) => v * v)),
        out['variance'],
        c.rtol,
        c.atol,
        `${c.name} variance`,
      )
    })
  }
})
