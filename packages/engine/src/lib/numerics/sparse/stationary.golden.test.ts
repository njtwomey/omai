import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as S from './index.ts'

describe('stationary iterations and the two-grid cycle against dense numpy and scipy.linalg', () => {
  for (const c of loadFixture('numerics/sparse/stationary').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const method = p['method'] as S.StationaryMethod
      const omega = p['omega'] as number
      const a = S.csr(t[0] as Tensor)
      const actual =
        c.fn === 'iterate'
          ? S.stationarySolve(a, t[1] as Tensor, { method, omega, rtol: 0, maxIterations: p['steps'] as number }).x
          : c.fn === 'matrix'
            ? S.iterationMatrix(a, { method, omega })
            : c.fn === 'radius'
              ? tensor(S.stationarySpectralRadius(a, { method, omega }))
              : c.fn === 'solution'
                ? S.stationarySolve(a, t[1] as Tensor, { method, omega, rtol: 1e-13, maxIterations: 5000 }).x
                : c.fn === 'twoGrid'
                  ? S.multigridSolve(a, t[1] as Tensor, {
                      shape: [15],
                      rtol: 0,
                      maxIterations: p['cycles'] as number,
                    }).x
                  : S.multigridSolve(a, t[1] as Tensor, { shape: [15], rtol: 1e-13 }).x
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})
